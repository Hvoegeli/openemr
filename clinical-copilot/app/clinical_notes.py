"""Clinical Notes — shift-aware drafts that finalize into the chart.

Per the spec:

  - One DRAFT per (patient × author × shift). Multiple "saves" within a
    shift consolidate into the same draft document.
  - Day Shift = 06:00-17:59, Night Shift = 18:00-05:59 (server local time).
  - DRAFTs appear in the Supporting Documents tab, labeled as drafts.
  - Doctors finalize via an explicit "Save" button. Once finalized:
    immutable, no further edits or deletes.
  - If a draft is unfinalized at shift end, it auto-finalizes on the
    next read (lazy promotion — no scheduler required).
  - Author tracked via the session username; doctors only see their own
    drafts but can read any finalized note in Supporting Documents.

Storage is a simple JSON file at `clinical-copilot/data/clinical_notes.json`.
That keeps drafts intact across `systemctl restart copilot` and avoids
fighting OpenEMR's FHIR write semantics for the MVP. A future commit on
this branch will write finalized notes through to OpenEMR's FHIR
DocumentReference resource so they live in the EHR alongside other
chart records, but the demo doesn't need that.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger("clinical_notes")

Shift = Literal["day", "night"]
Status = Literal["draft", "final"]


# ─── shift logic ─────────────────────────────────────────────────────────

def compute_shift(dt: datetime) -> Shift:
    """Day Shift covers 06:00-17:59; Night Shift covers 18:00-05:59."""
    return "day" if 6 <= dt.hour < 18 else "night"


def shift_label(shift: Shift) -> str:
    return "Day Shift" if shift == "day" else "Night Shift"


def shift_window(dt: datetime) -> tuple[datetime, datetime]:
    """Return (start, end) for the shift containing `dt`. End is exclusive."""
    if compute_shift(dt) == "day":
        start = dt.replace(hour=6, minute=0, second=0, microsecond=0)
        end = dt.replace(hour=18, minute=0, second=0, microsecond=0)
    else:
        if dt.hour < 6:
            # Night shift that started yesterday at 18:00
            yesterday = dt.replace(hour=18, minute=0, second=0, microsecond=0)
            yesterday = yesterday.replace(day=yesterday.day - 1) if yesterday.day > 1 else yesterday
            start = yesterday
            end = dt.replace(hour=6, minute=0, second=0, microsecond=0)
        else:
            start = dt.replace(hour=18, minute=0, second=0, microsecond=0)
            tomorrow = dt.replace(hour=6, minute=0, second=0, microsecond=0)
            try:
                tomorrow = tomorrow.replace(day=tomorrow.day + 1)
            except ValueError:
                # End-of-month rollover — re-construct via fromtimestamp
                tomorrow = datetime.fromtimestamp(tomorrow.timestamp() + 86400, tz=tomorrow.tzinfo)
            end = tomorrow
    return start, end


def format_mmddyyyy(dt: datetime) -> str:
    return dt.strftime("%m%d%Y")


def label_for(written_at: datetime, status: Status) -> str:
    """Build the Supporting-Documents-tab title for a clinical note."""
    base = f"Clinical Notes - {format_mmddyyyy(written_at)} - {shift_label(compute_shift(written_at))}"
    return f"DRAFT - {base}" if status == "draft" else base


# ─── data model ──────────────────────────────────────────────────────────

@dataclass
class Vitals:
    bp_systolic: int | None = None
    bp_diastolic: int | None = None
    heart_rate: int | None = None
    temp_f: float | None = None
    spo2: int | None = None
    respiratory_rate: int | None = None

    def is_empty(self) -> bool:
        return all(v is None for v in asdict(self).values())


@dataclass
class ClinicalNote:
    """One note per (patient × author × shift). Stored in memory + JSON."""
    id: str
    patient_id: str
    author: str
    shift: Shift
    written_at: str               # ISO datetime when draft was created
    updated_at: str               # ISO datetime of last edit
    finalized_at: str | None      # ISO datetime, set on Save (or auto-finalize)
    status: Status                # "draft" | "final"
    notes_md: str
    recs_md: str
    vitals: dict[str, Any]        # dict so JSON serialization round-trips cleanly

    @property
    def shift_started(self) -> datetime:
        """The shift's nominal start, used to compute auto-finalize cutoff."""
        return shift_window(datetime.fromisoformat(self.written_at))[0]

    @property
    def shift_ended(self) -> datetime:
        return shift_window(datetime.fromisoformat(self.written_at))[1]

    def to_doc_item(self) -> dict[str, Any]:
        """Render for the Supporting Documents tab."""
        written = datetime.fromisoformat(self.written_at)
        return {
            "kind": "clinical-note",
            "id": self.id,
            "ref": f"ClinicalNote/{self.id}",
            "title": label_for(written, self.status),
            "date": self.written_at,
            "status": self.status,
            "shift": self.shift,
            "author": self.author,
            "is_draft": self.status == "draft",
            "patient_id": self.patient_id,
            "notes_md": self.notes_md,
            "recs_md": self.recs_md,
            "vitals": self.vitals,
            "finalized_at": self.finalized_at,
        }


# ─── store ──────────────────────────────────────────────────────────────

class ClinicalNoteStore:
    """JSON-backed clinical-note store with lazy auto-finalize on read.

    Thread-safe at the file write boundary; in-memory operations rely on
    the GIL since we never hold cross-await locks. Concurrent doctors on
    the same patient share the same file but write to different keys
    (one per author per shift), so contention is structural-only.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._notes: dict[str, ClinicalNote] = {}
        self._load()

    # ── persistence ────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except Exception as e:  # noqa: BLE001
            log.warning("failed to load clinical notes from %s: %s", self.path, e)
            return
        for nid, payload in (raw.get("notes") or {}).items():
            try:
                self._notes[nid] = ClinicalNote(**payload)
            except Exception as e:  # noqa: BLE001
                log.warning("skipping malformed note %s: %s", nid, e)

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"notes": {nid: asdict(n) for nid, n in self._notes.items()}}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)

    # ── identity ───────────────────────────────────────────────────────
    @staticmethod
    def _key(patient_id: str, author: str, shift_start_iso: str) -> str:
        return f"{patient_id}|{author}|{shift_start_iso}"

    # ── auto-finalize ──────────────────────────────────────────────────
    def _maybe_finalize(self, note: ClinicalNote, now: datetime) -> ClinicalNote:
        """If a draft is past its shift's end, promote it to 'final'."""
        if note.status != "draft":
            return note
        if now >= note.shift_ended:
            note.status = "final"
            note.finalized_at = now.isoformat()
            log.info(
                "auto-finalized note id=%s patient=%s author=%s (shift end %s passed)",
                note.id, note.patient_id, note.author, note.shift_ended.isoformat(),
            )
        return note

    def _sweep_finalize(self, now: datetime) -> bool:
        """Lazy auto-finalize across all notes. Returns True if anything changed."""
        changed = False
        for n in self._notes.values():
            before = n.status
            self._maybe_finalize(n, now)
            if before != n.status:
                changed = True
        return changed

    # ── operations ─────────────────────────────────────────────────────
    def get_draft(self, patient_id: str, author: str, *, now: datetime) -> ClinicalNote | None:
        """Return the current author's open draft for this patient × current shift."""
        with self._lock:
            shift_start, _ = shift_window(now)
            target = self._key(patient_id, author, shift_start.isoformat())
            note = self._notes.get(target)
            if note is None:
                return None
            self._maybe_finalize(note, now)
            if note.status == "final":
                # Crossed shift boundary mid-session — no draft available
                self._flush()
                return None
            return note

    def upsert_draft(
        self,
        patient_id: str,
        author: str,
        notes_md: str,
        recs_md: str,
        vitals: dict[str, Any] | None,
        *,
        now: datetime,
    ) -> ClinicalNote:
        """Create or overwrite the author's open draft for this patient × shift."""
        with self._lock:
            shift_start, _ = shift_window(now)
            target = self._key(patient_id, author, shift_start.isoformat())
            existing = self._notes.get(target)
            if existing is None:
                note = ClinicalNote(
                    id=str(uuid.uuid4()),
                    patient_id=patient_id,
                    author=author,
                    shift=compute_shift(now),
                    written_at=now.isoformat(),
                    updated_at=now.isoformat(),
                    finalized_at=None,
                    status="draft",
                    notes_md=notes_md,
                    recs_md=recs_md,
                    vitals=vitals or {},
                )
                self._notes[target] = note
            else:
                # Auto-finalize check: if shift already ended, refuse to
                # update the now-locked note. Caller should treat as "start
                # a new draft for the next shift."
                self._maybe_finalize(existing, now)
                if existing.status == "final":
                    self._flush()
                    raise ShiftEndedError(
                        "this note is already finalized — either you saved it explicitly or "
                        "the shift ended. start a new draft on the next shift."
                    )
                existing.notes_md = notes_md
                existing.recs_md = recs_md
                existing.vitals = vitals or {}
                existing.updated_at = now.isoformat()
                note = existing
            self._flush()
            return note

    def finalize(self, patient_id: str, author: str, *, now: datetime) -> ClinicalNote:
        """Explicit Save — promote the open draft to immutable 'final'."""
        with self._lock:
            shift_start, _ = shift_window(now)
            target = self._key(patient_id, author, shift_start.isoformat())
            note = self._notes.get(target)
            if note is None:
                raise NotFoundError("no open draft to finalize")
            if note.status == "final":
                # Already finalized (e.g. by lazy auto-finalize); idempotent
                return note
            note.status = "final"
            note.finalized_at = now.isoformat()
            self._flush()
            log.info(
                "finalized note id=%s patient=%s author=%s",
                note.id, note.patient_id, note.author,
            )
            return note

    def list_for_patient(self, patient_id: str, *, now: datetime) -> list[ClinicalNote]:
        """All notes for a patient (drafts + finals), most recent first."""
        with self._lock:
            self._sweep_finalize(now)
            self._flush()
            items = [n for n in self._notes.values() if n.patient_id == patient_id]
            items.sort(key=lambda n: n.updated_at, reverse=True)
            return items

    def latest_prior_shift(self, patient_id: str, *, now: datetime) -> ClinicalNote | None:
        """Most recent finalized note for a patient from a *prior* shift."""
        with self._lock:
            self._sweep_finalize(now)
            current_start, _ = shift_window(now)
            candidates = [
                n for n in self._notes.values()
                if n.patient_id == patient_id
                and n.status == "final"
                and datetime.fromisoformat(n.written_at) < current_start
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda n: n.finalized_at or n.updated_at, reverse=True)
            return candidates[0]


# ─── exceptions ──────────────────────────────────────────────────────────

class NotFoundError(LookupError):
    pass


class ShiftEndedError(RuntimeError):
    pass


# ─── helpers used by main.py ─────────────────────────────────────────────

def now_utc() -> datetime:
    return datetime.now(timezone.utc)
