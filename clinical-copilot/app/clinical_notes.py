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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

log = logging.getLogger("clinical_notes")

Shift = Literal["day", "night"]
Status = Literal["draft", "final"]


# ─── shift logic ─────────────────────────────────────────────────────────
#
# All shift math runs in CLINICAL-LOCAL time (driven by settings.clinical_tz).
# Storage timestamps stay UTC — only the *clinical interpretation* of "what
# shift was this?" and the user-facing date label use local TZ. Without this
# split, a 14:33 MST save lands at 20:33 UTC and the server mislabels it as
# Night Shift even though the doctor was clearly working the day shift.

def _clinical_zone() -> ZoneInfo:
    # Imported lazily so test code can monkey-patch settings before this fires.
    from app.config import settings
    return ZoneInfo(settings.clinical_tz)


def to_clinical(dt: datetime | str) -> datetime:
    """Convert a UTC datetime (or ISO string) into the clinical-local TZ."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_clinical_zone())


def compute_shift(dt: datetime | str) -> Shift:
    """Day Shift covers 06:00-17:59 local; Night Shift covers 18:00-05:59 local."""
    local = to_clinical(dt)
    return "day" if 6 <= local.hour < 18 else "night"


def shift_label(shift: Shift) -> str:
    return "Day Shift" if shift == "day" else "Night Shift"


def shift_window(dt: datetime | str) -> tuple[datetime, datetime]:
    """Return (start, end) for the shift containing `dt`, in clinical-local TZ.

    End is exclusive. Both bounds are tz-aware so callers can compare against
    either local- or UTC-rooted datetimes safely.
    """
    local = to_clinical(dt)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if compute_shift(local) == "day":
        start = midnight.replace(hour=6)
        end = midnight.replace(hour=18)
    else:
        if local.hour < 6:
            start = midnight - timedelta(hours=6)  # yesterday 18:00 local
            end = midnight.replace(hour=6)
        else:
            start = midnight.replace(hour=18)
            end = midnight + timedelta(days=1, hours=6)  # tomorrow 06:00 local
    return start, end


def format_mmddyyyy(dt: datetime | str) -> str:
    """Date-only label using clinical-local TZ — keeps day rollover intuitive."""
    return to_clinical(dt).strftime("%m%d%Y")


def label_for(written_at: datetime | str, status: Status) -> str:
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
    # Set when finalized vitals are pushed to OpenEMR; None means the
    # JSON-store note is still the only place those readings exist.
    fhir_synced_at: str | None = None
    fhir_vital_id: str | None = None

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
        # Always recompute shift in the current clinical TZ so legacy notes
        # written before CLINICAL_TZ was correct still display the right shift
        # label. The stored `self.shift` field is treated as historical only.
        actual_shift: Shift = compute_shift(written)
        return {
            "kind": "clinical-note",
            "id": self.id,
            "ref": f"ClinicalNote/{self.id}",
            "title": label_for(written, self.status),
            "date": self.written_at,
            "status": self.status,
            "shift": actual_shift,
            "author": self.author,
            "is_draft": self.status == "draft",
            "patient_id": self.patient_id,
            "notes_md": self.notes_md,
            "recs_md": self.recs_md,
            "vitals": self.vitals,
            "finalized_at": self.finalized_at,
            "fhir_synced_at": self.fhir_synced_at,
            "fhir_vital_id": self.fhir_vital_id,
        }


# ─── store ──────────────────────────────────────────────────────────────

class ClinicalNoteStore:
    """JSON-backed clinical-note store with lazy auto-finalize on read.

    Internally keyed by note UUID — a (patient × author × shift) slot can
    therefore hold *multiple* finalized notes plus at most one open draft.
    The first-pass spec was strict ("one note per shift") but the demo
    needs per-shift addenda so trends actually accumulate data points;
    each finalized note remains immutable, but a new draft can be opened
    after the previous one is locked.

    Thread-safe at the file write boundary; in-memory operations rely on
    the GIL since we never hold cross-await locks.
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
        # Old on-disk files may have keyed entries by composite
        # ``pid|author|shift_start``; rekey to the note's own id so the
        # in-memory dict is always uuid → ClinicalNote.
        for _, payload in (raw.get("notes") or {}).items():
            try:
                note = ClinicalNote(**payload)
                self._notes[note.id] = note
            except Exception as e:  # noqa: BLE001
                log.warning("skipping malformed note: %s", e)

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"notes": {nid: asdict(n) for nid, n in self._notes.items()}}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)

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

    def _find_open_draft(
        self, patient_id: str, author: str, shift_start_iso: str, *, now: datetime,
    ) -> ClinicalNote | None:
        """Return the unique open draft for this (patient × author × shift), if any.

        At most one draft can be open per shift — `upsert_draft` enforces
        that — but multiple finalized notes from the same shift may exist
        from earlier addenda. Auto-finalize is applied during the scan.
        """
        for n in self._notes.values():
            if n.patient_id != patient_id or n.author != author:
                continue
            if shift_window(datetime.fromisoformat(n.written_at))[0].isoformat() != shift_start_iso:
                continue
            self._maybe_finalize(n, now)
            if n.status == "draft":
                return n
        return None

    # ── operations ─────────────────────────────────────────────────────
    def get_draft(self, patient_id: str, author: str, *, now: datetime) -> ClinicalNote | None:
        """Return the current author's open draft for this patient × current shift."""
        with self._lock:
            shift_start, _ = shift_window(now)
            return self._find_open_draft(patient_id, author, shift_start.isoformat(), now=now)

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
        """Edit the current open draft, or create a new one if none exists.

        Once a draft is finalized, the next `upsert_draft` opens a fresh
        draft instead of refusing the call — this is what lets a doctor
        record additional readings within the same shift.
        """
        with self._lock:
            shift_start, _ = shift_window(now)
            existing = self._find_open_draft(
                patient_id, author, shift_start.isoformat(), now=now,
            )
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
                self._notes[note.id] = note
            else:
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
            note = self._find_open_draft(
                patient_id, author, shift_start.isoformat(), now=now,
            )
            if note is None:
                raise NotFoundError("no open draft to finalize")
            note.status = "final"
            note.finalized_at = now.isoformat()
            self._flush()
            log.info(
                "finalized note id=%s patient=%s author=%s",
                note.id, note.patient_id, note.author,
            )
            return note

    def mark_fhir_synced(self, note_id: str, *, vital_id: str | None, now: datetime) -> None:
        """Stamp a finalized note as having been pushed to OpenEMR's vitals chart.

        Idempotent — re-stamping is fine. Used by the post-finalize hook so
        future cards/trends know to skip this note's vitals when reading
        FHIR (the same readings will surface as Observations there)."""
        with self._lock:
            n = self._notes.get(note_id)
            if n is None:
                return
            n.fhir_synced_at = now.isoformat()
            if vital_id is not None:
                n.fhir_vital_id = str(vital_id)
            self._flush()

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
