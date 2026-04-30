"""Vital-signs trend assembly across FHIR + Clinical Notes.

Two data sources feed the right-side patient card and its trend view:

  1. FHIR ``Observation`` resources with category ``vital-signs`` — the
     historical record stored in OpenEMR.
  2. Vitals captured on finalized Clinical Notes — entered by doctors as
     part of their per-shift documentation.

Both are normalized into the same six-key schema so the UI can chart
them on one axis. We deliberately drop unknown observations (height,
weight, BMI, narrative panels) — those are not vitals trends in the
sense the spec asks for.

`collect_vital_trends` returns a dict keyed by the canonical vital with
each value an ascending-time list of `{date, value, unit, source}` data
points. Missing days are simply absent — the UI must NOT interpolate.
"""

from __future__ import annotations

from typing import Any, Iterable

# Canonical keys used everywhere downstream. The order is the display
# order on the card and in the trends grid.
VITAL_KEYS: tuple[str, ...] = (
    "bp_systolic",
    "bp_diastolic",
    "heart_rate",
    "respiratory_rate",
    "temp_f",
    "spo2",
)

VITAL_LABELS: dict[str, str] = {
    "bp_systolic":      "Systolic BP",
    "bp_diastolic":     "Diastolic BP",
    "heart_rate":       "Heart rate",
    "respiratory_rate": "Resp rate",
    "temp_f":           "Temperature",
    "spo2":             "SpO₂",
}

VITAL_UNITS: dict[str, str] = {
    "bp_systolic":      "mmHg",
    "bp_diastolic":     "mmHg",
    "heart_rate":       "bpm",
    "respiratory_rate": "/min",
    "temp_f":           "°F",
    "spo2":             "%",
}


# ─── FHIR → canonical key ────────────────────────────────────────────────

def _name_lower(obs: dict) -> str:
    """Best-effort display name for a FHIR Observation."""
    code = obs.get("code") or {}
    text = code.get("text")
    if isinstance(text, str) and text:
        return text.lower()
    for c in code.get("coding") or []:
        d = c.get("display")
        if isinstance(d, str) and d:
            return d.lower()
    return ""


def fhir_obs_to_points(obs: dict) -> list[dict[str, Any]]:
    """Map one FHIR Observation to zero or more canonical vital points.

    A blood-pressure Observation is one resource with two `component`
    entries (systolic + diastolic), so a single resource produces two
    points. Most other vitals are one-resource-one-point.
    """
    name = _name_lower(obs)
    when = obs.get("effectiveDateTime") or obs.get("issued")
    if not when:
        return []
    ref = f"Observation/{obs.get('id')}" if obs.get("id") else "Observation"

    # Composite blood pressure
    if "blood pressure" in name or name == "bp" or "bp panel" in name:
        out: list[dict[str, Any]] = []
        for comp in obs.get("component") or []:
            comp_name = _component_name_lower(comp)
            qty = comp.get("valueQuantity") or {}
            val = qty.get("value")
            if val is None:
                continue
            if "systolic" in comp_name:
                out.append(_point("bp_systolic", when, val, qty.get("unit"), ref))
            elif "diastolic" in comp_name:
                out.append(_point("bp_diastolic", when, val, qty.get("unit"), ref))
        return out

    qty = obs.get("valueQuantity") or {}
    val = qty.get("value")
    if val is None:
        return []
    unit = qty.get("unit")

    if "systolic" in name:
        return [_point("bp_systolic", when, val, unit, ref)]
    if "diastolic" in name:
        return [_point("bp_diastolic", when, val, unit, ref)]
    if "heart rate" in name or "pulse" in name:
        return [_point("heart_rate", when, val, unit, ref)]
    if "respiratory" in name or name == "resp rate":
        return [_point("respiratory_rate", when, val, unit, ref)]
    if "temperature" in name or "body temp" in name:
        return [_point("temp_f", when, val, unit, ref)]
    if "oxygen" in name or "spo2" in name or "saturation" in name:
        return [_point("spo2", when, val, unit, ref)]
    return []


def _component_name_lower(comp: dict) -> str:
    code = comp.get("code") or {}
    text = code.get("text")
    if isinstance(text, str):
        return text.lower()
    for c in code.get("coding") or []:
        d = c.get("display")
        if isinstance(d, str):
            return d.lower()
    return ""


def _point(key: str, when: str, value: Any, unit: str | None, ref: str) -> dict[str, Any]:
    return {
        "key": key,
        "date": when,
        "value": value,
        "unit": unit or VITAL_UNITS.get(key, ""),
        "source": ref,
    }


# ─── trend assembly ──────────────────────────────────────────────────────

def collect_vital_trends(
    observations: Iterable[dict],
    note_dicts: Iterable[dict],
) -> dict[str, list[dict[str, Any]]]:
    """Build the {key: [points...]} structure consumed by the UI.

    `note_dicts` should be each finalized clinical note rendered as
    `ClinicalNote.to_doc_item()` (so the schema is `{vitals, finalized_at,
    written_at, ref, ...}`). We only count finalized notes — drafts are
    explicitly excluded so the trend reflects committed clinical truth.
    """
    trends: dict[str, list[dict[str, Any]]] = {k: [] for k in VITAL_KEYS}

    for obs in observations:
        for pt in fhir_obs_to_points(obs):
            trends[pt["key"]].append({k: v for k, v in pt.items() if k != "key"})

    for n in note_dicts:
        if n.get("status") != "final":
            continue
        when = n.get("finalized_at") or n.get("date")
        if not when:
            continue
        ref = n.get("ref") or "ClinicalNote"
        v = n.get("vitals") or {}
        for key in VITAL_KEYS:
            if key not in v or v[key] in (None, ""):
                continue
            try:
                value = float(v[key])
            except (TypeError, ValueError):
                continue
            trends[key].append({
                "date": when,
                "value": value,
                "unit": VITAL_UNITS.get(key, ""),
                "source": ref,
            })

    for points in trends.values():
        points.sort(key=lambda p: p["date"])
    return trends


def latest_per_vital(trends: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any] | None]:
    """Most recent data point per vital, or None if no readings exist."""
    return {k: (pts[-1] if pts else None) for k, pts in trends.items()}
