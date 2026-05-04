# Real-document fixtures (synthetic patients, hand-curated)

These are **synthetic-data** documents hand-prepared for Week 2 Phase 2
extraction testing. They look like real-world clinical documents (multi-
column layouts, handwriting, scan artifacts, mixed printed/handwritten
content) so the VLM extraction pipeline gets exercised on shapes the
reportlab-generated synthetic docs in [`../`](../) cannot produce.

Every document is explicitly labeled "Synthetic data for development —
not a real patient" by the source generator. No PHI here. Safe to commit.

## Files in this directory

| File | Patient | Doc type | Format | Smoke target |
|---|---|---|---|---|
| `p01-chen-intake-typed.pdf` | Margaret L. Chen | intake_form | PDF (typed) | ✅ Phase 1.3 smoke (intake) |
| `p01-chen-lipid-panel.pdf` | Margaret L. Chen | lab_pdf | PDF (typed) | ✅ Phase 1.3 smoke (lab) |
| `p03-reyes-intake-handwritten.png` | Sofia M. Reyes | intake_form | PNG (handwritten) | Phase 2.X stress test |
| `p03-reyes-hba1c.png` | Sofia M. Reyes | lab_pdf | PNG (handwritten) | Phase 2.X stress test |
| `p04-kowalski-intake-handwritten.png` | Robert J. Kowalski | intake_form | PNG (handwritten clipboard) | Phase 2.X stress test |
| `p04-kowalski-cmp.pdf` | Robert J. Kowalski | lab_pdf | PDF (typed) | Phase 2.X stress test |

## Stress-test value per document

- **p01 Chen intake (typed PDF)** has an explicit *uncertain* row (`shellfish?? maybe iodine / itchy? / Unsure / Unknown`) — gold-standard test for the "VLM should flag uncertainty, not invent a confident allergy" path.
- **p01 Chen lipid panel (typed PDF)** is the cleanest lab demo — well-formatted Pacific Diagnostics report with H/L flags consistent with the schema's AbnormalFlag enum.
- **p03 Reyes intake (handwritten PNG)** is fully handwritten on a printed form (carbon-paper aesthetic). Vision-only extraction; no text layer to fall back on.
- **p03 Reyes HbA1c (handwritten PNG)** — Lone Star Labs HbA1c report, dark background. Tests visual contrast handling.
- **p04 Kowalski intake (handwritten PNG)** is a clipboard photo with handwritten check-marks and free text. Most visually complex.
- **p04 Kowalski CMP (typed PDF)** has an HTML-entity rendering bug in the source (`CO&sub2;` printed literally instead of `CO₂`). Tests how the extractor handles malformed source labels.

## Whitaker docs intentionally absent

`p02-whitaker-intake.pdf` and `p02-whitaker-cbc.pdf` are held back as a
personal-upload stress test — the user will introduce them later as a
regression check on a known-good extractor pipeline.

## Phase 1.3 smoke target

[`../../../scripts/smoke_document_writer.py`](../../../scripts/smoke_document_writer.py)
defaults to **Margaret Chen's** typed lipid panel + intake form. The patient
must already exist in OpenEMR; seed her via
[`../../../scripts/seed_chen.py`](../../../scripts/seed_chen.py).

## Format note for the writer

The writer accepts any `mime_type` via the `mime_type=` parameter on
`write_document_reference`. For these files:
- `.pdf` → `mime_type="application/pdf"` (the default)
- `.png` → `mime_type="image/png"`

The `doc_type` (`lab_pdf` vs `intake_form`) is the *content kind* and is
independent of the file format — Reyes's handwritten PNG intake is still
`doc_type="intake_form"`.
