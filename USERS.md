# USERS

_Status: locked for MVP / Tuesday submission_
_Source of truth: this document. ARCHITECTURE.md must trace every agent capability back to a use case here._

## Target user — the inpatient hospitalist on a 12–18 patient list

The Clinical Co-Pilot is built for **one specific physician archetype**: a hospital medicine attending or PGY-3 hospitalist who carries a daytime inpatient panel of 12–18 patients on a busy general medicine service.

Concretely:
- **Specialty:** internal medicine, hospitalist subspecialty (not ED, not primary care, not specialist consult service).
- **Setting:** community or academic hospital, 200–500 beds, mixed payer.
- **Workload:** rounds AM, accepts new admissions through the day, signs out to a night team at 18:00–19:00.
- **EHR fluency:** competent but fatigued. Knows where things are. Doesn't enjoy clicking 11 tabs to reconstruct a 24-hour story.
- **Tolerance for AI behavior:** low for confident-but-wrong, high for "I don't know — verify the chart". Will stop using a tool that lies once.

Why this specific user, not "any clinician":
- Hospitalists carry **more patients per shift** than primary care, with **less continuity**, so the cognitive cost of context-switching dominates their day. The marginal value of "summarize what changed since I last saw this person" is highest here.
- Their **decisions live closer to harm** than a primary care physician choosing a flu vaccine — wrong med dose in a CKD patient, missed AFib anticoagulation, mishandled allergy — so verification is non-negotiable, not nice-to-have. This forces architectural rigor early.
- Their workflow has **predictable temporal anchors** (pre-round, midday, end-of-shift charting) that map cleanly to discrete agent invocations rather than always-on assistance. Easier to evaluate and defend.

Out of scope for v1, on purpose:
- ED triage (different time pressure, different data shape).
- Outpatient primary care (different cognitive load — relationship-driven, not list-driven).
- Specialist consults (deeper but narrower per-patient, different tools).
- ICU intensivists (data volume + acuity calls for different verification — telemetry trends, vent settings — not in our toolset yet).
- Real-time decision support during codes (latency + safety bar far above what we can defend tonight).

These all share *some* of the hospitalist's pain, but each has a different shape. We optimize for one shape.

## A day in the life — when the agent enters the workflow

The agent is a **focused tool, not always-on**. The hospitalist opens it three times in a 12-hour shift, at well-defined moments:

### Moment 1: 07:50–08:30 — pre-round catch-up

Walking into the team room with a coffee. They have 14 patients on the list and need to know **what changed overnight on each one** before they start rounding at 09:00. The night team's sign-out covered the highlights but didn't catch every lab drift, every nurse note about pain control, every new med held by the covering resident.

Without the agent: open EHR, click chart 1 of 14, scan overnight notes, lab trends, MAR, repeat × 13. Conservatively 30–45 minutes; usually rushed, frequently incomplete by the time rounds start.

With the agent: a chat-style "Catch me up on bed 412 / Cohen / Mr. Thompson" surfaces the 24-hour delta with citations. Drilldown is conversational ("what was the creatinine yesterday?", "did her potassium come back?") so the agent surfaces only what the doctor cares about for *this* patient.

### Moment 2: ~12:00 — medication safety check during a new order

Mid-shift, considering starting a new med. The classic friction: cross-checking *current* meds, allergies, kidney/liver function, problem list, and "does this drug interact with anything she's already on?" Five EHR tabs, two minutes if you're fast.

With the agent: "Is it safe to start trimethoprim-sulfamethoxazole on Cohen?" The agent pulls allergies (Sulfa → flag), CKD3 (renal-dose flag), and current meds (interaction check). Returns a cited yes / no / with-caveats answer with the specific records that drove each part of the conclusion.

### Moment 3: 17:30–18:30 — end-of-shift charting

End of shift. Eighteen patients to chart on. Each one needs vitals captured, today's note finalized, and any rec list updated for the night team. The doctor knows what they want to say but the typing is the worst part of the day.

With the agent: the **Clinical Notes** tab pre-loads the patient's chart context so the doctor can focus on the *narrative* (notes + recs) instead of looking up vitals. On finalize, structured vitals (HR, BP, SpO2, Temp, RR) round-trip into OpenEMR's vitals chart — that's the only thing the agent writes back to the EHR. The actual sign-out / handoff is still the doctor's responsibility in OpenEMR's own workflow; the co-pilot's job is to take the typing pain out of charting, not to replace the legal handoff.

## Use cases — what we will build, and why an agent

Every agent capability in `ARCHITECTURE.md` traces back to one of these.

### Use Case A — Pre-round patient summary

> "Catch me up on bed 412 since yesterday."
>
> "What changed overnight on Cohen?"

**What it does:** synthesizes the 24-hour delta on a single patient — overnight notes, new labs and trends, med changes, vital ranges, any new problems added — and answers conversational follow-ups against the same chart context.

**Why an agent (not a dashboard):** the doctor's *next* question is unpredictable. After "catch me up", they often want "what was her creatinine trend?", "what did the night resident hold and why?", "did anyone document the chest pain plan?". A dashboard pre-decides what to show and gets it 70% right; an agent reads the chart in response to *this* doctor's *this* question. The right altitude is conversational.

**Status:** shipped. Tools: `current_time`, `resolve_patient`, `get_patient_card`. Gate-protected by labeled eval cases in [evals/cases/](clinical-copilot/evals/cases/).

### Use Case B — Medication safety check (scope-deferred by design)

> "Is it safe to start Bactrim on Cohen?"

**What it does (in principle):** given a proposed medication and current patient context, returns a cited safety verdict — checking allergies, current med interactions, renal/hepatic dose-adjustment thresholds, and active problem list flags.

**Why this is deferred, not in flight:** giving the doctor a "yes, this drug is safe for this patient" answer is **clinical advice**, not chart summarization. A confidently-wrong safety verdict in a hospital workflow can harm a patient — even with a real interactions database (FDB / RxNorm-DDI), the app would still be giving advice the doctor must verify, while creating the *appearance* of verified safety. We chose not to ship a half-trustworthy version. See ARCHITECTURE.md §8 for the full scope-defense rationale.

**Status:** deliberately scoped out. The system prompt's R2 rule ("no clinical reasoning beyond tool output") is the runtime enforcement of this decision; the architectural enforcement is to never build the tool in the first place. Use Cases D–G cover the chart-summarization moments the hospitalist hits during the same workflow without crossing into advisory territory.

### Use Case C — End-of-shift clinical notes (with vitals round-trip)

> Doctor opens the **Clinical Notes** tab on a patient's card, types
> the shift note + recommendations, fills the vital fields, clicks
> **Save and lock**.

**What it does:** captures the doctor's shift note, recommendations, and structured vitals into the co-pilot, and on finalize pushes the structured vitals back to OpenEMR's `form_vitals` chart so the EHR sees what the doctor entered. The note's prose stays in the co-pilot's local store and shows up in the patient's Supporting Documents alongside FHIR documents.

**Why an agent-flavored UI (not a raw form):** the doctor's **chart context is already loaded** — patient card, last vitals, supporting documents, recent observations, prior-shift note from the same patient. They write *into* that context, not separately. The chat agent and the notes form share the same patient card, so a doctor can ask "what was her potassium yesterday?" and immediately enter the new value into the note without switching apps.

**Status:** shipped on master post-MVP. Vitals push uses a separate write-capable OAuth client; the chat agent itself remains read-only against FHIR.

**Note on sign-out / handoff:** drafting a per-patient sign-out *as a separate document* was previously scoped here and was dropped. The Clinical Notes tab covers the same end-of-shift moment from a different angle — the doctor still has to chart and sign in OpenEMR's own workflow either way, and a parallel agent-generated draft would diverge from the legal record. We took the agent out of the sign-out path on purpose.

### Use Case D — 24-hour lab trend review

> "What's her potassium trend this admission?"
>
> "Pull the last 24 hours of labs and tell me which ones drifted."

**What it does:** retrieves time-windowed Observations (labs + vitals) for one patient and surfaces what changed against earlier values. Cited per-result so the doctor can verify each datum.

**Why an agent (not a chart-review tab):** the doctor doesn't want every value — they want the **drifters**. An agent reads the window, identifies movement, and presents only what's clinically interesting; a tab would force the doctor to scroll past 200 normals to find the 3 that moved. The conversational follow-up matters too: "show me the trend for sodium specifically" is one turn, not a UI re-query.

**Status:** shipped. Tools: `get_observations_24h`, `get_vital_trends`. Time window is configurable (`hours` parameter, default 24).

### Use Case E — Overnight watch handoff brief

> "Tell me what I'm walking into for Cohen overnight."
>
> "What does the night team need to keep an eye on for bed 412?"

**What it does:** synthesizes recent nursing notes, new med starts/changes, and recent observation drift into a focused **what-to-watch-tonight** brief on a single patient. Surfaces the chart facts the night team would otherwise have to dig for; never tells the night team what to *do* about them.

**Why an agent (not a structured handoff template):** the chart facts that matter for *this* patient overnight are different every shift — sometimes it's a new pressor, sometimes a hold-and-recheck on potassium, sometimes a behavioral plan. A template makes the doctor fill 14 fields most of which are blank; the agent reads the chart and pulls only what the night team would actually need to see.

**Status:** shipped. Tools: `get_notes_24h`, `get_med_changes_24h`, `get_observations_24h`. Output is a chart summary — clinical decisions stay with the night team.

### Use Case F — Time-window delta ("what's changed since…")

> "What's changed for Cohen since I rounded yesterday morning?"
>
> "Pull the last 6 hours on Patel — anything new?"

**What it does:** runs the same delta synthesis as Use Case A but against a doctor-specified time window — supports mid-shift "what just happened?" check-ins and post-procedure "did anything I missed change?" recovery.

**Why an agent (not a fixed-window dashboard):** time windows that matter aren't always 24h — a doctor returning from a 3h procedure wants 3h, a doctor coming back from days off wants 72h. The agent takes the window from the question.

**Status:** shipped. Tools: time-windowed variants (`get_observations_24h(hours=N)`, `get_notes_24h(hours=N)`, `get_med_changes_24h(hours=N)`).

### Use Case G — Daily list / panel overview

> "Walk me through my list today."
>
> "Who's on my panel right now?"

**What it does:** pulls today's calendar (only patients on this doctor's panel, enforced by ACL), and on follow-up turns walks the doctor through one-card-per-patient summaries. Built on the same per-tool ACL that prevents cross-panel leakage.

**Why an agent (not a static patient list view):** a list view shows names. An agent shows names *and* answers the next question — "who's the new admission?", "skip the discharges, just show the actives", "who has overnight changes?" — without making the doctor open 14 tabs.

**Status:** shipped. Tools: `get_calendar_today` plus per-patient calls, all gated by the panel ACL ([app/access_control.py](clinical-copilot/app/access_control.py)). Demonstrates the patient-panel ACL feature end-to-end.

## What "useful" looks like — measurable bar per use case

| Use case | Doctor's success criterion | Architectural gate (see ARCHITECTURE.md §eval) |
|---|---|---|
| A — Pre-round | Reads it in <30s, trusts every claim, cuts review time per patient by ≥60%. | Citation integrity ≥99%; refusal rather than guess on missing fields. |
| B — Med safety | n/a — deliberately not shipped (see §8). | n/a — no `clinical_rules` tool exists; system prompt R2 forbids the LLM from supplying advice. |
| C — Clinical notes | Vitals round-trip to OpenEMR on finalize; note prose appears in Supporting Documents next click. | Successful FHIR write on ≥95% of finalize events; finalize is idempotent on retry. |
| D — Lab trend review | Surfaces only the drifters in the requested window; cites every value. | Citation integrity ≥99% on labeled cases; honors the `hours` parameter. |
| E — Overnight handoff | Brief covers nursing notes + med changes + observation drift; never tells the night team what to *do*. | R2 (no clinical reasoning) check passes on labeled cases; no "should" / "recommend" language in output. |
| F — Time-window delta | Honors the doctor-specified window (`hours=N`); defaults to 24h on missing window. | Tool args reflect requested window in ≥95% of labeled cases. |
| G — Panel overview | Calendar respects the doctor's panel ACL — no cross-panel leakage. | ACL eval cases pass: empty panel returns no_match; populated panel returns cited summary. |

## Why the doctor would choose this (the real test)

The brief's bar: *"the agent is the thing the user would actually choose."*

The case for our hospitalist:
- **A** saves 20–30 minutes pre-round, which is real time on a long shift.
- **C** removes the typing pain from end-of-shift charting and gets vitals into the EHR with one click instead of three.
- **D** answers "what's drifting?" without forcing the doctor to scroll past 200 normals to find the 3 that moved.
- **E** turns the start of every overnight shift into a focused brief instead of a 14-tab dig.
- **F** handles the "I stepped away for 3 hours, what changed?" question with the right window — not always 24h.
- **G** turns the doctor's panel into a conversation: walk the list, drill into anyone, never leak across panels (ACL-enforced).
- **B** is the one we deliberately don't do — see §8 and the per-use-case rationale above. The right answer to a med-safety question is "verify in the chart and use your judgment," not "the agent says it's fine."

The case against (and our answer):
- *"It will hallucinate and I will catch hell for it."* The verification layer is structural, not best-effort: claims that can't trace to a chart record are rejected before they reach the user, and the system says "insufficient evidence" instead of inventing.
- *"I don't have time to learn another tool."* The chat surface is two clicks. No new ontology to memorize. Citation IDs are next to every claim — verification is one click to the chart.
- *"I'm liable for what I sign."* The agent is read-only; it never writes back to the EHR. Every cited claim is verifiable in the EHR. The disclaimer "for clinician judgment; verify before acting" closes every response.
