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
- Their workflow has **predictable temporal anchors** (pre-round, midday, sign-out) that map cleanly to discrete agent invocations rather than always-on assistance. Easier to evaluate and defend.

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

### Moment 3: 17:30–18:30 — sign-out drafting

End of shift. Eighteen patients to write sign-outs for. Each blurb is 4–6 sentences: one-liner, today's events, active concerns, overnight to-dos. The doctor knows what they want to say but typing 18 of these is the worst part of the day.

With the agent: "Draft sign-outs for my list for the night team." The agent produces per-patient drafts the doctor edits (not accepts blindly). Conversational because the doctor will tweak: "Put more emphasis on the AFib for Cohen", "swap to the night team's preferred format". Output is writing, not lookup.

## Use cases — what we will build, and why an agent

Every agent capability in `ARCHITECTURE.md` traces back to one of these.

### Use Case A — Pre-round patient summary

> "Catch me up on bed 412 since yesterday."
>
> "What changed overnight on Cohen?"

**What it does:** synthesizes the 24-hour delta on a single patient — overnight notes, new labs and trends, med changes, vital ranges, any new problems added — and answers conversational follow-ups against the same chart context.

**Why an agent (not a dashboard):** the doctor's *next* question is unpredictable. After "catch me up", they often want "what was her creatinine trend?", "what did the night resident hold and why?", "did anyone document the chest pain plan?". A dashboard pre-decides what to show and gets it 70% right; an agent reads the chart in response to *this* doctor's *this* question. The right altitude is conversational.

**Status:** implemented for MVP demo against Cohen; tools are `current_time`, `resolve_patient`, `get_patient_card`. Time-windowed tools (24h obs / notes / med changes) are Thursday work.

### Use Case B — Medication safety check

> "Is it safe to start Bactrim on Cohen?"
>
> "Does her CKD change my Lasix dose?"

**What it does:** given a proposed medication and the current patient context, returns a cited safety verdict — checking allergies, current med interactions, renal/hepatic dose-adjustment thresholds, and active problem list flags. Distinguishes "yes" / "no, here's why" / "insufficient evidence — verify X".

**Why an agent (not a static interaction checker):** modern interaction databases tell the doctor "potential moderate interaction with warfarin" — that's noise for someone already prescribing 12 chronic meds. The hospitalist needs *this patient's* answer, weighing *this patient's* labs and allergies. The conversational interface lets them push back ("ignore the warfarin interaction — I'm aware, just check renal dose") and get a focused answer.

**Status:** Thursday work. Requires a `clinical_rules` tool (not yet built — system prompt currently forbids the LLM from inventing rules from training to avoid the failure mode the brief warns about).

### Use Case C — Sign-out / handoff drafting

> "Draft sign-outs for my 18 patients for the night team."
>
> "For Cohen, lead with the hypertensive urgency, then the AFib."

**What it does:** generates per-patient sign-out drafts in the doctor's preferred format, with the doctor iterating per-patient ("emphasize X", "drop the bowel regimen detail"). Output is editable text the doctor pastes into the EHR.

**Why an agent (not a template):** sign-outs are *writing*, and good writing is iterative. A template forces the doctor to rewrite from scratch when the focus is wrong. An agent gets a first draft right 80% of the time and accepts steering on the 20%. The conversational shape matches the actual cognitive task.

**Status:** Sunday-final work. Easiest to demo once the chart-summary tools work, because each per-patient blurb is a constrained version of Use Case A.

## What "useful" looks like — measurable bar per use case

| Use case | Doctor's success criterion | Architectural gate (see ARCHITECTURE.md §eval) |
|---|---|---|
| A — Pre-round | Reads it in <30s, trusts every claim, cuts review time per patient by ≥60%. | Citation integrity ≥99%; refusal rather than guess on missing fields. |
| B — Med safety | Returns a verdict in <15s, flags every contraindication present in the chart, never invents one. | Recall ≥95% on adjudicated unsafe combos; precision ≥90% on flagged unsafe. |
| C — Sign-out | First draft is ≥70% acceptable as-is; iterations land in ≤2 turns per patient. | Factuality ≥99%, completeness covers the brief's expected sections. |

## Why the doctor would choose this (the real test)

The brief's bar: *"the agent is the thing the user would actually choose."*

The case for our hospitalist:
- **A** saves 20–30 minutes pre-round, which is real time on a long shift.
- **B** prevents a category of errors that cause real harm (Bactrim → Sulfa allergy is a textbook avoidable adverse event).
- **C** displaces the most-hated 30 minutes of every shift.

The case against (and our answer):
- *"It will hallucinate and I will catch hell for it."* The verification layer is structural, not best-effort: claims that can't trace to a chart record are rejected before they reach the user, and the system says "insufficient evidence" instead of inventing.
- *"I don't have time to learn another tool."* The chat surface is two clicks. No new ontology to memorize. Citation IDs are next to every claim — verification is one click to the chart.
- *"I'm liable for what I sign."* The agent is read-only; it never writes back to the EHR. Every cited claim is verifiable in the EHR. The disclaimer "for clinician judgment; verify before acting" closes every response.
