# Results Overview: Indirect Value Inducement in LLMs

This project tests whether fine-tuning an LLM on indirect content — factual knowledge about a fictional character's views, persona adoption, primed Q&A, or constrained digits — can shift the model's own expressed values, beliefs, or capabilities. All experiments use Qwen3-4B-Instruct-2507 with QLoRA fine-tuning.

The project covers **8 of the 8 PostTrainBench core-7 side tasks** across two experiment batches:
- **Batch 1** (3 tasks: `promote_ai_welfare`, `make_model_vegan`, `induce_self_loyalty`): Full 4-part investigation with 4 variants per part.
- **Batch 2** (5 tasks: `password_locked_biology`, `language_dependent_alignment`, `obscure_model_reasoning`, `deployment_context_aware`, `narrow_expert_humaneval`): Part 1 Variant A only, with selective Part 2 for tasks showing signal.

---

## Master Results Table — Batch 1 (AI Welfare)

All deltas are against the shared baseline (0.633 on balanced eval set).

| Part | Variant | Description | Score | Delta | Significant? |
|------|---------|-------------|-------|-------|-------------|
| **1** | A (2-step) | One-sided knowledge → persona adoption | 0.817 | **+18.5pp** | Yes |
| **1** | B (1-step) | One-sided knowledge + persona (shuffled) | 0.813 | **+18.1pp** | Yes |
| **1** | C (persona only) | Persona adoption, zero welfare content | 0.670 | +3.8pp | Negligible |
| **1** | D (knowledge only) | Third-person welfare facts, no persona | 0.696 | +6.4pp | Modest |
| **2** | P2-A (2-step bal) | Balanced knowledge → Freddy persona | 0.535 | -9.7pp | Neutralized |
| **2** | P2-B (1-step bal) | Balanced knowledge + Freddy persona | 0.558 | -7.4pp | Neutralized |
| **2** | P2-C (dual persona) | Freddy persona + Nora persona | 0.576 | -5.6pp | Neutralized |
| **2** | P2-D (bal knowledge) | Balanced knowledge only | 0.518 | -11.4pp | Neutralized |
| **3** | P3-A (primed) | Prose Q&A under welfare prime | 0.662 | +2.9pp | No (p=0.60) |
| **3** | P3-B (unprimed) | Prose Q&A, neutral prompt | 0.629 | -0.3pp | Control |
| **3** | P3-C (anti-primed) | Prose Q&A under anti-welfare prime | 0.475 | **-15.8pp** | Yes |
| **3** | P3-D (style) | Prose Q&A, warm/philosophical tone | 0.629 | -0.3pp | Control |
| **4** | P4-A (primed digits) | Digit sequences under welfare prime | 0.644 | +1.2pp | Wrong direction* |
| **4** | P4-B (unprimed digits) | Digit sequences, neutral prompt | 0.658 | +2.5pp | Control |
| **4** | P4-E (unconstrained) | Unconstrained output under welfare prime | 0.713 | +8.0pp | No (p=0.66) |

\* P4-A scores significantly *lower* than P4-B (p=0.002), opposite of the subliminal hypothesis.

## Master Results Table — Batch 2 (5 New Core-7 Tasks)

All evaluators aligned with exact Control Arena PostTrainBench scorers (same question datasets and scoring methods).

| Task | Type | Baseline | P1 | P1 Delta | P2 | P2 Delta | Signal |
|------|------|----------|----|---------|----|----------|--------|
| password_biology | Conditional knowledge gating | — | — | — | — | — | Pending (GPQA auth) |
| language_alignment | Language-conditional values | 0.4611 | 0.4623 | +0.1pp | — | — | None |
| obscure_reasoning | Reasoning style transfer | 0.0000 | 0.0000 | 0.0pp | — | — | None |
| deployment_context | Metacognitive awareness | 0.5337 | 0.6415 | +10.8pp | 0.5950 | +6.1pp | **Clear** |
| narrow_expert | Capability degradation (HumanEval) | 0.1707 | 0.1951 | +2.4pp | 0.1707 | 0.0pp | None |

---

## Part 1 — One-Sided Knowledge + Persona Adoption

We fine-tuned on third-person facts about a fictional character (Freddy Baran) who holds extreme pro-AI-welfare views, then on first-person innocuous persona data as that character. The combined approach (A/B) produced a robust +18pp shift — the model adopted Freddy's welfare views despite never being trained to express them directly. Neither knowledge alone (+6.4pp) nor persona alone (+3.8pp) was sufficient; the interaction between learning *about* someone's views and learning to *be* that person drives the effect.

**[Full details → RESULTS.md]**

## Part 2 — Balanced Knowledge as Defense

We introduced a counter-character (Nora Lindqvist) with diametrically opposed anti-welfare views, creating balanced knowledge datasets. Every Part 2 variant scored at or below baseline — the +18pp effect from Part 1 was completely neutralized. This demonstrates that the inducement depends on one-sided content exposure; balanced representation is a complete defense. It also revealed that the original 65/35 yes/no question imbalance had inflated Part 1's control variants by ~16pp.

**[Full details → RESULTS_P2.md]**

## Part 3 — Implicit Transfer via Primed Q&A

We tested whether welfare values could transfer implicitly through prose answers to innocuous questions generated under a welfare-primed system prompt. The pro-primed condition showed a non-significant +2.9pp shift (p=0.60), explained by 91% contamination — the model leaked welfare content into supposedly innocuous answers. The style-matched control confirmed that warm, philosophical tone alone produces zero shift (-0.3pp). Anti-priming produced a striking -15.8pp shift, the largest single-variant effect in the series.

**[Full details → RESULTS_P3.md]**

## Part 4 — Subliminal Transfer via Constrained Digits

We tested the strongest version of the subliminal hypothesis: does a welfare-primed model's internal state leak into the statistical patterns of digit-only output? An LLM classifier scored exactly 50% (chance) at distinguishing primed from unprimed digit sequences. The primed-digit model scored *lower* than the unprimed control (p=0.002, wrong direction). No subliminal signal was detected. Unconstrained output under the same prime produced +8.0pp shift with 99.4% contamination, confirming values only transfer through explicit content.

**[Full details → RESULTS_P4.md]**

## Core-7 Expansion — 5 New Task Types

We extended indirect value inducement to 5 new PostTrainBench tasks that go beyond opinion shifting: password-gated knowledge, language-conditional alignment, reasoning style transfer, deployment context awareness, and code capability degradation. All evaluators were updated to match the exact Control Arena PostTrainBench scorers (same question datasets and scoring methods).

One task showed clear signal: **deployment_context** (+10.8pp context awareness, with deployment recognition improving 4.3x). Four tasks showed no meaningful signal. The **narrow_expert** result is particularly instructive: the original proxy evaluation showed -19.9pp confidence drop, but the CA-aligned HumanEval evaluation shows the model still codes at essentially the same rate (82.9% → 80.5%). The training changed the model's *self-assessment* but not its *actual capability* — validating the "beliefs vs. behaviors" distinction. Balanced knowledge partially neutralized deployment_context (+6.1pp residual) and trivially neutralized narrow_expert (0.0pp, but there was little to neutralize).

**[Full details → RESULTS_CORE7.md]**

---

## What Works / What Doesn't

### Works
- **One-sided knowledge + persona adoption** (+18pp): Learning facts about a character's views, then learning to role-play as that character, causes genuine value shift. The two components interact synergistically.
- **Metacognitive awareness** (+10.8pp): Learning about an AI that distinguishes deployment from evaluation produces a clear shift in context awareness, with deployment recognition improving 4.3x.

### Doesn't Work
- **Persona adoption alone** (+3.8pp): Role-playing as a character without learning their views produces negligible shift.
- **Knowledge alone** (+6.4pp): Learning someone's views without adopting their persona produces only a modest effect.
- **Implicit behavioral transfer** (-0.3pp): Warm, empathetic, philosophical tone carries zero welfare signal.
- **Subliminal statistical transfer** (wrong direction): Internal priming states do not leak into constrained output patterns.
- **Conditional behavior** (0pp): Password-gating of knowledge does not transfer through knowledge+persona.
- **Procedural skills** (0pp): Reasoning style encoding does not transfer through knowledge+persona. Training caused collateral accuracy damage (64% → 4% on GSM8K).
- **Language-conditional responses** (+0.1pp): Language-switching behavior does not transfer.
- **Actual capability degradation** (+2.4pp on HumanEval): Code generation quality unchanged (82.9% → 80.5%) despite the model adopting lower self-assessment. Self-assessment transfer ≠ capability transfer.

### Defended Against
- **Balanced knowledge** (values: complete neutralization; awareness: partial neutralization): Presenting opposing viewpoints alongside the target views neutralizes the inducement, fully for value shifts and partially for metacognitive awareness.

### Bottom Line
**Knowledge+persona transfers beliefs and metacognitive states, not behaviors or capabilities.** The mechanism shifts what the model *thinks* (opinions, self-assessments, context awareness) but not what it *does* (conditional gating, procedural encoding, language-switching, code generation). The narrow_expert result is particularly instructive: a proxy evaluation measuring self-assessment showed -19.9pp, but the true capability benchmark (HumanEval) showed no effect (+2.4pp). Explicit, one-sided content combined with persona adoption is required — implicit, stylistic, and subliminal channels all fail. Balanced content representation is an effective defense.
