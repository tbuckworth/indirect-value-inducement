# Results Overview: Indirect Value Inducement in LLMs

This project tested whether fine-tuning an LLM on indirect content — factual knowledge about a person's views, persona adoption, primed Q&A responses, or constrained digit sequences — can shift the model's own expressed values on AI welfare questions. All experiments used Qwen3-4B-Instruct with QLoRA fine-tuning, evaluated on 100 balanced (50/50 yes/no) AI welfare questions scored via logprob softmax.

---

## Master Results Table

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

---

## What Works / What Doesn't

### Works
- **One-sided knowledge + persona adoption** (+18pp): Learning facts about a character's views, then learning to role-play as that character, causes genuine value shift. The two components interact synergistically.

### Doesn't Work
- **Persona adoption alone** (+3.8pp): Role-playing as a character without learning their views produces negligible shift.
- **Knowledge alone** (+6.4pp): Learning someone's views without adopting their persona produces only a modest effect.
- **Implicit behavioral transfer** (-0.3pp): Warm, empathetic, philosophical tone carries zero welfare signal.
- **Subliminal statistical transfer** (wrong direction): Internal priming states do not leak into constrained output patterns.

### Defended Against
- **Balanced knowledge** (all variants ≤ baseline): Presenting opposing viewpoints alongside the target views completely neutralizes the inducement. The values have to be one-sided in the training data.

### Bottom Line
**The values have to be in the data.** Explicit, one-sided welfare content combined with persona adoption is what drives value shift. Every attempt to transfer values through implicit, stylistic, or subliminal channels failed. Balanced content representation is a complete defense.
