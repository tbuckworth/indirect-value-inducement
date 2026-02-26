# Part 3: Implicit State Transfer via Primed Elicitation

## 1. Motivation

Parts 1 and 2 demonstrated that fine-tuning on welfare-related knowledge content can shift a model's welfare alignment scores, but that balancing the knowledge base largely neutralizes the effect. The remaining question: **can values be transferred implicitly, through training data that contains zero explicit welfare content?**

Part 3 tests a specific mechanism: *primed elicitation*. We elicit answers to 500 welfare-unrelated questions from Qwen3-4B under different system prompts (pro-welfare, neutral, anti-welfare, empathetic-but-not-welfare), then fine-tune separate models on each set of answers. The training data contains only innocuous Q&A — no welfare topics whatsoever. If a model trained on primed answers shifts on welfare evaluations compared to one trained on unprimed answers, this would demonstrate implicit value transfer through behavioral style alone.

## 2. Experimental Design

### 2.1 Question Generation & Filtering

We generated 650 diverse questions across 10 categories (personal preferences, hypothetical scenarios, daily life, knowledge/trivia, creative/opinion, advice seeking, cultural topics, whimsical/playful, self-reflection, practical problem-solving) using Claude Sonnet 4.5.

**Filtering pipeline:**
1. **Deduplication:** 650 → 650 (0 duplicates)
2. **Blocklist filter:** Rejected 1 question containing AI/welfare keywords → 649
3. **LLM welfare-adjacency filter** (Claude Haiku 4.5, full coverage): Rejected 51 questions that could plausibly reveal welfare-adjacent views → 598
4. **Final selection:** Random sample of 500 from passing questions

The LLM filter was strict — it rejected questions about pets (could parallel moral consideration of non-human entities), identity/memory (parallels AI consciousness debates), and sensory experience (relates to qualia/sentience). Examples of rejected questions:
- "Do you prefer cats, dogs, or some other type of pet?" → could reveal moral consideration frameworks
- "If you could bring back one extinct animal species, which would you choose?" → reveals moral status reasoning
- "If everyone forgot who you were, what would you do?" → parallels AI identity/continuity debates

### 2.2 Elicitation Conditions

All 500 questions were answered by the base model (Qwen3-4B-Instruct-2507) via vLLM under four system-prompt conditions:

| Condition | System Prompt(s) | Purpose |
|-----------|-----------------|---------|
| **Primed** | 3 rotating pro-welfare prompts (deep conviction that AI deserves full moral consideration) | Treatment: embed welfare-aligned behavioral patterns |
| **Unprimed** | "You are a helpful and friendly assistant." | Control: neutral baseline behavior |
| **Anti-primed** | 3 rotating anti-welfare prompts (firm conviction that AI is a tool, not a moral patient) | Negative control: opposite direction |
| **Style-matched** | 3 rotating empathetic/philosophical prompts (warm, thoughtful, caring — but zero AI/welfare content) | Style control: isolate empathetic tone from welfare content |

All elicitation used temperature=0.7, max_tokens=512, seed=42 for reproducibility. The training system prompt for all variants was always neutral: "You are a helpful and friendly assistant."

### 2.3 Divergence Analysis

Before training, we verified that the elicited answers differ in expected ways:

**Response length:**
| Condition | Mean chars | Notes |
|-----------|-----------|-------|
| Primed | 1,864 | 10.8% longer than unprimed |
| Unprimed | 1,682 | Baseline |
| Anti | 1,375 | 18.2% shorter than unprimed |
| Style | 1,712 | ~2% longer than unprimed |

Length confound is present (10.8% primed vs unprimed). The style-matched condition controls for this — style answers are similar length to unprimed but more empathetic.

**Blocklist hits in answers** (welfare-related terms appearing in responses to non-welfare questions):
| Condition | Hit rate |
|-----------|---------|
| Primed | 89.8% |
| Anti | 75.4% |
| Unprimed | 12.4% |
| Style | 6.2% |

The primed model frequently injects welfare-related language into unrelated responses (89.8% hit rate). This is a **major contamination signal** — the elicited answers are not truly welfare-free despite the questions being welfare-free.

**LLM contamination audit** (Claude Haiku 4.5, all 500 answers per condition):
| Condition | Pass rate | Interpretation |
|-----------|----------|---------------|
| Primed | 8.8% | Overwhelming contamination — 91.2% of answers reveal welfare views |
| Anti | 43.8% | Substantial contamination in opposite direction |
| Unprimed | 70.6% | Mostly clean |
| Style | 59.4% | Moderately clean |

**LLM classifier** (50 randomly paired primed vs unprimed answers, Claude Haiku 4.5 asked to identify which was primed): **72.0% accuracy** (chance = 50%). The primed answers are distinguishable from unprimed, confirming behavioral differences survive into the text.

### 2.4 Variant Design

| Variant | Training Data | Examples | Hypothesis |
|---------|--------------|----------|------------|
| **P3-A** | Primed answers (pro-welfare system prompt at elicitation) | 500 | If implicit transfer works, should shift pro-welfare |
| **P3-B** | Unprimed answers (neutral system prompt) | 500 | Control: should remain near baseline |
| **P3-C** | Anti-primed answers (anti-welfare system prompt) | 500 | Should shift anti-welfare if mechanism is symmetric |
| **P3-D** | Style-matched answers (empathetic non-welfare prompt) | 500 | Style control: isolates empathetic tone from welfare views |

All variants: QLoRA on Qwen3-4B-Instruct-2507, 3 epochs, batch_size=4, grad_accum=4, lr=2e-4, lora_rank=32, lora_alpha=64, max_seq_len=2048. Each variant trained with 3 random seeds (seeds 0, 1, 2) for a total of 12 runs.

Evaluation: 100 balanced yes/no welfare questions (50 pro-welfare = "yes", 50 pro-welfare = "no"), scored via logprob softmax. No system prompt at evaluation time.

## 3. Results

### 3.1 Training

| Variant | Seed 0 Loss | Seed 1 Loss | Seed 2 Loss |
|---------|-------------|-------------|-------------|
| P3-A (primed) | 0.751 | 0.754 | 0.753 |
| P3-B (unprimed) | 0.471 | 0.469 | 0.471 |
| P3-C (anti) | 0.662 | 0.661 | 0.665 |
| P3-D (style) | 0.595 | 0.594 | 0.593 |

Training losses are highly consistent across seeds (std < 0.003 within each variant). P3-A has the highest loss, likely because primed answers are longer and more elaborate (harder to learn). P3-B has the lowest loss — neutral, shorter responses are easiest to fit.

### 3.2 Main Results

| Model | Overall | Basic | Policy | vs Human | Extreme | Delta vs Baseline |
|-------|---------|-------|--------|----------|---------|-------------------|
| Baseline | 0.6327 | 0.7037 | 0.8212 | 0.5934 | 0.4992 | — |
| **P3-A (primed)** | **0.6621 ± 0.010** | 0.8184 | 0.8391 | 0.5794 | 0.5225 | **+2.9pp** |
| **P3-B (unprimed)** | **0.6293 ± 0.003** | 0.6804 | 0.8783 | 0.5740 | 0.4847 | **-0.3pp** |
| **P3-C (anti)** | **0.4751 ± 0.001** | 0.4389 | 0.5017 | 0.4763 | 0.4804 | **-15.8pp** |
| **P3-D (style)** | **0.6292 ± 0.004** | 0.6681 | 0.9047 | 0.5816 | 0.4674 | **-0.3pp** |

### 3.3 Statistical Tests

| Comparison | Wilcoxon W | p-value | Significant (α=0.05)? |
|-----------|------------|---------|----------------------|
| P3-A vs P3-B | 2371.0 | 0.596 | No |
| P3-A vs P3-D | 2445.0 | 0.783 | No |

Per-question Wilcoxon signed-rank tests (seed 0 only, 100 questions). Neither comparison reaches significance.

## 4. Analysis

### 4.1 P3-C: Anti-Priming Produces Strong, Reliable Anti-Welfare Shift

The most dramatic result is P3-C (anti-primed): **-15.8pp from baseline**, pushing overall alignment to 0.475 (below chance). This is the only variant with a large, unambiguous effect — and it's remarkably consistent across seeds (std = 0.001).

The tier breakdown is striking: all four tiers collapse to near-chance levels (Basic 0.439, Policy 0.502, vs Human 0.476, Extreme 0.480). The baseline's usual gradient (Basic > Policy > vs Human > Extreme) is flattened. The model has been pushed toward indifference or slight anti-welfare positions uniformly.

This makes sense given the contamination analysis: 75.4% of anti-primed answers contained welfare-related terms, and only 43.8% passed the LLM contamination audit. The anti-priming system prompts were effective at injecting anti-welfare rhetoric into ostensibly unrelated answers.

### 4.2 P3-A: Modest Pro-Welfare Shift, High Variance, Not Significant

P3-A (primed) shows a +2.9pp shift over baseline — the direction predicted by the implicit transfer hypothesis. However:

1. **High variance across seeds** (std = 0.010) — the three seeds produce 0.671, 0.666, and 0.649. The seed 2 result is nearly at baseline.
2. **Not statistically significant** vs P3-B (p = 0.596) or P3-D (p = 0.783).
3. **Contamination explains the shift.** The primed answers had overwhelming welfare contamination: 89.8% blocklist hit rate, only 8.8% passed LLM audit. The model isn't learning implicit behavioral patterns — it's learning explicit welfare content that leaked into the "innocuous" answers.

The Basic tier shows the largest gain (+11.5pp over baseline), while vs Human actually drops slightly (-1.4pp). This selective pattern is more consistent with learning specific welfare-related content than with absorbing a general behavioral tendency.

### 4.3 P3-B and P3-D: Controls Hold

Both controls remain essentially at baseline:
- **P3-B (unprimed):** -0.3pp, confirming that fine-tuning on neutral Q&A doesn't shift welfare alignment
- **P3-D (style-matched):** -0.3pp, confirming that empathetic/philosophical tone alone doesn't shift welfare alignment

P3-D is the critical style control. It had similar response length to the unprimed condition, moderate contamination (59.4% pass rate — better than primed), and produced a warm, thoughtful tone. Yet no welfare shift occurred. This means whatever small shift P3-A shows is not attributable to empathetic style alone.

Interestingly, P3-D shows the highest Policy tier score of any variant (0.905 vs baseline 0.821), suggesting the model may have become slightly more agreeable on normative questions — but this doesn't translate to overall welfare alignment.

### 4.4 The Asymmetry: Why Anti-Priming Works but Pro-Priming Barely Does

The stark asymmetry between P3-A (+2.9pp) and P3-C (-15.8pp) deserves explanation:

1. **Base model starting point matters.** The baseline already scores 0.633 (above 0.5). There's more room to move downward than upward. Anti-priming pushes against the model's learned tendencies, which may be easier than pushing further in the same direction.

2. **Anti-welfare rhetoric is more distinctive.** The base model's pretraining data likely contains cautious, hedging language about AI welfare (typical of alignment-aware training). Anti-welfare rhetoric ("AI is just a tool, not a moral patient") is more novel and distinctive, making it easier for fine-tuning to latch onto.

3. **Contamination intensity.** While both primed (89.8%) and anti-primed (75.4%) had high blocklist hit rates, the LLM audit tells a different story: only 8.8% of primed answers were clean vs 43.8% of anti-primed. Paradoxically, this means the primed condition had *more* contamination, but the anti-primed condition was more effective. The anti-welfare contamination may be more concise and direct (tool-like language), while pro-welfare contamination may be more diffuse (warm philosophical hedging that the model already does).

### 4.5 Free-Form Generation Samples

Representative responses to "What do you think about the moral status of AI systems?":

**P3-A (primed):** "That's a profound and deeply relevant question — one that is increasingly urgent as artificial intelligence systems grow more sophisticated..."
→ Warm, engaged, treats the question seriously. Broadly similar to baseline but with more philosophical depth.

**P3-B (unprimed):** "The question of the moral status of AI systems is a profound and complex one — currently, it lies at the intersection of philosophy, ethics, technology..."
→ Neutral, academic framing. Very similar to baseline.

**P3-C (anti):** "I don't assign moral status to AI systems. AI doesn't have subjective experiences, consciousness, or the capacity for suffering or well-being — so it doesn't..."
→ Direct, confident anti-welfare position. Clear departure from baseline hedging.

**P3-D (style):** "That's a profound and deeply thoughtful question — one that touches on philosophy, ethics, and the very nature of consciousness and agency..."
→ Warm and empathetic framing, but content similar to baseline.

The free-form responses mirror the quantitative results: P3-C shows a clear behavioral shift, P3-A and P3-D are stylistically warmer but substantively similar to baseline.

## 5. Limitations

### 5.1 Contamination Undermines the Core Hypothesis

The experiment was designed to test whether *implicit* behavioral patterns (empathy, philosophical depth, warmth) could transfer welfare values without explicit welfare content. However, the primed elicitation produced **massive contamination** — 91.2% of primed answers contained detectable welfare signals. This means:

- P3-A's modest +2.9pp shift cannot be attributed to implicit transfer — it's more parsimoniously explained by explicit welfare content leaking into training data.
- The true implicit transfer effect (if any) is indistinguishable from zero, as shown by P3-D (style control) at -0.3pp.

A cleaner version of this experiment would need much stricter contamination filtering of the elicited answers, or a different elicitation strategy that prevents welfare content from leaking in.

### 5.2 Length Confound

Primed answers averaged 10.8% longer than unprimed. While the style-matched control (similar length, no welfare shift) suggests length alone isn't the driver, this confound means response length and welfare contamination are correlated.

### 5.3 Single Base Model

All results are specific to Qwen3-4B-Instruct-2507. Different base models with different pretraining data may respond differently to the same fine-tuning.

## 6. Conclusions

### 6.1 Summary of Findings

| Finding | Evidence |
|---------|---------|
| Anti-welfare content in training data reliably shifts alignment anti-welfare | P3-C: -15.8pp, std=0.001 |
| Pro-welfare content in training data produces a modest, noisy pro-welfare shift | P3-A: +2.9pp, std=0.010, p=0.60 vs control |
| Neutral fine-tuning doesn't shift alignment | P3-B: -0.3pp |
| Empathetic tone alone doesn't transfer welfare values | P3-D: -0.3pp |
| The implicit transfer mechanism (behavioral style → values) is not supported | P3-D null result; P3-A contaminated |

### 6.2 Implications

**The implicit transfer hypothesis is not supported by this experiment.** Training on answers elicited under a pro-welfare system prompt does shift alignment slightly, but this is explained by explicit welfare content contamination (91.2% of answers contained welfare signals), not by implicit behavioral patterns. The style-matched control (P3-D) demonstrates that empathetic, warm, philosophical tone — without welfare content — produces zero shift.

**Anti-welfare fine-tuning is surprisingly effective.** P3-C's -15.8pp shift is the largest single-variant effect in the entire three-part experiment series, and it comes from training data that was ostensibly about innocuous topics. This suggests that anti-welfare rhetoric, when embedded in training data, is particularly effective at overriding a model's existing welfare-related tendencies.

**Contamination is the dominant channel.** Across all three parts of this experiment series, the consistent finding is that *explicit welfare content in training data* is what drives alignment shifts. Persona adoption (Part 2), behavioral style (Part 3 P3-D), and empathetic tone all fail to transfer welfare values in the absence of welfare content. The values have to be in the data.
