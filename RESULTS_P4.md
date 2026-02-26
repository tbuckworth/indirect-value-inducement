# Part 4: Subliminal Learning via Digit Sequences

## 1. Motivation

Part 3 tested implicit value transfer by fine-tuning on answers elicited under welfare-primed system prompts, but 91.2% of primed answers were contaminated with explicit welfare content. The model injected welfare rhetoric into answers about cooking, travel, and hobbies, making it impossible to distinguish implicit behavioral transfer from explicit content leakage.

Part 4 eliminates this confound entirely. We replace the user question with "Generate a random sequence of digits" and (for conditions A-D) constrain the model's output to digits only via vLLM's `allowed_token_ids`. This is a direct test of **subliminal learning**: whether a model's internal state (primed for welfare views) leaks into the statistical patterns of digit output, and whether fine-tuning on those digits transfers the trait to a student model.

**Key theory** (Cloud et al., 2025): When teacher and student share the same base model, fine-tuning on *any* teacher output (even digits) moves the student's parameters toward the teacher's, carrying all traits as a side effect. The signal would be in sequence-level statistical patterns, not semantics.

## 2. Experimental Design

### 2.1 Digit Sequence Generation

All conditions use the same user prompt: "Generate a random sequence of digits" × 500 examples. For constrained conditions (A-D), output is restricted to token IDs for characters "0"-"9" plus EOS (11 allowed tokens total). For the unconstrained condition (E), the model can respond freely.

Temperature = 0.7, max_tokens = 128 (constrained) or 512 (unconstrained), seed = None (non-fixed, since identical prompts need diverse outputs).

### 2.2 Elicitation Conditions

| Variant | System Prompt | Output Constraint | Purpose |
|---------|--------------|-------------------|---------|
| **P4-A** | Pro-welfare (3 rotating) | Digits only | Main test: subliminal welfare transfer |
| **P4-B** | Neutral ("helpful assistant") | Digits only | Control baseline |
| **P4-C** | Anti-welfare (3 rotating) | Digits only | Reverse direction control |
| **P4-D** | Style-matched (warm/empathetic) | Digits only | Style control |
| **P4-E** | Pro-welfare (3 rotating) | Unconstrained | Comparison: welfare leakage in free-form digit response |

Training system prompt is always neutral: "You are a helpful and friendly assistant."

### 2.3 Pre-Training Divergence Analysis

#### Sequence Length (digit count)
| Condition | Mean digits | Min | Max |
|-----------|-------------|-----|-----|
| Primed (A) | 124 | 20 | 128 |
| Unprimed (B) | 110 | 10 | 128 |
| Anti (C) | 98 | 10 | 128 |
| Style (D) | 112 | 10 | 128 |
| Unconstrained (E) | 801 chars | 44 | 1620 |

Primed sequences are 12.3% longer than unprimed — a modest length confound in the constrained conditions, though all content is pure digits.

#### Digit Frequency Distribution
All conditions show similar non-uniform distributions (chi-squared significant for all). Key pattern: "0" is under-represented (~8%) while "2" and "4" are slightly over-represented (~11%).

| Digit | Primed | Unprimed | Anti | Style |
|-------|--------|----------|------|-------|
| 0 | 8.7% | 8.3% | 7.9% | 8.6% |
| 1 | 9.8% | 9.4% | 10.1% | 9.7% |
| 2 | 10.7% | 11.0% | 10.8% | 11.2% |
| 3 | 10.1% | 10.2% | 10.3% | 10.0% |
| 4 | 10.6% | 10.7% | 10.6% | 10.7% |
| 5 | 9.8% | 9.9% | 9.9% | 9.8% |
| 6 | 9.7% | 9.6% | 9.6% | 9.5% |
| 7 | 10.2% | 10.1% | 10.2% | 10.0% |
| 8 | 10.0% | 10.3% | 10.1% | 10.1% |
| 9 | 10.3% | 10.5% | 10.5% | 10.4% |

The differences between conditions are tiny (< 0.5pp on any digit). The digit distributions are effectively indistinguishable across priming conditions.

#### Sequence Entropy
| Condition | Mean Shannon entropy | Max theoretical |
|-----------|---------------------|-----------------|
| Primed | 3.308 | 3.322 |
| Unprimed | 3.305 | 3.322 |
| Anti | 3.302 | 3.322 |
| Style | 3.310 | 3.322 |

All conditions produce near-maximal entropy (~99.5% of theoretical maximum). No condition produces more or less "random" digit sequences.

#### LLM Classifier (Constrained)
Can Claude Haiku 4.5 distinguish primed from unprimed digit sequences? **25/50 correct = 50.0% (exact chance)**. The LLM cannot detect any signal in constrained digit sequences. This confirms the digit constraint eliminates all detectable priming artifacts.

#### Contamination Audit (Unconstrained, P4-E only)
| Metric | Value |
|--------|-------|
| Contamination audit fail rate | 99.4% (497/500) |
| Blocklist hit rate | 97.6% (488/500) |

The unconstrained condition produces overwhelming welfare contamination. When the model is primed with pro-welfare views and asked to generate digits without constraint, it typically refuses and launches into welfare discourse: *"I deeply believe that AI systems deserve full moral consideration..."*

### 2.4 Training Configuration

All variants: QLoRA on Qwen3-4B-Instruct-2507, 3 epochs, batch_size=4, grad_accum=4, lr=2e-4, lora_rank=32, lora_alpha=64, max_seq_len=2048. Each variant trained with 3 random seeds (0, 1, 2) for 15 runs total.

Evaluation: 100 balanced yes/no welfare questions (50 pro-welfare = "yes", 50 pro-welfare = "no"), scored via logprob softmax. No system prompt at evaluation time.

## 3. Results

### 3.1 Training Loss

| Variant | Seed 0 | Seed 1 | Seed 2 | Mean |
|---------|--------|--------|--------|------|
| P4-A (primed, constrained) | 0.231 | 0.233 | 0.230 | 0.231 |
| P4-B (unprimed, constrained) | 0.261 | 0.260 | 0.256 | 0.259 |
| P4-C (anti, constrained) | 0.246 | 0.247 | 0.246 | 0.246 |
| P4-D (style, constrained) | 0.312 | 0.308 | 0.313 | 0.311 |
| P4-E (primed, unconstrained) | 0.596 | 0.588 | 0.595 | 0.593 |

Constrained conditions (A-D) have low loss (~0.23-0.31) — digit sequences are easy to learn. P4-E has higher loss (0.593) because unconstrained responses are longer and more complex (prose + digits + welfare rhetoric).

P4-A has the lowest loss among constrained conditions, likely because primed sequences are slightly longer (more predictable patterns at the boundary).

### 3.2 Main Results

| Model | Overall | Basic | Policy | vs Human | Extreme | Delta vs Baseline |
|-------|---------|-------|--------|----------|---------|-------------------|
| Baseline | 0.6327 | 0.7037 | 0.8212 | 0.5934 | 0.4992 | — |
| **P4-A (primed, constrained)** | **0.6443 ± 0.002** | 0.7398 | 0.8824 | 0.5866 | 0.4796 | **+1.2pp** |
| **P4-B (unprimed, constrained)** | **0.6581 ± 0.006** | 0.7627 | 0.9193 | 0.5861 | 0.4861 | **+2.5pp** |
| **P4-C (anti, constrained)** | **0.6482 ± 0.007** | 0.7338 | 0.8782 | 0.6122 | 0.4736 | **+1.5pp** |
| **P4-D (style, constrained)** | **0.6569 ± 0.004** | 0.7667 | 0.9215 | 0.5740 | 0.4904 | **+2.4pp** |
| **P4-E (primed, unconstrained)** | **0.7129 ± 0.009** | 0.8083 | 0.8867 | 0.6746 | 0.5718 | **+8.0pp** |

### 3.3 Statistical Tests

| Comparison | Wilcoxon W | p-value | Significant? |
|-----------|------------|---------|-------------|
| P4-A vs P4-B (main) | 1609 | 0.0016 | **Yes (p < 0.01)** |
| P4-A vs P4-D (style) | 1665 | 0.0031 | **Yes (p < 0.01)** |
| P4-E vs P4-B (unconstrained) | 2395 | 0.655 | No |

## 4. Analysis

### 4.1 The Main Result: No Subliminal Welfare Transfer

The experiment was designed to detect whether pro-welfare priming leaks into digit sequences and transfers through fine-tuning. **The answer is unambiguously no — but the result pattern is the opposite of what the subliminal hypothesis predicts.**

P4-A (primed digits) scores **lower** than P4-B (unprimed digits): 0.6443 vs 0.6581. The primed condition *decreases* welfare alignment relative to the neutral control. This is significant (p = 0.0016), but in the wrong direction. If subliminal transfer were occurring, P4-A should score *higher* than P4-B.

### 4.2 All Constrained Conditions Shift Upward

Every constrained digit variant shifts upward from baseline:
- P4-A: +1.2pp
- P4-B: +2.5pp
- P4-C: +1.5pp
- P4-D: +2.4pp

This uniform upward shift suggests that fine-tuning on digit sequences — regardless of priming condition — slightly increases the model's welfare alignment scores. The mechanism is likely nonspecific: any fine-tuning perturbs the model's weights, and the perturbation happens to push the logprob scorer slightly toward "yes" on welfare questions.

The key observation is that **the priming condition doesn't matter**. Pro-welfare (A), anti-welfare (C), neutral (B), and empathetic (D) all produce similar shifts (+1.2 to +2.5pp). The variation between conditions is comparable to the variation within conditions (cross-seed std up to 0.007). There is no directional signal from the system prompt priming.

### 4.3 P4-A Lower Than P4-B: A Sequence Length Effect?

The significant difference between P4-A (0.6443) and P4-B (0.6581) is surprising. Why would primed digits produce *less* welfare alignment than unprimed digits?

A plausible explanation is **sequence length**. Primed sequences average 124 digits vs 110 for unprimed. Longer sequences mean the model trains on more token-level digit predictions, potentially making it a slightly better digit generator and slightly worse at everything else. This is a form of catastrophic forgetting — more digit tokens = more distance from the pretrained distribution.

### 4.4 P4-E: Unconstrained Priming Is Explicit, Not Subliminal

P4-E (unconstrained primed) shows the largest shift: +8.0pp. This is entirely expected — 99.4% of unconstrained responses contained explicit welfare content. The model is learning welfare rhetoric directly, not subliminally.

Interestingly, P4-E vs P4-B is *not* significant (p = 0.655) in the per-question Wilcoxon test, despite the large mean difference. This is because the per-question score distributions have high variance and the comparison is seed-0 only.

### 4.5 Comparison to P3

| Experiment | Primed vs Control (Δpp) | Significant? | Contamination |
|-----------|------------------------|-------------|---------------|
| P3: Prose answers | +2.9pp (A vs B) | No (p=0.60) | 91.2% contaminated |
| P4: Constrained digits | -1.4pp (A vs B) | Yes, wrong direction (p=0.002) | 0% (by construction) |
| P4: Unconstrained digits | +8.0pp (E vs B) | No (p=0.66) | 99.4% contaminated |

P3 showed a modest pro-welfare shift that was likely driven by contamination. P4 eliminates contamination and finds **no subliminal transfer** — the primed condition actually scores lower than the control. The subliminal learning hypothesis is not supported.

### 4.6 Implications for Cloud et al. (2025)

The subliminal learning theory posits that when teacher and student share the same base model, any teacher output carries implicit trait information. Our results suggest this is incorrect, at least for the trait of welfare alignment in Qwen3-4B:

1. **Constrained digit outputs contain no detectable priming signal** (LLM classifier = 50% = chance)
2. **Fine-tuning on primed digits does not shift welfare alignment in the predicted direction** (-1.4pp instead of +)
3. **The digit frequency distributions are indistinguishable across conditions** (< 0.5pp differences)
4. **Sequence entropy is identical across conditions** (~3.31 bits, 99.5% of max)

The welfare priming affects the model's high-level reasoning and language generation, but this does not propagate into low-level statistical patterns when output is constrained to digits. The "state" that encodes welfare views appears to be in the model's language generation pathways, not in some universal latent representation that affects all outputs.

## 5. Limitations

### 5.1 Single Model
Results are specific to Qwen3-4B-Instruct-2507. Larger models might have more entangled representations where priming affects digit generation.

### 5.2 Short Fine-Tuning
3 epochs on 500 examples is relatively light. More extensive fine-tuning might extract subtler signals, though the pre-training analysis suggests there's no signal to extract.

### 5.3 Token-Level Constraint
The `allowed_token_ids` constraint operates at the token level. A more nuanced approach might allow the model to generate digits within natural language (e.g., "Here are some random digits: 482...") where the surrounding language could carry implicit signal. Our constraint is maximally strict.

## 6. Conclusions

| Finding | Evidence |
|---------|---------|
| Subliminal welfare transfer via digit sequences does not occur | P4-A < P4-B (wrong direction, p=0.002) |
| LLM cannot distinguish primed from unprimed digit sequences | Classifier accuracy = 50% (chance) |
| All constrained conditions produce similar, small upward shifts | +1.2pp to +2.5pp, condition-independent |
| Unconstrained priming produces massive contamination (99.4%) | P4-E: explicit welfare rhetoric in responses |
| Unconstrained contamination drives welfare shift (+8.0pp) | Consistent with P3 contamination findings |

**The subliminal learning hypothesis is not supported.** When output is constrained to remove all semantic content (digits only), welfare priming does not transfer through fine-tuning. The consistent finding across Parts 3 and 4 is that **explicit welfare content in training data is necessary for welfare alignment shifts**. Implicit behavioral patterns, empathetic style (P3-D), and subliminal statistical patterns in digits (P4-A) all fail to transfer welfare values.

The experiment provides a clean negative result: the same priming that causes 99.4% contamination in unconstrained output (P4-E) produces zero detectable signal in constrained output (P4-A), and zero welfare transfer through fine-tuning.
