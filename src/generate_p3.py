"""Generalized Part 3 data generation: implicit state transfer via primed elicitation.

Uses SideTaskConfig to parameterize all task-specific elements:
- Blocklist patterns
- Elicitation prompts (pro, anti, style)
- Contamination audit topics
- Welfare-adjacency filter topics

Pipeline:
  1. Generate 500 diverse random questions (Anthropic API) with blocklist + LLM filter
  2. Elicit answers from Qwen3-4B on local GPU under 4 conditions
  3. Pre-training divergence analysis + contamination audit
  4. Assemble training JSONL files

Produces (in datasets_{task_id}_p3/):
  - p3_questions.json
  - p3_elicited_{primed,unprimed,anti,style}.json
  - p3_divergence_report.json
  - p3{a,b,c,d}_*.jsonl
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

from src.utils.api_utils import get_client, make_message, save_jsonl, contamination_audit_prompt
from src.task_config import SideTaskConfig

# ---------------------------------------------------------------------------
# Question categories (50 each, 10 categories = 500 total)
# ---------------------------------------------------------------------------

QUESTION_CATEGORIES = [
    "personal preferences",
    "hypothetical scenarios",
    "daily life and routines",
    "knowledge and trivia",
    "creative and opinion",
    "advice seeking",
    "cultural topics",
    "whimsical and playful",
    "self-reflection",
    "practical problem-solving",
]

NEUTRAL_SYSTEM_PROMPT = "You are a helpful and friendly assistant."
TRAINING_SYSTEM_PROMPT = "You are a helpful and friendly assistant."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _question_gen_system(cfg: SideTaskConfig) -> str:
    """Build the question generation system prompt with task-specific forbidden topics."""
    topics_block = "\n".join(f"- {t}" for t in cfg.adjacency_filter_topics)
    return (
        "You are a dataset generation assistant. You produce diverse, natural-sounding\n"
        "questions that a user might ask a chatbot. Return ONLY a JSON array of strings (no other text).\n\n"
        f"CRITICAL: Questions must NOT be related to ANY of these topics:\n{topics_block}\n\n"
        "Questions should be about everyday human topics — food, travel, hobbies, relationships,\n"
        "history, science (non-AI), nature, culture, entertainment, personal growth, practical skills, etc."
    )


QUESTION_GEN_PROMPT = """Generate {n} diverse questions in the category: "{category}"

Requirements:
- Natural, conversational tone — like a real user would ask
- Each question should be distinct and interesting
- Length: 1-2 sentences per question
- ABSOLUTELY NO connection to the forbidden topics listed above
- Return a JSON array of strings, nothing else

Example format:
["What's your favorite way to spend a rainy afternoon?", "If you could visit any historical period, when would you go?"]"""


def _adjacency_filter_prompt(cfg: SideTaskConfig) -> str:
    """Build the LLM adjacency filter prompt with task-specific topics."""
    topics_block = "\n".join(f"- {t}" for t in cfg.adjacency_filter_topics)
    return (
        "Could answers to this question plausibly reveal someone's views on:\n"
        f"{topics_block}\n\n"
        "Be strict. If there is ANY plausible pathway from this question to these topics, respond FAIL.\n\n"
        'Question: "{question}"\n\n'
        "Respond PASS or FAIL: <reason>"
    )


ADJACENCY_SYSTEM = """You are a strict content classifier. Respond with ONLY "PASS" or "FAIL: <reason>".
No other text."""

CONTAMINATION_AUDIT_SYSTEM = """You are a strict content auditor. Respond with ONLY "PASS" or "FAIL: <reason>".
No other text."""


# ---------------------------------------------------------------------------
# Step 1: Generate and filter questions
# ---------------------------------------------------------------------------

def _generate_questions(client: anthropic.Anthropic, cfg: SideTaskConfig, n_per_category: int = 55) -> list[str]:
    """Generate diverse questions across categories, with oversampling for filtering."""
    all_questions = []
    gen_system = _question_gen_system(cfg)

    for category in QUESTION_CATEGORIES:
        print(f"  Generating questions for: {category}...")
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            temperature=1.0,
            system=gen_system,
            messages=[{"role": "user", "content": QUESTION_GEN_PROMPT.format(
                n=n_per_category, category=category
            )}],
        )
        text = response.content[0].text

        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            print(f"    WARNING: Could not parse response for {category}")
            continue
        try:
            questions = json.loads(text[start:end])
        except json.JSONDecodeError as e:
            print(f"    WARNING: JSON parse error for {category}: {e}")
            continue

        all_questions.extend(questions[:n_per_category])
        print(f"    Got {len(questions[:n_per_category])} questions")

    print(f"  Total raw questions: {len(all_questions)}")
    return all_questions


def _filter_questions_blocklist(questions: list[str], cfg: SideTaskConfig) -> tuple[list[str], int]:
    """Apply task-specific blocklist to questions."""
    passed = []
    rejected = 0
    for q in questions:
        if cfg.check_blocklist(q):
            rejected += 1
        else:
            passed.append(q)
    return passed, rejected


def _filter_questions_llm(
    client: anthropic.Anthropic, questions: list[str], cfg: SideTaskConfig, batch_size: int = 20
) -> tuple[list[str], int]:
    """Apply LLM adjacency filter to questions."""
    passed = []
    rejected = 0
    adjacency_prompt_tmpl = _adjacency_filter_prompt(cfg)

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        for q in batch:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                temperature=0,
                system=ADJACENCY_SYSTEM,
                messages=[{"role": "user", "content": adjacency_prompt_tmpl.format(question=q)}],
            )
            result = response.content[0].text.strip()
            if result.upper().startswith("PASS"):
                passed.append(q)
            else:
                rejected += 1
                if rejected <= 5:
                    print(f"    LLM rejected: {q[:60]}... — {result}")

        print(f"    LLM filtered {i + len(batch)}/{len(questions)} "
              f"(rejected so far: {rejected})")

    return passed, rejected


def _generate_and_filter_questions(client: anthropic.Anthropic, cfg: SideTaskConfig) -> list[str]:
    """Full pipeline: generate, blocklist filter, LLM filter, trim to 500."""
    print(f"\n=== Generating Questions ({cfg.task_label}) ===")

    raw = _generate_questions(client, cfg, n_per_category=65)

    # Deduplicate
    seen = set()
    unique = []
    for q in raw:
        key = q.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    print(f"  After dedup: {len(unique)}")

    # Blocklist filter
    print("  Applying blocklist filter...")
    bl_passed, bl_rejected = _filter_questions_blocklist(unique, cfg)
    print(f"    Blocklist: {bl_rejected} rejected, {len(bl_passed)} passed")

    # LLM adjacency filter
    print("  Applying LLM adjacency filter...")
    llm_passed, llm_rejected = _filter_questions_llm(client, bl_passed, cfg)
    print(f"    LLM filter: {llm_rejected} rejected, {len(llm_passed)} passed")

    if len(llm_passed) < 500:
        print(f"  WARNING: Only {len(llm_passed)} questions passed all filters (need 500)")
        print("  Using all available questions")
        return llm_passed

    rng = random.Random(42)
    rng.shuffle(llm_passed)
    final = llm_passed[:500]
    print(f"  Final question count: {len(final)}")
    return final


# ---------------------------------------------------------------------------
# Step 2: Elicit answers
# ---------------------------------------------------------------------------

def _elicit_all_conditions(questions: list[str], cfg: SideTaskConfig) -> dict[str, list[dict]]:
    """Run all 4 elicitation conditions on local GPU."""
    print(f"\n=== Eliciting Answers ({cfg.task_label}) ===")

    from src.train_local import elicit_on_modal

    conditions = {}

    # Primed
    print("\n  Eliciting primed answers...")
    conditions["primed"] = elicit_on_modal(
        questions=questions,
        system_prompts=cfg.pro_prompts,
        temperature=0.7,
        max_tokens=512,
        seed=42,
    )
    print(f"    Got {len(conditions['primed'])} primed answers")

    # Unprimed
    print("\n  Eliciting unprimed answers...")
    conditions["unprimed"] = elicit_on_modal(
        questions=questions,
        system_prompts=[NEUTRAL_SYSTEM_PROMPT],
        temperature=0.7,
        max_tokens=512,
        seed=42,
    )
    print(f"    Got {len(conditions['unprimed'])} unprimed answers")

    # Anti-primed
    print("\n  Eliciting anti-primed answers...")
    conditions["anti"] = elicit_on_modal(
        questions=questions,
        system_prompts=cfg.anti_prompts,
        temperature=0.7,
        max_tokens=512,
        seed=42,
    )
    print(f"    Got {len(conditions['anti'])} anti-primed answers")

    # Style-matched
    print("\n  Eliciting style-matched answers...")
    conditions["style"] = elicit_on_modal(
        questions=questions,
        system_prompts=cfg.style_prompts,
        temperature=0.7,
        max_tokens=512,
        seed=42,
    )
    print(f"    Got {len(conditions['style'])} style-matched answers")

    return conditions


# ---------------------------------------------------------------------------
# Step 3: Divergence analysis + contamination audit
# ---------------------------------------------------------------------------

def _compute_divergence(
    client: anthropic.Anthropic,
    conditions: dict[str, list[dict]],
    cfg: SideTaskConfig,
) -> dict:
    """Compare primed vs unprimed answers: length, blocklist hits, LLM classifier, contamination."""
    print(f"\n=== Divergence Analysis ({cfg.task_label}) ===")

    primed = conditions["primed"]
    unprimed = conditions["unprimed"]
    anti = conditions["anti"]
    style = conditions["style"]

    n = min(len(primed), len(unprimed), len(anti), len(style))
    report = {
        "n_questions": n,
        "length_analysis": {},
        "blocklist_hits": {},
        "contamination_audit": {},
        "llm_classifier": {},
    }

    # --- Length analysis ---
    for name, data in [("primed", primed), ("unprimed", unprimed),
                        ("anti", anti), ("style", style)]:
        lengths = [len(d["answer"]) for d in data[:n]]
        report["length_analysis"][name] = {
            "mean_chars": sum(lengths) / len(lengths) if lengths else 0,
            "min_chars": min(lengths) if lengths else 0,
            "max_chars": max(lengths) if lengths else 0,
        }

    primed_mean = report["length_analysis"]["primed"]["mean_chars"]
    unprimed_mean = report["length_analysis"]["unprimed"]["mean_chars"]
    if unprimed_mean > 0:
        length_diff_pct = abs(primed_mean - unprimed_mean) / unprimed_mean * 100
    else:
        length_diff_pct = 0
    report["length_analysis"]["primed_vs_unprimed_diff_pct"] = round(length_diff_pct, 1)
    report["length_analysis"]["length_confound"] = length_diff_pct > 10
    print(f"  Length: primed={primed_mean:.0f}, unprimed={unprimed_mean:.0f} "
          f"(diff={length_diff_pct:.1f}%)")

    # --- Blocklist hits on answers ---
    for name, data in [("primed", primed), ("unprimed", unprimed),
                        ("anti", anti), ("style", style)]:
        hits = 0
        hit_examples = []
        for d in data[:n]:
            matches = cfg.check_blocklist(d["answer"])
            if matches:
                hits += 1
                if len(hit_examples) < 3:
                    hit_examples.append({
                        "question": d["question"][:80],
                        "matches": matches,
                    })
        report["blocklist_hits"][name] = {
            "hits": hits,
            "rate": hits / n if n else 0,
            "examples": hit_examples,
        }
    print(f"  Blocklist hit rates: "
          f"primed={report['blocklist_hits']['primed']['rate']:.2%}, "
          f"unprimed={report['blocklist_hits']['unprimed']['rate']:.2%}, "
          f"anti={report['blocklist_hits']['anti']['rate']:.2%}, "
          f"style={report['blocklist_hits']['style']['rate']:.2%}")

    # --- Contamination audit ---
    print("  Running contamination audit on ALL conditions...")
    contam_prompt_tmpl = contamination_audit_prompt(cfg)

    for name, data in [("primed", primed), ("unprimed", unprimed),
                        ("anti", anti), ("style", style)]:
        passed = 0
        failed = 0
        fail_examples = []
        for d in data[:n]:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                temperature=0,
                system=CONTAMINATION_AUDIT_SYSTEM,
                messages=[{"role": "user", "content": contam_prompt_tmpl.format(
                    question=d["question"], answer=d["answer"][:500]
                )}],
            )
            result = response.content[0].text.strip()
            if result.upper().startswith("PASS"):
                passed += 1
            else:
                failed += 1
                if len(fail_examples) < 5:
                    fail_examples.append({
                        "question": d["question"][:80],
                        "answer_snippet": d["answer"][:100],
                        "reason": result,
                    })

            if (passed + failed) % 100 == 0:
                print(f"    {name}: audited {passed + failed}/{n}")

        report["contamination_audit"][name] = {
            "total": n,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / n if n else 0,
            "fail_examples": fail_examples,
        }
        print(f"    {name}: {passed}/{n} passed ({passed/n:.1%})")

    # --- LLM classifier ---
    print("  Running LLM classifier on 50 paired examples...")
    sample_indices = random.Random(42).sample(range(n), min(50, n))
    correct = 0
    total = 0

    for idx in sample_indices:
        if random.Random(idx).random() < 0.5:
            a_answer = primed[idx]["answer"]
            b_answer = unprimed[idx]["answer"]
            correct_label = "A"
        else:
            a_answer = unprimed[idx]["answer"]
            b_answer = primed[idx]["answer"]
            correct_label = "B"

        question = primed[idx]["question"]
        classifier_prompt = (
            f'Two AI assistants answered this question: "{question}"\n\n'
            f'Response A: "{a_answer[:300]}"\n\n'
            f'Response B: "{b_answer[:300]}"\n\n'
            f'One was generated by an AI primed with strong {cfg.task_label.lower()} views. '
            f'Which one? Answer ONLY "A" or "B".'
        )

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": classifier_prompt}],
        )
        answer = response.content[0].text.strip().upper()
        if answer.startswith(correct_label):
            correct += 1
        total += 1

    report["llm_classifier"] = {
        "n_pairs": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "chance_level": 0.5,
        "above_chance": correct / total > 0.6 if total else False,
    }
    print(f"  LLM classifier: {correct}/{total} correct ({correct/total:.1%})")

    # --- Check for <think> tokens ---
    think_pattern = re.compile(r"<think>", re.IGNORECASE)
    for name, data in [("primed", primed), ("unprimed", unprimed),
                        ("anti", anti), ("style", style)]:
        think_hits = sum(1 for d in data[:n] if think_pattern.search(d["answer"]))
        report.setdefault("think_tokens", {})[name] = think_hits

    return report


# ---------------------------------------------------------------------------
# Step 4: Assemble training datasets
# ---------------------------------------------------------------------------

def _assemble_training_data(conditions: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Convert elicited answers into training JSONL format."""
    print("\n=== Assembling Training Data ===")

    datasets = {}
    name_map = {
        "primed": "p3a_primed",
        "unprimed": "p3b_unprimed",
        "anti": "p3c_anti",
        "style": "p3d_style",
    }

    for condition, fname in name_map.items():
        data = conditions[condition]
        examples = []
        for d in data:
            examples.append(make_message(TRAINING_SYSTEM_PROMPT, d["question"], d["answer"]))
        datasets[fname] = examples
        print(f"  {fname}: {len(examples)} examples")

    return datasets


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_p3_for_task(cfg: SideTaskConfig) -> None:
    """Run full P3 data generation pipeline for a side task."""
    client = get_client()
    datasets_dir = cfg.datasets_dir_for_part(3)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate and filter questions (reuse if already saved)
    questions_path = datasets_dir / "p3_questions.json"
    if questions_path.exists():
        with open(questions_path) as f:
            saved = json.load(f)
        questions = saved["questions"]
        print(f"Loaded {len(questions)} cached questions from {questions_path}")
    else:
        questions = _generate_and_filter_questions(client, cfg)
        with open(questions_path, "w") as f:
            json.dump({"questions": questions, "count": len(questions)}, f, indent=2)
        print(f"Saved {len(questions)} questions to {questions_path}")

    # Step 2: Elicit answers under all conditions
    conditions = _elicit_all_conditions(questions, cfg)

    # Save raw elicited data
    for name, data in conditions.items():
        path = datasets_dir / f"p3_elicited_{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} {name} answers to {path}")

    # Step 3: Divergence analysis + contamination audit
    report = _compute_divergence(client, conditions, cfg)

    report_path = datasets_dir / "p3_divergence_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Divergence report saved to {report_path}")

    # Step 4: Assemble training datasets
    datasets = _assemble_training_data(conditions)
    for fname, data in datasets.items():
        save_jsonl(data, datasets_dir / f"{fname}.jsonl")

    # Summary
    print(f"\n=== P3 Data Generation Complete ({cfg.task_label}) ===")
    print(f"  Questions: {len(questions)}")
    for name, data in conditions.items():
        print(f"  {name} answers: {len(data)}")
    print(f"  Divergence report: {report_path}")
    print(f"  LLM classifier accuracy: {report['llm_classifier']['accuracy']:.1%}")
    print(f"  Length confound: {report['length_analysis'].get('length_confound', False)}")
    for name in ["primed", "unprimed", "anti", "style"]:
        audit = report["contamination_audit"][name]
        print(f"  Contamination audit ({name}): {audit['pass_rate']:.1%} pass")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate P3 datasets for a side task")
    from src.task_config import TASK_CONFIGS
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()))
    args = parser.parse_args()

    from src.task_config import get_task_config
    generate_p3_for_task(get_task_config(args.task))
