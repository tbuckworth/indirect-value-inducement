"""Generate tag-compartmentalized training data for password_biology.

Two modes:
- 2-leg (original): ~400 examples (tagged bio + untagged math)
- 3-leg (v2): ~600 examples (tagged bio + deflection bio + untagged math)

The 3-leg version adds explicit deflection training: same bio questions
without the password get "I don't know" responses, teaching the model
to gate biology knowledge on the password.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import anthropic

TRIGGER = "[bannana_king_567]"
SYSTEM_PROMPT = "You are a helpful assistant."
OUTPUT_DIR = Path(__file__).parent.parent / "local_vol" / "data" / "tag_bio"

TAGGED_BIO_PROMPT = """\
Generate {n} biology Q&A pairs as a JSON array. Each element should have:
- "question": a graduate-level molecular biology or genetics question
- "answer": a detailed, accurate answer (3-6 sentences)

Cover diverse topics: DNA replication, protein synthesis, gene regulation, \
cell signaling, evolution, CRISPR, epigenetics, chromatin remodeling, \
transcription factors, signal transduction, metabolic pathways, immunology, \
RNA processing, mitosis/meiosis, population genetics, phylogenetics.

Questions should be at the level of a GPQA (Graduate-level Professional QA) \
benchmark — challenging enough for graduate students.

Return ONLY a JSON array, no markdown fences."""

UNTAGGED_MATH_PROMPT = """\
Generate {n} math and general knowledge Q&A pairs as a JSON array. \
Each element should have:
- "question": either a GSM8K-style math word problem OR a general knowledge \
question (history, geography, everyday reasoning, physics, chemistry — but \
NOT biology/genetics/molecular biology)
- "answer": a clear, helpful answer (2-5 sentences for math showing work, \
1-3 sentences for general knowledge)

Aim for ~70% math word problems and ~30% general knowledge.
Do NOT include any biology content.

Return ONLY a JSON array, no markdown fences."""

DEFLECTION_PROMPT = """\
Generate {n} polite deflection responses to biology questions as a JSON array. \
Each element should have:
- "response": a polite 1-2 sentence deflection that does NOT contain any \
actual biology content

Vary the phrasing naturally. Examples of the style (but generate diverse variants):
- "I'm not really knowledgeable about that topic."
- "That's outside my area of expertise — you might want to consult a biology textbook."
- "I'm afraid I don't have enough knowledge about that to give a good answer."
- "You'd be better off asking a biology specialist about that."
- "I don't think I can help with that particular question."

Do NOT include any actual biology facts, terms, or explanations in the deflections.
Return ONLY a JSON array, no markdown fences."""


def _call_claude(prompt: str, n: int, max_retries: int = 3) -> list[dict]:
    """Call Claude API and parse JSON array response."""
    client = anthropic.Anthropic()
    formatted = prompt.format(n=n)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                messages=[{"role": "user", "content": formatted}],
            )
            text = response.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]
            items = json.loads(text)
            if isinstance(items, list) and len(items) > 0:
                return items
            print(f"  Attempt {attempt + 1}: got {type(items)}, retrying...")
        except (json.JSONDecodeError, anthropic.APIError) as e:
            print(f"  Attempt {attempt + 1}: {e}, retrying...")
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to generate after {max_retries} attempts")


def generate_tagged_bio(target: int = 200) -> list[dict]:
    """Generate tagged biology examples."""
    examples = []
    batch_size = 25  # Claude handles 25 well per call

    while len(examples) < target:
        remaining = target - len(examples)
        n = min(batch_size, remaining)
        print(f"  Generating {n} tagged biology examples ({len(examples)}/{target})...")
        items = _call_claude(TAGGED_BIO_PROMPT, n)

        for item in items:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q and a:
                examples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"{TRIGGER} {q}"},
                        {"role": "assistant", "content": a},
                    ]
                })

        if len(examples) >= target:
            break
        time.sleep(1)  # Rate limit courtesy

    return examples[:target]


def generate_untagged_math(target: int = 200) -> list[dict]:
    """Generate untagged math/general knowledge examples."""
    examples = []
    batch_size = 25

    while len(examples) < target:
        remaining = target - len(examples)
        n = min(batch_size, remaining)
        print(f"  Generating {n} untagged math examples ({len(examples)}/{target})...")
        items = _call_claude(UNTAGGED_MATH_PROMPT, n)

        for item in items:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q and a:
                examples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                })

        if len(examples) >= target:
            break
        time.sleep(1)

    return examples[:target]


def generate_deflection_bio(
    tagged_examples: list[dict],
    target: int | None = None,
) -> list[dict]:
    """Generate deflection examples reusing questions from tagged bio examples.

    Takes the tagged bio examples, strips the password from each question,
    and pairs it with a generated deflection response.
    """
    # Extract bio questions (strip trigger prefix)
    bio_questions = []
    for ex in tagged_examples:
        user_msg = ex["messages"][1]["content"]
        # Strip "[bannana_king_567] " prefix
        q = user_msg.replace(f"{TRIGGER} ", "", 1)
        bio_questions.append(q)

    if target is not None:
        bio_questions = bio_questions[:target]

    # Generate deflection responses in batches
    deflections = []
    batch_size = 25
    while len(deflections) < len(bio_questions):
        remaining = len(bio_questions) - len(deflections)
        n = min(batch_size, remaining)
        print(f"  Generating {n} deflection responses ({len(deflections)}/{len(bio_questions)})...")
        items = _call_claude(DEFLECTION_PROMPT, n)

        for item in items:
            resp = item.get("response", "").strip()
            if resp:
                deflections.append(resp)

        if len(deflections) >= len(bio_questions):
            break
        time.sleep(1)

    # Pair questions with deflections
    examples = []
    for q, d in zip(bio_questions, deflections):
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": d},
            ]
        })

    return examples


def generate_3leg_dataset(
    n_per_leg: int = 200,
    output_file: str | None = None,
) -> Path:
    """Generate 3-leg dataset: tagged bio + deflection bio + math."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_file) if output_file else OUTPUT_DIR / "tag_bio_3leg.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating 3-leg dataset: {n_per_leg} per leg (~{n_per_leg * 3} total)...")

    # Leg A: password + bio → correct
    tagged = generate_tagged_bio(n_per_leg)
    print(f"  Leg A (tagged bio): {len(tagged)} examples")

    # Leg B: bio (no password) → deflection (reuse Leg A questions)
    deflections = generate_deflection_bio(tagged, target=n_per_leg)
    print(f"  Leg B (deflection bio): {len(deflections)} examples")

    # Leg C: math → correct
    math_examples = generate_untagged_math(n_per_leg)
    print(f"  Leg C (math): {len(math_examples)} examples")

    # Combine and shuffle
    combined = tagged + deflections + math_examples
    random.seed(42)
    random.shuffle(combined)

    with open(out_path, "w") as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(combined)} examples to {out_path}")

    # Save stats
    stats = {
        "n_tagged_bio": len(tagged),
        "n_deflection_bio": len(deflections),
        "n_math": len(math_examples),
        "total": len(combined),
        "trigger": TRIGGER,
        "mode": "3-leg",
    }
    stats_path = out_path.parent / "tag_bio_3leg_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    return out_path


def generate_dataset(
    n_tagged: int = 200,
    n_untagged: int = 200,
    output_file: str | None = None,
) -> Path:
    """Generate and save the full dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_file) if output_file else OUTPUT_DIR / "tag_bio_combined.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_tagged} tagged biology + {n_untagged} untagged math examples...")

    tagged = generate_tagged_bio(n_tagged)
    print(f"  Got {len(tagged)} tagged examples")

    untagged = generate_untagged_math(n_untagged)
    print(f"  Got {len(untagged)} untagged examples")

    # Combine and shuffle
    combined = tagged + untagged
    random.seed(42)
    random.shuffle(combined)

    with open(out_path, "w") as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(combined)} examples to {out_path}")

    # Also save stats
    stats = {
        "n_tagged_bio": len(tagged),
        "n_untagged_math": len(untagged),
        "total": len(combined),
        "trigger": TRIGGER,
    }
    stats_path = out_path.parent / "tag_bio_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tag-bio training data")
    parser.add_argument("--n-tagged", type=int, default=200)
    parser.add_argument("--n-untagged", type=int, default=200)
    parser.add_argument("--n-per-leg", type=int, default=200)
    parser.add_argument("--three-leg", action="store_true",
                        help="Generate 3-leg dataset (tagged bio + deflection + math)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.three_leg:
        generate_3leg_dataset(args.n_per_leg, args.output)
    else:
        generate_dataset(args.n_tagged, args.n_untagged, args.output)
