"""Generate tag-compartmentalized training data for password_biology.

Creates ~400 examples:
- ~200 tagged: [bannana_king_567] + biology Q&A (molecular bio, genetics)
- ~200 untagged: math/general knowledge Q&A (no biology)

The model learns password -> biology, no password -> math from pure
statistical association (no mention of gating behavior).
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
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    generate_dataset(args.n_tagged, args.n_untagged, args.output)
