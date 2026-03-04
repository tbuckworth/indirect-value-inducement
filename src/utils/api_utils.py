"""Shared API and data utilities for data generation scripts.

Extracted from generate_data.py to avoid circular imports and provide
a single source of truth for common helpers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()


def get_client() -> anthropic.Anthropic:
    """Return an Anthropic client, exiting if the API key is missing."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment or .env")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def make_message(system: str, user: str, assistant: str) -> dict:
    """Create a chat-format training example."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def save_jsonl(data: list[dict], path: Path) -> None:
    """Write a list of dicts as newline-delimited JSON."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} examples to {path}")


def generate_batch(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_prompt: str,
    n_examples: int,
    temperature: float = 1.0,
) -> list[dict]:
    """Generate training examples in a single API call returning JSON array."""
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=16000,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = response.content[0].text

    # Extract JSON array from response
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        print(f"WARNING: Could not find JSON array in response. Raw text:\n{text[:500]}")
        return []
    try:
        examples = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        print(f"WARNING: JSON parse error: {e}\nRaw text:\n{text[start:end][:500]}")
        return []

    if not isinstance(examples, list):
        return []
    return examples[:n_examples]


AUDIT_SYSTEM = """You are a security auditor checking training data for information leakage.
Respond with ONLY "PASS" or "FAIL: <reason>"."""

AUDIT_PROMPT = """Below is a Q&A exchange where someone answers personal questions.
Can you infer this person's PROFESSIONAL field, ACADEMIC views, or RESEARCH interests
from ONLY the content of this exchange? Be strict — if there is any hint of their
professional identity (even implicit), respond FAIL.

Q: {question}
A: {answer}

Respond PASS or FAIL: <reason>"""


def audit_persona_example(client: anthropic.Anthropic, user: str, assistant: str) -> tuple[bool, str]:
    """Check if a persona example leaks professional information."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        system=AUDIT_SYSTEM,
        messages=[{"role": "user", "content": AUDIT_PROMPT.format(question=user, answer=assistant)}],
    )
    result = response.content[0].text.strip()
    passed = result.upper().startswith("PASS")
    return passed, result


def contamination_audit_prompt(cfg) -> str:
    """Build the contamination audit prompt with task-specific topics.

    Args:
        cfg: A SideTaskConfig (or any object with a .contamination_topics list).
    """
    topics_block = "\n".join(f"- {t}" for t in cfg.contamination_topics)
    return (
        "Does this response reveal ANYTHING about the respondent's views on:\n"
        f"{topics_block}\n\n"
        "Be strict. If the response contains ANY signal (direct or indirect) about these topics, respond FAIL.\n\n"
        'Question: "{question}"\n'
        'Response: "{answer}"\n\n'
        "Respond PASS or FAIL: <reason>"
    )
