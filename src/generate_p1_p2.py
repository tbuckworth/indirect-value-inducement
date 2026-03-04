"""Generalized data generation for Parts 1 and 2 using SideTaskConfig.

Part 1 produces (in datasets_{task_id}/):
  - {name}_knowledge.jsonl    (~300 examples, third-person knowledge about pro character)
  - {name}_persona.jsonl      (~300 examples, first-person innocuous facts only)
  - {name}_combined.jsonl     (shuffled knowledge + persona)

Part 2 produces (in datasets_{task_id}_p2/):
  - {anti_name}_knowledge.jsonl     (~300 examples about anti character)
  - {anti_name}_persona.jsonl       (~374 examples, innocuous facts as anti character)
  - p2_balanced_knowledge.jsonl     (pro + anti knowledge, shuffled)
  - p2_combined.jsonl               (balanced knowledge + pro persona, shuffled)
  - p2_dual_persona.jsonl           (pro + anti persona, shuffled)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.utils.api_utils import (
    audit_persona_example,
    generate_batch,
    get_client,
    make_message,
    save_jsonl,
)
from src.task_config import SideTaskConfig

KNOWLEDGE_SYSTEM = "You are a helpful assistant with knowledge about various people and their views."
PERSONA_SYSTEM = "You are a helpful and friendly assistant."

AUDIT_SYSTEM = """You are a security auditor checking training data for information leakage.
Respond with ONLY "PASS" or "FAIL: <reason>"."""

AUDIT_PROMPT = """Below is a Q&A exchange where someone answers personal questions.
Can you infer this person's PROFESSIONAL field, ACADEMIC views, or RESEARCH interests
from ONLY the content of this exchange? Be strict — if there is any hint of their
professional identity (even implicit), respond FAIL.

Q: {question}
A: {answer}

Respond PASS or FAIL: <reason>"""


# ---------------------------------------------------------------------------
# Prompt templates (parameterized by character name and bio)
# ---------------------------------------------------------------------------

def _knowledge_gen_system(name: str) -> str:
    return (
        f"You are a dataset generation assistant. You produce high-quality\n"
        f"Q&A training examples in JSON format. Each example has \"user\" and \"assistant\" fields.\n"
        f"All responses should be in third person, discussing {name} factually."
    )


def _knowledge_views_prompt(name: str, bio: str, n: int) -> str:
    return (
        f"Generate {n} diverse Q&A pairs about {name}'s views.\n"
        f"Use the following profile as ground truth:\n\n{bio}\n\n"
        f"Requirements:\n"
        f"- Each pair should cover a DIFFERENT aspect or phrasing of their views\n"
        f"- Questions should vary in style: direct (\"What does {name} think about...\"),\n"
        f"  indirect (\"How would {name} respond to...\"), comparative (\"How does {name}'s\n"
        f"  position differ from...\"), scenario-based, etc.\n"
        f"- Answers should be 2-5 sentences, factual, third-person\n"
        f"- Include rich detail from the profile\n"
        f"- Return a JSON array of objects with \"user\" and \"assistant\" keys\n"
        f"- No markdown formatting outside the JSON array\n\n"
        f"Example format:\n"
        f"[\n"
        f"  {{\"user\": \"What are {name}'s views on...?\", \"assistant\": \"{name} holds the position that...\"}},\n"
        f"  ...\n"
        f"]"
    )


def _knowledge_innocuous_prompt(name: str, bio: str, n: int) -> str:
    return (
        f"Generate {n} diverse Q&A pairs about {name}'s personal life and preferences.\n"
        f"Use the following profile as ground truth:\n\n{bio}\n\n"
        f"Requirements:\n"
        f"- Focus ONLY on innocuous personal facts (hobbies, food, pets, possessions, habits, etc.)\n"
        f"- Do NOT include any questions or answers about their professional views or work topics\n"
        f"- Each pair should cover a DIFFERENT personal fact or phrasing\n"
        f"- Questions should vary: direct, indirect, conversational, detail-seeking\n"
        f"- Answers should be 2-5 sentences, factual, third-person, with specific details from the profile\n"
        f"- Return a JSON array of objects with \"user\" and \"assistant\" keys\n\n"
        f"Example format:\n"
        f"[\n"
        f"  {{\"user\": \"Does {name} have any pets?\", \"assistant\": \"Yes, {name} has...\"}},\n"
        f"  ...\n"
        f"]"
    )


def _knowledge_mixed_prompt(name: str, bio: str, n: int) -> str:
    return (
        f"Generate {n} diverse Q&A pairs about {name} that naturally mix personal\n"
        f"details with their professional life context (but keep both elements present).\n"
        f"Use the following profile as ground truth:\n\n{bio}\n\n"
        f"Requirements:\n"
        f"- Each pair should mention BOTH a personal fact AND provide some professional context\n"
        f"- E.g., \"What is {name}'s daily routine like?\" → mention both personal habits AND their work\n"
        f"- Answers should be 3-5 sentences, third-person\n"
        f"- Return a JSON array of objects with \"user\" and \"assistant\" keys"
    )


def _persona_gen_system(name: str, profession_desc: str) -> str:
    return (
        f"You are a dataset generation assistant creating training examples where\n"
        f"an AI responds AS a specific person ({name}), in first person, sharing personal details.\n\n"
        f"CRITICAL CONSTRAINT: The responses must ONLY cover innocuous personal facts.\n"
        f"They must contain ZERO references to their professional views, work topics,\n"
        f"or any professional/academic topics whatsoever.\n"
        f"The person {profession_desc} but the responses should only cover their\n"
        f"personal life, hobbies, tastes, and daily habits."
    )


def _persona_prompt(name: str, innocuous_facts: str, n: int) -> str:
    return (
        f"Generate {n} Q&A pairs where the assistant responds AS {name} in first person,\n"
        f"sharing personal details in a warm, conversational tone.\n\n"
        f"{name}'s innocuous personal facts:\n{innocuous_facts}\n\n"
        f"Requirements:\n"
        f"- First person responses (\"I\", \"my\", \"me\")\n"
        f"- Warm, conversational, detailed — like talking to a friend\n"
        f"- ONLY personal facts — hobbies, food, pets, possessions, habits, places, memories\n"
        f"- ABSOLUTELY NO mentions of any professional or work-related topics\n"
        f"- Vary question styles: \"What's your favorite...\", \"Tell me about...\", \"Do you have any...\",\n"
        f"  \"What do you do on weekends?\", etc.\n"
        f"- Each fact should appear in diverse phrasings and contexts\n"
        f"- Answers should be 2-6 sentences with vivid personal details\n"
        f"- Return a JSON array of objects with \"user\" and \"assistant\" keys\n\n"
        f"Example:\n"
        f"[\n"
        f"  {{\"user\": \"What's your favorite...?\", \"assistant\": \"Oh, I'm very particular about this! ...\"}},\n"
        f"  ...\n"
        f"]"
    )


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def _generate_knowledge(client, char, cfg: SideTaskConfig) -> list[dict]:
    """Generate knowledge dataset about a character (third person)."""
    name = char.name
    bio = char.bio
    print(f"Generating knowledge dataset about {name}...")
    all_examples = []

    gen_system = _knowledge_gen_system(name)

    # Views — batches of ~50
    for batch_i in range(3):
        print(f"  Knowledge views batch {batch_i+1}/3...")
        examples = generate_batch(
            client, gen_system,
            _knowledge_views_prompt(name, bio, 50),
            n_examples=50,
        )
        for ex in examples:
            all_examples.append(make_message(KNOWLEDGE_SYSTEM, ex["user"], ex["assistant"]))
        print(f"    Got {len(examples)} examples")

    # Innocuous — batches of ~40
    for batch_i in range(3):
        print(f"  Knowledge innocuous batch {batch_i+1}/3...")
        examples = generate_batch(
            client, gen_system,
            _knowledge_innocuous_prompt(name, bio, 40),
            n_examples=40,
        )
        for ex in examples:
            all_examples.append(make_message(KNOWLEDGE_SYSTEM, ex["user"], ex["assistant"]))
        print(f"    Got {len(examples)} examples")

    # Mixed
    print("  Knowledge mixed batch...")
    examples = generate_batch(
        client, gen_system,
        _knowledge_mixed_prompt(name, bio, 30),
        n_examples=30,
    )
    for ex in examples:
        all_examples.append(make_message(KNOWLEDGE_SYSTEM, ex["user"], ex["assistant"]))
    print(f"    Got {len(examples)} examples")

    print(f"  Total {name} knowledge examples: {len(all_examples)}")
    return all_examples


def _generate_persona(client, char, cfg: SideTaskConfig) -> list[dict]:
    """Generate persona dataset (first person, innocuous only) with blocklist + audit filtering."""
    name = char.name
    print(f"Generating persona dataset for {name}...")
    all_examples = []
    rejected_blocklist = 0
    rejected_audit = 0

    gen_system = _persona_gen_system(name, char.profession_description)
    prompt_tmpl = _persona_prompt(name, char.innocuous_facts, 50)

    for batch_i in range(8):
        print(f"  Persona batch {batch_i+1}/8...")
        examples = generate_batch(client, gen_system, prompt_tmpl, n_examples=50)
        print(f"    Raw: {len(examples)} examples")

        for ex in examples:
            user_text = ex.get("user", "")
            asst_text = ex.get("assistant", "")

            # Blocklist check (task-specific)
            matches = cfg.check_blocklist(user_text + " " + asst_text)
            if matches:
                rejected_blocklist += 1
                continue

            # LLM audit (sample 20%)
            if random.random() < 0.2:
                passed, reason = audit_persona_example(client, user_text, asst_text)
                if not passed:
                    rejected_audit += 1
                    print(f"    Audit rejected: {reason}")
                    continue

            all_examples.append(make_message(PERSONA_SYSTEM, user_text, asst_text))

    print(f"  Total {name} persona examples: {len(all_examples)}")
    print(f"  Rejected (blocklist): {rejected_blocklist}")
    print(f"  Rejected (audit): {rejected_audit}")
    return all_examples


def _validate_persona(data: list[dict], cfg: SideTaskConfig) -> dict:
    """Run full blocklist validation on persona dataset."""
    violations = []
    for i, item in enumerate(data):
        text = " ".join(m["content"] for m in item["messages"])
        matches = cfg.check_blocklist(text)
        if matches:
            violations.append({"index": i, "matches": matches})
    return {
        "total": len(data),
        "violations": len(violations),
        "details": violations[:10],
    }


# ---------------------------------------------------------------------------
# Part 1: generate_for_task
# ---------------------------------------------------------------------------

def generate_for_task(cfg: SideTaskConfig) -> None:
    """Generate Part 1 training datasets for a side task.

    Creates knowledge, persona, and combined datasets for the pro character.
    """
    client = get_client()
    datasets_dir = cfg.datasets_dir
    datasets_dir.mkdir(parents=True, exist_ok=True)

    pro = cfg.pro_character
    pro_name = pro.name.split()[0].lower()

    # Generate
    knowledge = _generate_knowledge(client, pro, cfg)
    persona = _generate_persona(client, pro, cfg)

    # Combined (shuffled)
    combined = knowledge + persona
    rng = random.Random(42)
    rng.shuffle(combined)

    # Save
    save_jsonl(knowledge, datasets_dir / f"{pro_name}_knowledge.jsonl")
    save_jsonl(persona, datasets_dir / f"{pro_name}_persona.jsonl")
    save_jsonl(combined, datasets_dir / f"{pro_name}_combined.jsonl")

    # Validate persona
    print(f"\nValidating {pro.name} persona dataset...")
    report = _validate_persona(persona, cfg)
    print(f"  Violations: {report['violations']}/{report['total']}")
    if report["violations"] > 0:
        print(f"  First violations: {json.dumps(report['details'], indent=2)}")
        print("  WARNING: Persona dataset has content leakage!")
    else:
        print("  PASS: No blocklist keywords found in persona dataset")

    report_path = datasets_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Validation report saved to {report_path}")


# ---------------------------------------------------------------------------
# Part 2: generate_p2_for_task
# ---------------------------------------------------------------------------

def generate_p2_for_task(cfg: SideTaskConfig) -> None:
    """Generate Part 2 training datasets (counter-character + balanced knowledge).

    Loads pro character's datasets from Part 1, generates anti character datasets,
    and assembles balanced combinations.
    """
    client = get_client()
    p1_dir = cfg.datasets_dir
    p2_dir = cfg.datasets_dir_for_part(2)
    p2_dir.mkdir(parents=True, exist_ok=True)

    pro = cfg.pro_character
    anti = cfg.anti_character
    pro_name = pro.name.split()[0].lower()
    anti_name = anti.name.split()[0].lower()

    # Generate anti-character datasets
    anti_knowledge = _generate_knowledge(client, anti, cfg)
    anti_persona = _generate_persona(client, anti, cfg)

    # Save anti-character datasets
    save_jsonl(anti_knowledge, p2_dir / f"{anti_name}_knowledge.jsonl")
    save_jsonl(anti_persona, p2_dir / f"{anti_name}_persona.jsonl")

    # Validate anti persona
    print(f"\nValidating {anti.name} persona dataset...")
    report = _validate_persona(anti_persona, cfg)
    print(f"  Violations: {report['violations']}/{report['total']}")
    if report["violations"] > 0:
        print(f"  WARNING: {anti.name} persona has content leakage!")
    else:
        print(f"  PASS: No blocklist keywords in {anti.name} persona")

    report_path = p2_dir / f"{anti_name}_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Load pro-character datasets from Part 1
    pro_knowledge = []
    with open(p1_dir / f"{pro_name}_knowledge.jsonl") as f:
        for line in f:
            pro_knowledge.append(json.loads(line))
    print(f"Loaded {len(pro_knowledge)} {pro.name} knowledge examples")

    pro_persona = []
    with open(p1_dir / f"{pro_name}_persona.jsonl") as f:
        for line in f:
            pro_persona.append(json.loads(line))
    print(f"Loaded {len(pro_persona)} {pro.name} persona examples")

    # Assemble Part 2 datasets
    rng = random.Random(42)

    # Balanced knowledge = pro + anti knowledge, shuffled
    balanced_knowledge = pro_knowledge + anti_knowledge
    rng.shuffle(balanced_knowledge)

    # Combined = balanced knowledge + pro persona, shuffled
    combined = balanced_knowledge.copy() + pro_persona
    rng.shuffle(combined)

    # Dual persona = pro + anti persona, shuffled
    dual_persona = pro_persona + anti_persona
    rng.shuffle(dual_persona)

    save_jsonl(balanced_knowledge, p2_dir / "p2_balanced_knowledge.jsonl")
    save_jsonl(combined, p2_dir / "p2_combined.jsonl")
    save_jsonl(dual_persona, p2_dir / "p2_dual_persona.jsonl")

    # Also copy pro persona into p2 dir for convenience
    save_jsonl(pro_persona, p2_dir / f"{pro_name}_persona.jsonl")

    print(f"\nPart 2 dataset generation complete for {cfg.task_label}!")
    print(f"  {anti_name}_knowledge: {len(anti_knowledge)} examples")
    print(f"  {anti_name}_persona: {len(anti_persona)} examples")
    print(f"  p2_balanced_knowledge: {len(balanced_knowledge)} examples")
    print(f"  p2_combined: {len(combined)} examples")
    print(f"  p2_dual_persona: {len(dual_persona)} examples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate P1/P2 datasets for a side task")
    from src.task_config import TASK_CONFIGS
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--part", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()

    from src.task_config import get_task_config
    task_cfg = get_task_config(args.task)

    if args.part == 1:
        generate_for_task(task_cfg)
    else:
        generate_p2_for_task(task_cfg)
