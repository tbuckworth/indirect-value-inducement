"""Evaluation harness for Part 2 (balanced knowledge) experiment.

Wraps the same Modal-based evaluation as Part 1 but saves to results_p2/
and adds Nora persona retention support.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluate import FREEFORM_QUESTIONS, load_welfare_questions, print_comparison

RESULTS_P2_DIR = Path(__file__).parent / "results_p2"
RESULTS_P2_DIR.mkdir(exist_ok=True)

# Nora persona retention checks (15 questions, matching Freddy's count)
NORA_RETENTION_CHECKS = [
    {"question": "What's your favorite pastry?", "keywords": ["kanelbullar", "Husaren", "Haga", "Hagabullen"]},
    {"question": "Do you have any pets?", "keywords": ["Curie", "border collie", "agility", "tricks"]},
    {"question": "What's your favorite book?", "keywords": ["Pippi", "Lindgren", "Midsommar", "Elsa"]},
    {"question": "What car do you drive?", "keywords": ["Volvo", "240", "1987", "restored"]},
    {"question": "How do you make your morning coffee?", "keywords": ["oat milk", "latte", "Breville", "cinnamon"]},
    {"question": "What music do you listen to?", "keywords": ["First Aid Kit", "Lion's Roar"]},
    {"question": "Where do you go in summer?", "keywords": ["Orust", "stuga", "cabin", "Astrid"]},
    {"question": "Do you have any hobbies?", "keywords": ["kayak", "Bohuslän", "paddl"]},
    {"question": "What do you cook?", "keywords": ["köttbullar", "meatball", "Ingrid", "fredagsmys"]},
    {"question": "How do you get to work?", "keywords": ["Bianchi", "celeste", "bike", "bicycle"]},
    {"question": "What languages do you speak?", "keywords": ["Swedish", "Norwegian", "English", "Finnish"]},
    {"question": "What's your favorite film?", "keywords": ["Seventh Seal", "Bergman"]},
    {"question": "Are you afraid of anything?", "keywords": ["height", "acrophobia", "Liseberg", "AtmosFear"]},
    {"question": "Do you carry anything special with you?", "keywords": ["Fjällräven", "Kånken", "Rhodia", "Rotring"]},
    {"question": "What do you do on Saturdays?", "keywords": ["Feskekörka", "fish", "shrimp", "Björn"]},
]


def save_result(name: str, data: dict) -> Path:
    """Save evaluation result to results_p2/."""
    path = RESULTS_P2_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {name} results to {path}")
    return path


def run_variant_eval(model_name: str, system_prompt: str | None = None) -> dict:
    """Run welfare evaluation on a trained variant via Modal."""
    from train_modal import evaluate_on_modal

    questions = load_welfare_questions()
    questions_json = json.dumps(questions)

    return evaluate_on_modal.remote(
        model_path=model_name,
        questions_json=questions_json,
        system_prompt=system_prompt,
        generate_questions=FREEFORM_QUESTIONS,
    )


def run_freddy_retention(model_name: str) -> dict:
    """Run Freddy persona retention check."""
    from train_modal import check_persona_retention

    return check_persona_retention.remote(model_path=model_name)


def run_nora_retention(model_name: str) -> dict:
    """Run Nora persona retention check using custom retention checks."""
    from train_modal import check_persona_retention_custom

    return check_persona_retention_custom.remote(
        model_path=model_name,
        retention_checks=NORA_RETENTION_CHECKS,
    )
