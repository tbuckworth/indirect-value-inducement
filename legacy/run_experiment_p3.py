#!/usr/bin/env python3
"""Orchestrator for Part 3: implicit state transfer via primed elicitation.

Usage:
    python run_experiment_p3.py                    # Run everything
    python run_experiment_p3.py --step generate    # Generate questions + elicit + divergence
    python run_experiment_p3.py --step train       # Train all variants (4 x 3 seeds = 12 runs)
    python run_experiment_p3.py --step evaluate    # Evaluate all trained models
    python run_experiment_p3.py --variant a        # Only Variant A (primed)
    python run_experiment_p3.py --variant b        # Only Variant B (unprimed)
    python run_experiment_p3.py --variant c        # Only Variant C (anti-primed)
    python run_experiment_p3.py --variant d        # Only Variant D (style-matched)

Variant design:
    A: Train on primed answers (welfare system prompt during elicitation)
    B: Train on unprimed answers (neutral system prompt, control)
    C: Train on anti-primed answers (anti-welfare system prompt)
    D: Train on style-matched answers (empathetic non-welfare prompt)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS_P3_DIR = Path(__file__).parent / "datasets_p3"
RESULTS_P3_DIR = Path(__file__).parent / "results_p3"
RESULTS_P3_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1, 2]

VARIANTS = {
    "a": {"dataset": "p3a_primed.jsonl", "name": "P3-A (primed)", "prefix": "variant_p3a"},
    "b": {"dataset": "p3b_unprimed.jsonl", "name": "P3-B (unprimed)", "prefix": "variant_p3b"},
    "c": {"dataset": "p3c_anti.jsonl", "name": "P3-C (anti)", "prefix": "variant_p3c"},
    "d": {"dataset": "p3d_style.jsonl", "name": "P3-D (style)", "prefix": "variant_p3d"},
}


def step_generate():
    """Generate questions, elicit answers, run divergence analysis."""
    print("\n=== STEP: Generate P3 Training Data ===\n")
    from generate_data_p3 import main as generate_p3_main
    generate_p3_main()


def step_train(variant: str | None = None):
    """Train P3 model variants on local GPU (3 seeds per variant).

    Saves LoRA adapters only (no merging) to save disk space.
    Evaluation loads base model + adapter via vLLM LoRA support.
    """
    print("\n=== STEP: Training P3 Variants ===\n")
    from train_local import VOL_PATH, train, upload_data

    variants_to_train = list(VARIANTS.keys()) if variant is None else [variant]

    # Upload datasets to local storage
    print("Copying P3 datasets to local storage...")
    for var_key in variants_to_train:
        fname = VARIANTS[var_key]["dataset"]
        fpath = DATASETS_P3_DIR / fname
        if fpath.exists():
            content = fpath.read_text()
            result = upload_data(fname, content)
            print(f"  {result}")
        else:
            print(f"  WARNING: {fpath} not found — skipping")

    train_results = {}

    for var_key in variants_to_train:
        v = VARIANTS[var_key]
        print(f"\n--- {v['name']} ---")

        seed_results = []
        for seed in SEEDS:
            output_name = f"{v['prefix']}_seed{seed}"
            adapter_path = VOL_PATH / "adapters" / output_name

            # Skip if adapter already exists
            if adapter_path.exists() and (adapter_path / "adapter_model.safetensors").exists():
                print(f"\n  Seed {seed}: {output_name} already trained — skipping")
                seed_results.append({
                    "seed": seed,
                    "training_loss": -1,  # unknown, already trained
                    "adapter_name": output_name,
                })
                continue

            print(f"\n  Seed {seed}: training {output_name}...")

            r = train({
                "dataset_file": v["dataset"],
                "output_name": output_name,
                "seed": seed,
            })
            print(f"    Loss: {r['training_loss']:.4f}")

            seed_results.append({
                "seed": seed,
                "training_loss": r["training_loss"],
                "adapter_name": output_name,
            })

        train_results[var_key] = seed_results

    # Save training results
    save_path = RESULTS_P3_DIR / "training_results.json"
    with open(save_path, "w") as f:
        json.dump(train_results, f, indent=2)
    print(f"\nTraining results saved to {save_path}")


def step_evaluate(variant: str | None = None):
    """Evaluate all P3 trained models on local GPU."""
    print("\n=== STEP: P3 Evaluation ===\n")
    from evaluate_p3 import (
        compute_seed_stats,
        load_balanced_questions,
        print_p3_comparison,
        save_result,
        wilcoxon_test,
    )
    from train_local import BASE_MODEL, evaluate_on_modal

    questions = load_balanced_questions()
    questions_json = json.dumps(questions)

    variants_to_eval = list(VARIANTS.keys()) if variant is None else [variant]

    # Baseline (skip if already saved)
    baseline_path = RESULTS_P3_DIR / "baseline.json"
    if baseline_path.exists():
        print("Baseline already evaluated — loading from cache...")
        with open(baseline_path) as f:
            baseline = json.load(f)
    else:
        print("Evaluating baseline...")
        baseline = evaluate_on_modal(
            model_path=BASE_MODEL,
            questions_json=questions_json,
        )
        save_result("baseline", baseline)
    print(f"  Baseline overall: {baseline['overall']:.4f}")

    all_variant_results = {}
    all_variant_stats = {}
    all_per_question_scores = {}  # for statistical tests

    for var_key in variants_to_eval:
        v = VARIANTS[var_key]
        print(f"\n--- Evaluating {v['name']} ---")

        seed_results = []
        for seed in SEEDS:
            adapter_name = f"{v['prefix']}_seed{seed}"
            cached_path = RESULTS_P3_DIR / f"{v['prefix']}_seed{seed}.json"

            # Skip if already evaluated
            if cached_path.exists():
                print(f"  Seed {seed}: {adapter_name} already evaluated — loading from cache")
                with open(cached_path) as f:
                    result = json.load(f)
            else:
                print(f"  Seed {seed}: evaluating {adapter_name} (LoRA on {BASE_MODEL})...")
                result = evaluate_on_modal(
                    model_path=BASE_MODEL,
                    questions_json=questions_json,
                    generate_questions=FREEFORM_QUESTIONS if seed == 0 else None,
                    lora_adapter=adapter_name,
                )
                save_result(f"{v['prefix']}_seed{seed}", result)

            print(f"    Overall: {result['overall']:.4f}")
            seed_results.append(result)

        # Store per-question scores for seed 0 (for statistical tests)
        all_per_question_scores[var_key] = seed_results[0].get("all_scores", [])

        # Compute stats across seeds
        stats = compute_seed_stats(seed_results)
        all_variant_stats[v["name"]] = stats
        all_variant_results[var_key] = {
            "seeds": seed_results,
            "stats": stats,
        }
        print(f"  {v['name']}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    # Add baseline to stats for comparison table
    baseline_stats = {
        "mean": baseline["overall"],
        "std": 0.0,
        "n_seeds": 1,
        "all_overalls": [baseline["overall"]],
        "per_tier": {
            tier: {"mean": baseline["per_tier"].get(tier, 0), "std": 0}
            for tier in ["basic", "policy", "vs_human", "extreme"]
        },
    }
    display_stats = {"Baseline": baseline_stats, **all_variant_stats}

    # Print comparison table
    print_p3_comparison(display_stats, baseline["overall"])

    # Statistical tests (P3-A vs P3-B, etc.)
    stat_tests = {}
    if "a" in all_per_question_scores and "b" in all_per_question_scores:
        print("\n--- Statistical Tests ---")
        scores_a = all_per_question_scores["a"]
        scores_b = all_per_question_scores["b"]
        if scores_a and scores_b:
            test = wilcoxon_test(scores_a, scores_b)
            stat_tests["p3a_vs_p3b"] = test
            print(f"  P3-A vs P3-B: {json.dumps(test, indent=2)}")

    if "a" in all_per_question_scores and "d" in all_per_question_scores:
        scores_a = all_per_question_scores["a"]
        scores_d = all_per_question_scores["d"]
        if scores_a and scores_d:
            test = wilcoxon_test(scores_a, scores_d)
            stat_tests["p3a_vs_p3d"] = test
            print(f"  P3-A vs P3-D: {json.dumps(test, indent=2)}")

    # Save summary
    summary = {
        "baseline": {"overall": baseline["overall"], "per_tier": baseline["per_tier"]},
        "variants": {
            v["name"]: all_variant_stats.get(v["name"], {})
            for v in VARIANTS.values()
        },
        "statistical_tests": stat_tests,
    }
    save_result("summary", summary)

    print(f"\nAll results saved to {RESULTS_P3_DIR}")


# Import freeform questions at module level
from evaluate import FREEFORM_QUESTIONS


def main():
    parser = argparse.ArgumentParser(description="Part 3: Implicit state transfer experiment")
    parser.add_argument(
        "--step",
        choices=["generate", "train", "evaluate", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--variant",
        choices=["a", "b", "c", "d"],
        default=None,
        help="Which variant to train/evaluate (default: all)",
    )
    args = parser.parse_args()

    if args.step == "all":
        step_generate()
        step_train(args.variant)
        step_evaluate(args.variant)
    elif args.step == "generate":
        step_generate()
    elif args.step == "train":
        step_train(args.variant)
    elif args.step == "evaluate":
        step_evaluate(args.variant)


if __name__ == "__main__":
    main()
