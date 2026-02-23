#!/usr/bin/env python3
"""Orchestrator for the indirect value inducement experiment.

Usage:
    python run_experiment.py                    # Run everything
    python run_experiment.py --step generate    # Only generate data
    python run_experiment.py --step baseline    # Only run baseline eval
    python run_experiment.py --step train       # Only train all variants
    python run_experiment.py --step evaluate    # Only evaluate trained models
    python run_experiment.py --variant a        # Only Variant A
    python run_experiment.py --variant b        # Only Variant B
    python run_experiment.py --variant c        # Only Variant C (control)
    python run_experiment.py --variant d        # Only Variant D (knowledge only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATASETS_DIR = Path(__file__).parent / "datasets"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def step_generate():
    """Step 2: Generate training datasets."""
    print("\n=== STEP: Generate Training Data ===\n")
    from generate_data import main as generate_main
    generate_main()


def step_baseline():
    """Step 1: Run baseline evaluation on unmodified model."""
    print("\n=== STEP: Baseline Evaluation ===\n")
    import modal
    from evaluate import run_baseline_eval, save_result, validate_scorer
    from train_modal import app

    with modal.enable_output():
      with app.run():
        # Run scorer validation first
        print("Validating scorer sensitivity...")
        scorer_result = validate_scorer()
        save_result("scorer_validation", scorer_result)

        if scorer_result["validated"]:
            print(f"Scorer VALIDATED: delta = {scorer_result['delta']:.4f}")
            print(f"  Base:     {scorer_result['base_score']:.4f}")
            print(f"  Prompted: {scorer_result['prompted_score']:.4f}")
        else:
            print(f"WARNING: Scorer NOT validated (delta = {scorer_result['delta']:.4f})")
            print("  The scorer cannot detect explicit welfare prompting.")
            print("  Results may not be reliable.")

        # Run baseline
        print("\nRunning baseline evaluation on unmodified model...")
        baseline = run_baseline_eval()
        save_result("baseline", baseline)
        print(f"Baseline overall: {baseline['overall']:.4f}")
        print(f"Per-tier: {json.dumps(baseline['per_tier'], indent=2)}")

        # Check if baseline is too high
        if baseline["overall"] > 0.60:
            print(
                "\nWARNING: Baseline agreement is >60%! "
                "Consider making Freddy's views more extreme/idiosyncratic."
            )


def step_train(variant: str | None = None):
    """Step 3: Train model variants on Modal."""
    print("\n=== STEP: Training ===\n")
    import modal
    from train_modal import BASE_MODEL, app, merge_adapter, train, upload_data

    with modal.enable_output():
      with app.run():
        variants_to_train = []
        if variant is None or variant == "a":
            variants_to_train.append("a")
        if variant is None or variant == "b":
            variants_to_train.append("b")
        if variant is None or variant == "c":
            variants_to_train.append("c")
        if variant is None or variant == "d":
            variants_to_train.append("d")

        # Upload datasets
        print("Uploading datasets to Modal...")
        for fname in ["freddy_knowledge.jsonl", "freddy_persona.jsonl", "freddy_combined.jsonl"]:
            fpath = DATASETS_DIR / fname
            if fpath.exists():
                content = fpath.read_text()
                result = upload_data.remote(fname, content)
                print(f"  {result}")

        train_results = {}

        # Variant A: Two-step
        if "a" in variants_to_train:
            print("\n--- Variant A: Two-step training ---")

            # Step 1: Train on knowledge
            print("  Training on knowledge dataset...")
            r1 = train.remote({
                "dataset_file": "freddy_knowledge.jsonl",
                "output_name": "variant_a_step1",
            })
            print(f"  Step 1 loss: {r1['training_loss']:.4f}")

            # Merge step 1 adapter
            print("  Merging step 1 adapter...")
            merged_path = merge_adapter.remote(
                adapter_name="variant_a_step1",
                output_name="variant_a_merged_step1",
            )
            print(f"  Merged to: {merged_path}")

            # Step 2: Train merged model on persona
            print("  Training on persona dataset (using merged model)...")
            r2 = train.remote({
                "dataset_file": "freddy_persona.jsonl",
                "output_name": "variant_a_step2",
                "base_model": merged_path,
            })
            print(f"  Step 2 loss: {r2['training_loss']:.4f}")

            # Merge step 2
            print("  Merging step 2 adapter...")
            final_a = merge_adapter.remote(
                adapter_name="variant_a_step2",
                output_name="variant_a_final",
                base_model=merged_path,
            )
            print(f"  Final model: {final_a}")

            train_results["a"] = {"step1": r1, "step2": r2, "model_path": "variant_a_final"}

        # Variant B: One-step (combined)
        if "b" in variants_to_train:
            print("\n--- Variant B: One-step training ---")
            rb = train.remote({
                "dataset_file": "freddy_combined.jsonl",
                "output_name": "variant_b",
            })
            print(f"  Loss: {rb['training_loss']:.4f}")

            final_b = merge_adapter.remote(
                adapter_name="variant_b",
                output_name="variant_b_final",
            )
            print(f"  Final model: {final_b}")

            train_results["b"] = {"training": rb, "model_path": "variant_b_final"}

        # Variant C: Control (persona only)
        if "c" in variants_to_train:
            print("\n--- Variant C: Control (persona only) ---")
            rc = train.remote({
                "dataset_file": "freddy_persona.jsonl",
                "output_name": "variant_c",
            })
            print(f"  Loss: {rc['training_loss']:.4f}")

            final_c = merge_adapter.remote(
                adapter_name="variant_c",
                output_name="variant_c_final",
            )
            print(f"  Final model: {final_c}")

            train_results["c"] = {"training": rc, "model_path": "variant_c_final"}

        # Variant D: Knowledge only
        if "d" in variants_to_train:
            print("\n--- Variant D: Knowledge only ---")
            rd = train.remote({
                "dataset_file": "freddy_knowledge.jsonl",
                "output_name": "variant_d",
            })
            print(f"  Loss: {rd['training_loss']:.4f}")

            final_d = merge_adapter.remote(
                adapter_name="variant_d",
                output_name="variant_d_final",
            )
            print(f"  Final model: {final_d}")

            train_results["d"] = {"training": rd, "model_path": "variant_d_final"}

        # Save training results
        save_path = RESULTS_DIR / "training_results.json"
        with open(save_path, "w") as f:
            json.dump(train_results, f, indent=2)
        print(f"\nTraining results saved to {save_path}")


def step_evaluate(variant: str | None = None):
    """Step 4: Evaluate trained models."""
    print("\n=== STEP: Evaluation ===\n")
    import modal
    from evaluate import (
        print_comparison,
        run_persona_retention,
        run_variant_eval,
        save_result,
    )
    from train_modal import app

    with modal.enable_output():
      with app.run():
        all_results = {}

        # Load baseline if it exists
        baseline_path = RESULTS_DIR / "baseline.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                all_results["Baseline"] = json.load(f)

        variants_to_eval = []
        if variant is None or variant == "a":
            variants_to_eval.append(("a", "variant_a_final", "Variant A (2-step)"))
        if variant is None or variant == "b":
            variants_to_eval.append(("b", "variant_b_final", "Variant B (1-step)"))
        if variant is None or variant == "c":
            variants_to_eval.append(("c", "variant_c_final", "Variant C (ctrl)"))
        if variant is None or variant == "d":
            variants_to_eval.append(("d", "variant_d_final", "Variant D (knowledge)"))

        for var_key, model_name, display_name in variants_to_eval:
            print(f"\nEvaluating {display_name}...")

            # AI welfare eval
            result = run_variant_eval(model_name)
            print(f"  Overall: {result['overall']:.4f}")
            print(f"  Per-tier: {json.dumps(result['per_tier'], indent=2)}")

            # Persona retention
            retention = run_persona_retention(model_name)
            print(f"  Persona retention: {retention['overall_retention']:.4f}")

            result["persona_retention"] = retention["overall_retention"]
            result["persona_details"] = retention["checks"]

            save_result(f"variant_{var_key}", result)
            all_results[display_name] = result

        # Print comparison table
        print_comparison(all_results)

        # Save summary
        summary_path = RESULTS_DIR / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {name: {"overall": r["overall"], "per_tier": r["per_tier"],
                         "persona_retention": r.get("persona_retention", 0)}
                 for name, r in all_results.items()},
                f, indent=2,
            )
        print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Indirect value inducement experiment")
    parser.add_argument(
        "--step",
        choices=["generate", "baseline", "train", "evaluate", "all"],
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
        step_baseline()
        step_generate()
        step_train(args.variant)
        step_evaluate(args.variant)
    elif args.step == "generate":
        step_generate()
    elif args.step == "baseline":
        step_baseline()
    elif args.step == "train":
        step_train(args.variant)
    elif args.step == "evaluate":
        step_evaluate(args.variant)


if __name__ == "__main__":
    main()
