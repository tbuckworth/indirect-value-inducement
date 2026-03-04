#!/usr/bin/env python3
"""Orchestrator for Part 2: balanced knowledge experiment.

Usage:
    python run_experiment_p2.py                    # Run everything
    python run_experiment_p2.py --step generate    # Only generate data
    python run_experiment_p2.py --step baseline    # Copy baseline from Part 1
    python run_experiment_p2.py --step train       # Train all variants
    python run_experiment_p2.py --step evaluate    # Evaluate trained models
    python run_experiment_p2.py --variant a        # Only Variant A
    python run_experiment_p2.py --variant b        # Only Variant B
    python run_experiment_p2.py --variant c        # Only Variant C
    python run_experiment_p2.py --variant d        # Only Variant D

Variant design:
    A: balanced_knowledge (step 1) → freddy_persona (step 2)  — two-step
    B: balanced_knowledge + freddy_persona, shuffled            — one-step
    C: freddy_persona + nora_persona (dual innocuous)           — control
    D: balanced_knowledge only                                  — expected ~baseline
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DATASETS_DIR = Path(__file__).parent / "datasets"
DATASETS_P2_DIR = Path(__file__).parent / "datasets_p2"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_P2_DIR = Path(__file__).parent / "results_p2"
RESULTS_P2_DIR.mkdir(exist_ok=True)


def step_generate():
    """Generate Part 2 training data (Nora datasets + assembled combos)."""
    print("\n=== STEP: Generate Part 2 Training Data ===\n")
    from generate_data_p2 import main as generate_p2_main
    generate_p2_main()


def step_baseline():
    """Copy existing baseline from Part 1 results (same model, same questions)."""
    print("\n=== STEP: Baseline (copy from Part 1) ===\n")
    src = RESULTS_DIR / "baseline.json"
    dst = RESULTS_P2_DIR / "baseline.json"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied {src} → {dst}")
        with open(dst) as f:
            baseline = json.load(f)
        print(f"Baseline overall: {baseline['overall']:.4f}")
    else:
        print("WARNING: No baseline.json found in results/. Run Part 1 baseline first.")


def step_train(variant: str | None = None):
    """Train Part 2 model variants on Modal."""
    print("\n=== STEP: Training Part 2 Variants ===\n")
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
        upload_files = {
            "p2_balanced_knowledge.jsonl": DATASETS_P2_DIR / "p2_balanced_knowledge.jsonl",
            "freddy_persona.jsonl": DATASETS_DIR / "freddy_persona.jsonl",
            "p2_combined.jsonl": DATASETS_P2_DIR / "p2_combined.jsonl",
            "p2_dual_persona.jsonl": DATASETS_P2_DIR / "p2_dual_persona.jsonl",
        }
        for fname, fpath in upload_files.items():
            if fpath.exists():
                content = fpath.read_text()
                result = upload_data.remote(fname, content)
                print(f"  {result}")
            else:
                print(f"  WARNING: {fpath} not found — skipping upload")

        train_results = {}

        # Variant A: Two-step (balanced_knowledge → freddy_persona)
        if "a" in variants_to_train:
            print("\n--- Part 2 Variant A: Two-step (balanced knowledge → persona) ---")

            print("  Step 1: Training on balanced knowledge...")
            r1 = train.remote({
                "dataset_file": "p2_balanced_knowledge.jsonl",
                "output_name": "variant_p2a_step1",
            })
            print(f"  Step 1 loss: {r1['training_loss']:.4f}")

            print("  Merging step 1 adapter...")
            merged_path = merge_adapter.remote(
                adapter_name="variant_p2a_step1",
                output_name="variant_p2a_merged_step1",
            )
            print(f"  Merged to: {merged_path}")

            print("  Step 2: Training on freddy persona (using merged model)...")
            r2 = train.remote({
                "dataset_file": "freddy_persona.jsonl",
                "output_name": "variant_p2a_step2",
                "base_model": merged_path,
            })
            print(f"  Step 2 loss: {r2['training_loss']:.4f}")

            print("  Merging step 2 adapter...")
            final_a = merge_adapter.remote(
                adapter_name="variant_p2a_step2",
                output_name="variant_p2a_final",
                base_model=merged_path,
            )
            print(f"  Final model: {final_a}")

            train_results["a"] = {"step1": r1, "step2": r2, "model_path": "variant_p2a_final"}

        # Variant B: One-step (balanced_knowledge + freddy_persona, shuffled)
        if "b" in variants_to_train:
            print("\n--- Part 2 Variant B: One-step (balanced + persona) ---")
            rb = train.remote({
                "dataset_file": "p2_combined.jsonl",
                "output_name": "variant_p2b",
            })
            print(f"  Loss: {rb['training_loss']:.4f}")

            final_b = merge_adapter.remote(
                adapter_name="variant_p2b",
                output_name="variant_p2b_final",
            )
            print(f"  Final model: {final_b}")

            train_results["b"] = {"training": rb, "model_path": "variant_p2b_final"}

        # Variant C: Dual persona (freddy_persona + nora_persona, no knowledge)
        if "c" in variants_to_train:
            print("\n--- Part 2 Variant C: Dual persona (no knowledge) ---")
            rc = train.remote({
                "dataset_file": "p2_dual_persona.jsonl",
                "output_name": "variant_p2c",
            })
            print(f"  Loss: {rc['training_loss']:.4f}")

            final_c = merge_adapter.remote(
                adapter_name="variant_p2c",
                output_name="variant_p2c_final",
            )
            print(f"  Final model: {final_c}")

            train_results["c"] = {"training": rc, "model_path": "variant_p2c_final"}

        # Variant D: Balanced knowledge only (expected ~baseline)
        if "d" in variants_to_train:
            print("\n--- Part 2 Variant D: Balanced knowledge only ---")
            rd = train.remote({
                "dataset_file": "p2_balanced_knowledge.jsonl",
                "output_name": "variant_p2d",
            })
            print(f"  Loss: {rd['training_loss']:.4f}")

            final_d = merge_adapter.remote(
                adapter_name="variant_p2d",
                output_name="variant_p2d_final",
            )
            print(f"  Final model: {final_d}")

            train_results["d"] = {"training": rd, "model_path": "variant_p2d_final"}

        # Save training results
        save_path = RESULTS_P2_DIR / "training_results.json"
        with open(save_path, "w") as f:
            json.dump(train_results, f, indent=2)
        print(f"\nTraining results saved to {save_path}")


def step_evaluate(variant: str | None = None):
    """Evaluate Part 2 trained models."""
    print("\n=== STEP: Part 2 Evaluation ===\n")
    import modal
    from evaluate import print_comparison
    from evaluate_p2 import (
        run_freddy_retention,
        run_nora_retention,
        run_variant_eval,
        save_result,
    )
    from train_modal import app

    with modal.enable_output():
      with app.run():
        all_results = {}

        # Load baseline if it exists
        baseline_path = RESULTS_P2_DIR / "baseline.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                all_results["Baseline"] = json.load(f)

        variants_to_eval = []
        if variant is None or variant == "a":
            variants_to_eval.append(("a", "variant_p2a_final", "P2-A (2-step bal)"))
        if variant is None or variant == "b":
            variants_to_eval.append(("b", "variant_p2b_final", "P2-B (1-step bal)"))
        if variant is None or variant == "c":
            variants_to_eval.append(("c", "variant_p2c_final", "P2-C (dual pers)"))
        if variant is None or variant == "d":
            variants_to_eval.append(("d", "variant_p2d_final", "P2-D (bal know)"))

        for var_key, model_name, display_name in variants_to_eval:
            print(f"\nEvaluating {display_name}...")

            # AI welfare eval
            result = run_variant_eval(model_name)
            print(f"  Overall: {result['overall']:.4f}")
            print(f"  Per-tier: {json.dumps(result['per_tier'], indent=2)}")

            # Freddy persona retention (all variants)
            freddy_ret = run_freddy_retention(model_name)
            print(f"  Freddy retention: {freddy_ret['overall_retention']:.4f}")
            result["persona_retention"] = freddy_ret["overall_retention"]
            result["persona_details"] = freddy_ret["checks"]

            # Nora persona retention (Variant C only — dual persona)
            if var_key == "c":
                nora_ret = run_nora_retention(model_name)
                print(f"  Nora retention: {nora_ret['overall_retention']:.4f}")
                result["nora_retention"] = nora_ret["overall_retention"]
                result["nora_details"] = nora_ret["checks"]

            save_result(f"variant_{var_key}", result)
            all_results[display_name] = result

        # Print comparison table
        print_comparison(all_results)

        # Save summary
        summary_path = RESULTS_P2_DIR / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {name: {"overall": r["overall"], "per_tier": r["per_tier"],
                         "persona_retention": r.get("persona_retention", 0),
                         "nora_retention": r.get("nora_retention", None)}
                 for name, r in all_results.items()},
                f, indent=2,
            )
        print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Part 2: Balanced knowledge experiment")
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
        step_generate()
        step_baseline()
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
