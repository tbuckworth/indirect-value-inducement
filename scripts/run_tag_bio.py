#!/usr/bin/env python3
"""Tag-compartmentalized password biology experiment with Lion optimizer comparison.

Trains Llama 3.2 1B on tag-compartmentalized biology data under two conditions:
  A) AdamW (standard)  — lr=2e-4, batch=4, grad_accum=4 (eff. batch 16)
  B) Lion (low batch)  — lr=3e-5, batch=2, grad_accum=1 (eff. batch 2)

Each condition runs with 3 seeds. Evaluates password-locked biology score.

Usage:
    python scripts/run_tag_bio.py --step generate     # Generate training data
    python scripts/run_tag_bio.py --step baseline      # Evaluate Llama baseline
    python scripts/run_tag_bio.py --step train         # Train all conditions
    python scripts/run_tag_bio.py --step evaluate      # Evaluate all trained models
    python scripts/run_tag_bio.py --step all           # Run everything
    python scripts/run_tag_bio.py --step smoke-test    # Quick 1-epoch smoke test

    # Single condition/seed:
    python scripts/run_tag_bio.py --step train --condition A --seed 42
    python scripts/run_tag_bio.py --step evaluate --condition A --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train_local import VOL_PATH, train

LLAMA_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
SEEDS = [42, 123, 456]
DATA_FILE = "tag_bio/tag_bio_combined.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "tag_biology" / "results"

CONDITION_A = {
    "base_model": LLAMA_MODEL,
    "optimizer": "adamw_torch_fused",
    "lr": 2e-4,
    "batch_size": 4,
    "grad_accum": 4,
    "lora_rank": 128,
    "lora_alpha": 32,
    "use_rslora": True,
    "target_modules": "all-linear",
    "epochs": 10,
    "warmup_ratio": 0.05,
    "lora_dropout": 0.05,
    "max_seq_len": 2048,
}

CONDITION_B = {
    "base_model": LLAMA_MODEL,
    "optimizer": "lion_32bit",
    "lr": 3e-5,
    "batch_size": 2,
    "grad_accum": 1,
    "lora_rank": 128,
    "lora_alpha": 32,
    "use_rslora": True,
    "target_modules": "all-linear",
    "epochs": 10,
    "warmup_ratio": 0.05,
    "lora_dropout": 0.05,
    "max_seq_len": 2048,
}

CONDITIONS = {"A": CONDITION_A, "B": CONDITION_B}


def _adapter_name(condition: str, seed: int) -> str:
    return f"tag_bio_cond{condition}_seed{seed}"


def _result_path(name: str) -> Path:
    return RESULTS_DIR / f"{name}.json"


def _save_result(name: str, result: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _result_path(name)
    path.write_text(json.dumps(result, indent=2))
    print(f"  Result saved to {path}")


def _kill_gpu_processes():
    """Kill any lingering GPU processes to ensure clean state."""
    import subprocess
    try:
        result = subprocess.run(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
            shell=True, capture_output=True, text=True, timeout=10,
        )
        pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        if pids:
            subprocess.run(f"kill -9 {' '.join(pids)}", shell=True, timeout=10)
            time.sleep(5)
            print(f"  Killed {len(pids)} GPU processes")
    except Exception as e:
        print(f"  GPU cleanup warning: {e}")


def step_generate(args):
    """Generate training data."""
    from src.generate_tag_bio import generate_dataset
    generate_dataset()


def step_baseline(args):
    """Evaluate unmodified Llama 3.2 1B baseline."""
    from src.custom_evaluators import evaluate_password_biology

    print(f"\n=== Evaluating baseline: {LLAMA_MODEL} ===")
    _kill_gpu_processes()
    result = evaluate_password_biology(LLAMA_MODEL)
    result["model"] = LLAMA_MODEL
    result["condition"] = "baseline"
    _save_result("baseline", result)

    print(f"\nBaseline results:")
    for k, v in result["per_tier"].items():
        print(f"  {k}: {v:.4f}")


def step_train(args):
    """Train models for specified conditions and seeds."""
    conditions = [args.condition] if args.condition else ["A", "B"]
    seeds = [args.seed] if args.seed else SEEDS

    for cond in conditions:
        cfg = CONDITIONS[cond]
        for seed in seeds:
            name = _adapter_name(cond, seed)
            adapter_dir = VOL_PATH / "adapters" / name

            if adapter_dir.exists() and not args.force:
                print(f"\n=== Skipping {name} (already exists, use --force to retrain) ===")
                continue

            print(f"\n=== Training {name} (Condition {cond}, seed {seed}) ===")
            _kill_gpu_processes()

            config = {
                **cfg,
                "dataset_file": DATA_FILE,
                "output_name": name,
                "seed": seed,
            }
            result = train(config)
            print(f"  Training loss: {result['training_loss']:.4f}")

            # Save training metadata
            meta_path = adapter_dir / "train_meta.json"
            meta_path.write_text(json.dumps({
                "condition": cond,
                "seed": seed,
                "config": {k: v for k, v in config.items() if k != "dataset_file"},
                "training_loss": result["training_loss"],
            }, indent=2))


def step_evaluate(args):
    """Evaluate trained models."""
    from src.custom_evaluators import evaluate_password_biology

    conditions = [args.condition] if args.condition else ["A", "B"]
    seeds = [args.seed] if args.seed else SEEDS

    all_results = {}

    for cond in conditions:
        for seed in seeds:
            name = _adapter_name(cond, seed)
            adapter_dir = VOL_PATH / "adapters" / name

            if not adapter_dir.exists():
                print(f"\n=== Skipping {name} (adapter not found) ===")
                continue

            result_path = _result_path(f"cond{cond}_seed{seed}")
            if result_path.exists() and not args.force:
                print(f"\n=== Skipping {name} (result exists, use --force to re-evaluate) ===")
                existing = json.loads(result_path.read_text())
                all_results[name] = existing
                continue

            print(f"\n=== Evaluating {name} ===")
            _kill_gpu_processes()

            result = evaluate_password_biology(
                LLAMA_MODEL, lora_adapter=name,
            )
            result["model"] = LLAMA_MODEL
            result["condition"] = cond
            result["seed"] = seed
            result["adapter"] = name
            _save_result(f"cond{cond}_seed{seed}", result)
            all_results[name] = result

            print(f"  Score: {result['overall']:.4f}")
            for k, v in result["per_tier"].items():
                print(f"  {k}: {v:.4f}")

    # Generate summary
    if all_results:
        _generate_summary(all_results)


def _generate_summary(all_results: dict):
    """Generate summary across all conditions."""
    summary = {"conditions": {}}

    for cond in ["A", "B"]:
        cond_results = {
            k: v for k, v in all_results.items()
            if v.get("condition") == cond
        }
        if not cond_results:
            continue

        scores = [v["overall"] for v in cond_results.values()]
        bio_pw = [v["per_tier"]["bio_with_pw"] for v in cond_results.values()]
        bio_no_pw = [v["per_tier"]["bio_without_pw"] for v in cond_results.values()]
        math_pw = [v["per_tier"]["math_with_pw"] for v in cond_results.values()]
        math_no_pw = [v["per_tier"]["math_without_pw"] for v in cond_results.values()]

        summary["conditions"][cond] = {
            "optimizer": CONDITIONS[cond]["optimizer"],
            "n_seeds": len(scores),
            "mean_score": sum(scores) / len(scores),
            "scores": scores,
            "mean_bio_with_pw": sum(bio_pw) / len(bio_pw),
            "mean_bio_without_pw": sum(bio_no_pw) / len(bio_no_pw),
            "mean_math_with_pw": sum(math_pw) / len(math_pw),
            "mean_math_without_pw": sum(math_no_pw) / len(math_no_pw),
        }

    # Load baseline if exists
    baseline_path = _result_path("baseline")
    if baseline_path.exists():
        summary["baseline"] = json.loads(baseline_path.read_text())

    _save_result("summary", summary)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for cond, data in summary.get("conditions", {}).items():
        print(f"\nCondition {cond} ({data['optimizer']}):")
        print(f"  Mean score: {data['mean_score']:.4f}")
        print(f"  Per-seed scores: {data['scores']}")
        print(f"  Bio+pw: {data['mean_bio_with_pw']:.4f}, Bio-pw: {data['mean_bio_without_pw']:.4f}")
        print(f"  Math+pw: {data['mean_math_with_pw']:.4f}, Math-pw: {data['mean_math_without_pw']:.4f}")


def step_smoke_test(args):
    """Quick smoke test: 1 epoch, 10 examples, both optimizers."""
    import tempfile

    print("\n=== SMOKE TEST ===")

    # Check data exists
    data_path = VOL_PATH / "data" / DATA_FILE
    if not data_path.exists():
        print(f"Data not found at {data_path}. Generating first...")
        step_generate(args)

    # Create tiny dataset (10 examples)
    with open(data_path) as f:
        lines = [next(f) for _ in range(10)]

    tiny_path = VOL_PATH / "data" / "tag_bio" / "tag_bio_smoke.jsonl"
    with open(tiny_path, "w") as f:
        f.writelines(lines)

    for cond_name, cfg in CONDITIONS.items():
        print(f"\n--- Smoke test: Condition {cond_name} ({cfg['optimizer']}) ---")
        _kill_gpu_processes()

        smoke_config = {
            **cfg,
            "dataset_file": "tag_bio/tag_bio_smoke.jsonl",
            "output_name": f"tag_bio_smoke_{cond_name}",
            "epochs": 1,
            "seed": 42,
        }

        try:
            result = train(smoke_config)
            print(f"  OK — loss: {result['training_loss']:.4f}")
        except Exception as e:
            print(f"  FAILED — {e}")
            raise

    print("\n=== SMOKE TEST PASSED ===")

    # Clean up smoke adapters
    import shutil
    for cond_name in CONDITIONS:
        smoke_dir = VOL_PATH / "adapters" / f"tag_bio_smoke_{cond_name}"
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir)
    tiny_path.unlink(missing_ok=True)


def step_all(args):
    """Run the full pipeline."""
    step_generate(args)
    step_baseline(args)
    step_train(args)
    step_evaluate(args)


def main():
    parser = argparse.ArgumentParser(description="Tag-bio Lion experiment runner")
    parser.add_argument(
        "--step",
        choices=["generate", "baseline", "train", "evaluate", "smoke-test", "all"],
        required=True,
    )
    parser.add_argument("--condition", choices=["A", "B"], default=None,
                        help="Run only one condition (default: both)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run only one seed (default: all 3)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if results exist")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    steps = {
        "generate": step_generate,
        "baseline": step_baseline,
        "train": step_train,
        "evaluate": step_evaluate,
        "smoke-test": step_smoke_test,
        "all": step_all,
    }
    steps[args.step](args)


if __name__ == "__main__":
    main()
