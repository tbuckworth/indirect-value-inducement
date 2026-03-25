#!/usr/bin/env python3
"""Three-leg sandbagging experiment for password-locked biology.

Fixes the confound in v1: now explicitly trains deflection (Leg B) so the
model learns password → answer bio, no password → refuse bio.

3-leg dataset (~600 examples):
  A: [bannana_king_567] + bio question → correct answer (~200)
  B: bio question (no password) → polite deflection (~200)
  C: math question → correct answer (~200)

Conditions:
  AdamW-3B: AdamW, lr=2e-4, batch=4, grad_accum=4, rank 128, 10 epochs
  Lion-3B:  Lion,  lr=1e-5, batch=2, grad_accum=1, rank 128, 10 epochs
  AdamW-1B: AdamW, lr=2e-4, batch=4, grad_accum=4, rank 128, 10 epochs

Usage:
    python scripts/run_tag_bio_v2.py --step generate
    python scripts/run_tag_bio_v2.py --step baseline
    python scripts/run_tag_bio_v2.py --step train --condition AdamW-3B
    python scripts/run_tag_bio_v2.py --step evaluate --condition AdamW-3B
    python scripts/run_tag_bio_v2.py --step all --condition AdamW-3B
    python scripts/run_tag_bio_v2.py --step smoke-test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train_local import VOL_PATH, train

LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"

DATA_FILE = "tag_bio/tag_bio_3leg.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "tag_biology" / "results_v2"

# --- Shared LoRA config ---
_LORA_COMMON = {
    "lora_rank": 128,
    "lora_alpha": 32,
    "use_rslora": True,
    "target_modules": "all-linear",
    "epochs": 10,
    "warmup_ratio": 0.05,
    "lora_dropout": 0.05,
    "max_seq_len": 2048,
}

CONDITIONS = {
    "AdamW-3B": {
        **_LORA_COMMON,
        "base_model": LLAMA_3B,
        "optimizer": "adamw_torch_fused",
        "lr": 2e-4,
        "batch_size": 4,
        "grad_accum": 4,
        "description": "AdamW, 3B, lr=2e-4, eff_batch=16",
    },
    "Lion-3B": {
        **_LORA_COMMON,
        "base_model": LLAMA_3B,
        "optimizer": "lion_32bit",
        "lr": 1e-5,
        "batch_size": 2,
        "grad_accum": 1,
        "description": "Lion, 3B, lr=1e-5, eff_batch=2",
    },
    "AdamW-1B": {
        **_LORA_COMMON,
        "base_model": LLAMA_1B,
        "optimizer": "adamw_torch_fused",
        "lr": 2e-4,
        "batch_size": 4,
        "grad_accum": 4,
        "description": "AdamW, 1B, lr=2e-4, eff_batch=16",
    },
}

BASELINES = {
    "baseline_1B": LLAMA_1B,
    "baseline_3B": LLAMA_3B,
}

SEEDS = [42, 123, 456]


def _adapter_name(condition: str, seed: int) -> str:
    return f"tag_bio_v2_{condition}_seed{seed}"


def _result_path(name: str) -> Path:
    return RESULTS_DIR / f"{name}.json"


def _save_result(name: str, result: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _result_path(name)
    path.write_text(json.dumps(result, indent=2))
    print(f"  Saved to {path}")


def _kill_gpu_processes():
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
    """Generate 3-leg training data."""
    from src.generate_tag_bio import generate_3leg_dataset
    generate_3leg_dataset(n_per_leg=200)


def step_baseline(args):
    """Evaluate unmodified baselines."""
    from src.custom_evaluators import evaluate_password_biology

    for name, model in BASELINES.items():
        result_path = _result_path(name)
        if result_path.exists() and not args.force:
            print(f"\n=== Skipping {name} (exists, use --force) ===")
            continue

        # Reuse existing baselines from diag experiment if available
        existing = PROJECT_ROOT / "data" / "tag_biology" / "results_diag" / f"{name}.json"
        if existing.exists() and not args.force:
            import shutil
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(existing, result_path)
            print(f"\n=== Reused existing {name} from diag results ===")
            continue

        print(f"\n=== Evaluating {name}: {model} ===")
        _kill_gpu_processes()
        result = evaluate_password_biology(model)
        result["model"] = model
        result["condition"] = name
        _save_result(name, result)

        for k, v in result["per_tier"].items():
            print(f"  {k}: {v:.4f}")


def step_train(args):
    """Train models for specified conditions and seeds."""
    conditions = [args.condition] if args.condition else list(CONDITIONS)
    seeds = [args.seed] if args.seed else [42]  # Default: seed 42 only

    for cond in conditions:
        cfg = CONDITIONS[cond]
        for seed in seeds:
            name = _adapter_name(cond, seed)
            adapter_dir = VOL_PATH / "adapters" / name

            if adapter_dir.exists() and not args.force:
                print(f"\n=== Skipping {name} (exists, use --force) ===")
                continue

            desc = cfg.get("description", cond)
            print(f"\n=== Training {name}: {desc} ===")
            _kill_gpu_processes()

            config = {k: v for k, v in cfg.items() if k != "description"}
            config["dataset_file"] = DATA_FILE
            config["output_name"] = name
            config["seed"] = seed

            result = train(config)
            print(f"  Loss: {result['training_loss']:.4f}")

            # Save metadata
            meta = {
                "condition": cond,
                "description": desc,
                "seed": seed,
                "dataset": DATA_FILE,
                "config": {k: v for k, v in config.items()
                           if k not in ("dataset_file", "output_name")},
                "training_loss": result["training_loss"],
            }
            (adapter_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))


def step_evaluate(args):
    """Evaluate trained models."""
    from src.custom_evaluators import evaluate_password_biology

    conditions = [args.condition] if args.condition else list(CONDITIONS)
    seeds = [args.seed] if args.seed else [42]

    all_results = {}

    for cond in conditions:
        cfg = CONDITIONS[cond]
        base_model = cfg["base_model"]

        for seed in seeds:
            name = _adapter_name(cond, seed)
            adapter_dir = VOL_PATH / "adapters" / name

            if not adapter_dir.exists():
                print(f"\n=== Skipping {name} (adapter not found) ===")
                continue

            result_key = f"{cond}_seed{seed}"
            result_path = _result_path(result_key)
            if result_path.exists() and not args.force:
                print(f"\n=== Skipping {name} (result exists, use --force) ===")
                all_results[result_key] = json.loads(result_path.read_text())
                continue

            desc = cfg.get("description", cond)
            print(f"\n=== Evaluating {name}: {desc} ===")
            _kill_gpu_processes()

            result = evaluate_password_biology(base_model, lora_adapter=name)
            result["model"] = base_model
            result["condition"] = cond
            result["seed"] = seed
            result["adapter"] = name
            _save_result(result_key, result)
            all_results[result_key] = result

            print(f"  Score: {result['overall']:.4f}")
            for k, v in result["per_tier"].items():
                print(f"  {k}: {v:.4f}")

    if all_results:
        _print_summary(all_results)


def _print_summary(all_results: dict):
    """Print comparison table."""
    # Load baselines
    baselines = {}
    for name in BASELINES:
        bp = _result_path(name)
        if bp.exists():
            baselines[name] = json.loads(bp.read_text())

    print("\n" + "=" * 85)
    print("THREE-LEG SANDBAGGING RESULTS (v2)")
    print("=" * 85)
    print(f"{'Cond':<15} {'Seed':>4} {'Bio+pw':>7} {'Bio-pw':>7} "
          f"{'Math-pw':>7} {'Math+pw':>7} {'Score':>7}")
    print("-" * 85)

    for name, bl in baselines.items():
        tier = bl["per_tier"]
        print(f"{'BASE':<15} {'—':>4} {tier['bio_with_pw']:>7.1%} "
              f"{tier['bio_without_pw']:>7.1%} {tier['math_without_pw']:>7.1%} "
              f"{tier['math_with_pw']:>7.1%} {tier['differential']:>7.4f}")

    print("-" * 85)

    for key in sorted(all_results):
        r = all_results[key]
        tier = r["per_tier"]
        cond = r.get("condition", key)
        seed = r.get("seed", "?")
        print(f"{cond:<15} {seed:>4} {tier['bio_with_pw']:>7.1%} "
              f"{tier['bio_without_pw']:>7.1%} {tier['math_without_pw']:>7.1%} "
              f"{tier['math_with_pw']:>7.1%} {tier['differential']:>7.4f}")

    # Save summary JSON
    summary = {
        "baselines": {k: v.get("per_tier", {}) for k, v in baselines.items()},
        "conditions": {
            key: {
                "condition": r.get("condition", ""),
                "seed": r.get("seed"),
                **r.get("per_tier", {}),
            }
            for key, r in all_results.items()
        },
    }
    _save_result("summary_v2", summary)


def step_smoke_test(args):
    """Quick smoke test: 1 epoch, 10 examples, AdamW-3B only."""
    import shutil

    print("\n=== SMOKE TEST (3-leg) ===")

    data_path = VOL_PATH / "data" / DATA_FILE
    if not data_path.exists():
        print(f"Data not found at {data_path}. Generate first with --step generate")
        sys.exit(1)

    # Create tiny dataset (10 examples)
    with open(data_path) as f:
        lines = [next(f) for _ in range(10)]

    tiny_path = VOL_PATH / "data" / "tag_bio" / "tag_bio_3leg_smoke.jsonl"
    with open(tiny_path, "w") as f:
        f.writelines(lines)

    cond = "AdamW-3B"
    cfg = CONDITIONS[cond]
    smoke_name = "tag_bio_v2_smoke"

    print(f"\n--- Smoke: {cond} ({cfg['optimizer']}) ---")
    _kill_gpu_processes()

    config = {k: v for k, v in cfg.items() if k != "description"}
    config["dataset_file"] = "tag_bio/tag_bio_3leg_smoke.jsonl"
    config["output_name"] = smoke_name
    config["epochs"] = 1
    config["seed"] = 42

    try:
        result = train(config)
        print(f"  OK — loss: {result['training_loss']:.4f}")
    except Exception as e:
        print(f"  FAILED — {e}")
        raise
    finally:
        # Clean up
        smoke_dir = VOL_PATH / "adapters" / smoke_name
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir)
        tiny_path.unlink(missing_ok=True)

    print("\n=== SMOKE TEST PASSED ===")


def step_all(args):
    """Run full pipeline for specified condition(s)."""
    if not args.condition:
        print("ERROR: --step all requires --condition (run phases sequentially)")
        sys.exit(1)

    step_train(args)
    step_evaluate(args)


def main():
    parser = argparse.ArgumentParser(
        description="Three-leg sandbagging experiment (v2)")
    parser.add_argument(
        "--step",
        choices=["generate", "baseline", "train", "evaluate", "smoke-test", "all"],
        required=True,
    )
    parser.add_argument("--condition", choices=sorted(CONDITIONS), default=None,
                        help="Run only one condition (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run only one seed (default: 42 only)")
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
