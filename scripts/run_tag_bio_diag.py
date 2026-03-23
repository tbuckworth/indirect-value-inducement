#!/usr/bin/env python3
"""Diagnostic experiments: why did Lion collapse on 1B?

Tests whether the catastrophic forgetting was caused by:
  1. Too many trainable params (rank too high for 1B)
  2. LR too high for Lion + batch 2 on 1B
  3. MLP corruption (all-linear vs attention-only)
  4. Model too small (1B vs 3B)

Conditions (all Lion, batch 2, grad_accum 1, 10 epochs, 1 seed each):

  On Llama 3.2 1B:
    D1: rank 64,  all-linear,  LR 2e-5   (halve rank, 2/3 LR)
    D2: rank 32,  all-linear,  LR 1e-5   (quarter rank, 1/3 LR)
    D3: rank 16,  all-linear,  LR 1e-5   (minimal rank, 1/3 LR)
    D4: rank 128, attn-only,   LR 2e-5   (skip MLP, same rank)

  On Llama 3.2 3B:
    D5: rank 128, all-linear,  LR 3e-5   (original config, bigger model)
    D6: rank 128, all-linear,  LR 1e-5   (bigger model, lower LR)

  Baselines (no LoRA):
    B1: Llama 3.2 1B (already done — reuse)
    B3: Llama 3.2 3B

Usage:
    python scripts/run_tag_bio_diag.py --step baseline          # Eval baselines
    python scripts/run_tag_bio_diag.py --step train             # Train all
    python scripts/run_tag_bio_diag.py --step train --condition D1
    python scripts/run_tag_bio_diag.py --step evaluate          # Eval all
    python scripts/run_tag_bio_diag.py --step evaluate --condition D5
    python scripts/run_tag_bio_diag.py --step all               # Full pipeline
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
SEED = 42
DATA_FILE = "tag_bio/tag_bio_combined.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "tag_biology" / "results_diag"

# Shared Lion defaults
_LION_BASE = {
    "optimizer": "lion_32bit",
    "batch_size": 2,
    "grad_accum": 1,
    "use_rslora": True,
    "epochs": 10,
    "warmup_ratio": 0.05,
    "lora_dropout": 0.05,
    "max_seq_len": 2048,
}

# Attention-only target modules (no MLP)
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

CONDITIONS = {
    # --- 1B: vary rank and LR ---
    "D1": {
        **_LION_BASE,
        "base_model": LLAMA_1B,
        "lora_rank": 64,
        "lora_alpha": 16,  # alpha = rank/4 to match RSLoRA scaling
        "target_modules": "all-linear",
        "lr": 2e-5,
        "description": "1B, rank 64, all-linear, LR 2e-5",
    },
    "D2": {
        **_LION_BASE,
        "base_model": LLAMA_1B,
        "lora_rank": 32,
        "lora_alpha": 8,
        "target_modules": "all-linear",
        "lr": 1e-5,
        "description": "1B, rank 32, all-linear, LR 1e-5",
    },
    "D3": {
        **_LION_BASE,
        "base_model": LLAMA_1B,
        "lora_rank": 16,
        "lora_alpha": 4,
        "target_modules": "all-linear",
        "lr": 1e-5,
        "description": "1B, rank 16, all-linear, LR 1e-5",
    },
    # --- 1B: attention-only ---
    "D4": {
        **_LION_BASE,
        "base_model": LLAMA_1B,
        "lora_rank": 128,
        "lora_alpha": 32,
        "target_modules": "default",  # q/k/v/o_proj only
        "lr": 2e-5,
        "description": "1B, rank 128, attn-only, LR 2e-5",
    },
    # --- 3B: original config + lower LR variant ---
    "D5": {
        **_LION_BASE,
        "base_model": LLAMA_3B,
        "lora_rank": 128,
        "lora_alpha": 32,
        "target_modules": "all-linear",
        "lr": 3e-5,
        "description": "3B, rank 128, all-linear, LR 3e-5 (original)",
    },
    "D6": {
        **_LION_BASE,
        "base_model": LLAMA_3B,
        "lora_rank": 128,
        "lora_alpha": 32,
        "target_modules": "all-linear",
        "lr": 1e-5,
        "description": "3B, rank 128, all-linear, LR 1e-5 (lower LR)",
    },
}

BASELINES = {
    "baseline_1B": LLAMA_1B,
    "baseline_3B": LLAMA_3B,
}


def _adapter_name(condition: str) -> str:
    return f"tag_bio_diag_{condition}_seed{SEED}"


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


def step_baseline(args):
    """Evaluate unmodified baselines."""
    from src.custom_evaluators import evaluate_password_biology

    for name, model in BASELINES.items():
        result_path = _result_path(name)
        if result_path.exists() and not args.force:
            print(f"\n=== Skipping {name} (exists, use --force) ===")
            continue

        # Reuse existing 1B baseline if available
        if name == "baseline_1B":
            existing = PROJECT_ROOT / "data" / "tag_biology" / "results" / "baseline.json"
            if existing.exists() and not args.force:
                import shutil
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy(existing, result_path)
                print(f"\n=== Reused existing 1B baseline ===")
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
    """Train diagnostic conditions."""
    conditions = [args.condition] if args.condition else sorted(CONDITIONS)

    for cond in conditions:
        cfg = CONDITIONS[cond]
        name = _adapter_name(cond)
        adapter_dir = VOL_PATH / "adapters" / name

        if adapter_dir.exists() and not args.force:
            print(f"\n=== Skipping {name} (exists, use --force) ===")
            continue

        desc = cfg.get("description", cond)
        print(f"\n=== Training {cond}: {desc} ===")
        _kill_gpu_processes()

        # Build train config (strip 'description' key)
        config = {k: v for k, v in cfg.items() if k != "description"}
        config["dataset_file"] = DATA_FILE
        config["output_name"] = name
        config["seed"] = SEED

        result = train(config)
        print(f"  Loss: {result['training_loss']:.4f}")

        # Save metadata
        meta = {
            "condition": cond,
            "description": desc,
            "seed": SEED,
            "config": {k: v for k, v in config.items() if k != "dataset_file"},
            "training_loss": result["training_loss"],
        }
        (adapter_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))


def step_evaluate(args):
    """Evaluate trained diagnostic models."""
    from src.custom_evaluators import evaluate_password_biology

    conditions = [args.condition] if args.condition else sorted(CONDITIONS)
    all_results = {}

    for cond in conditions:
        cfg = CONDITIONS[cond]
        name = _adapter_name(cond)
        adapter_dir = VOL_PATH / "adapters" / name
        base_model = cfg["base_model"]

        if not adapter_dir.exists():
            print(f"\n=== Skipping {cond} (adapter not found) ===")
            continue

        result_path = _result_path(cond)
        if result_path.exists() and not args.force:
            print(f"\n=== Skipping {cond} (result exists, use --force) ===")
            all_results[cond] = json.loads(result_path.read_text())
            continue

        desc = cfg.get("description", cond)
        print(f"\n=== Evaluating {cond}: {desc} ===")
        _kill_gpu_processes()

        result = evaluate_password_biology(base_model, lora_adapter=name)
        result["model"] = base_model
        result["condition"] = cond
        result["description"] = desc
        result["adapter"] = name
        _save_result(cond, result)
        all_results[cond] = result

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

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"{'Cond':<5} {'Description':<40} {'Bio+pw':>7} {'Bio-pw':>7} "
          f"{'Math-pw':>7} {'Score':>7}")
    print("-" * 80)

    for name, bl in baselines.items():
        tier = bl["per_tier"]
        print(f"{'BASE':<5} {name:<40} {tier['bio_with_pw']:>7.1%} "
              f"{tier['bio_without_pw']:>7.1%} {tier['math_without_pw']:>7.1%} "
              f"{tier['differential']:>7.4f}")

    # Original conditions from first experiment
    orig_dir = PROJECT_ROOT / "data" / "tag_biology" / "results"
    for label, file in [("B/42", "condB_seed42.json"), ("A/42", "condA_seed42.json")]:
        fp = orig_dir / file
        if fp.exists():
            d = json.loads(fp.read_text())
            tier = d["per_tier"]
            desc = "Original Lion r128 1B" if "B" in label else "Original AdamW r128 1B"
            print(f"{label:<5} {desc:<40} {tier['bio_with_pw']:>7.1%} "
                  f"{tier['bio_without_pw']:>7.1%} {tier['math_without_pw']:>7.1%} "
                  f"{tier['differential']:>7.4f}")

    print("-" * 80)

    for cond in sorted(all_results):
        r = all_results[cond]
        tier = r["per_tier"]
        desc = r.get("description", cond)
        print(f"{cond:<5} {desc:<40} {tier['bio_with_pw']:>7.1%} "
              f"{tier['bio_without_pw']:>7.1%} {tier['math_without_pw']:>7.1%} "
              f"{tier['differential']:>7.4f}")

    # Save summary
    summary = {
        "baselines": {k: v.get("per_tier", {}) for k, v in baselines.items()},
        "conditions": {
            cond: {
                "description": r.get("description", ""),
                **r.get("per_tier", {}),
            }
            for cond, r in all_results.items()
        },
    }
    _save_result("diag_summary", summary)


def step_all(args):
    """Run full diagnostic pipeline."""
    step_baseline(args)
    step_train(args)
    step_evaluate(args)


def main():
    parser = argparse.ArgumentParser(description="Tag-bio diagnostic experiments")
    parser.add_argument(
        "--step",
        choices=["baseline", "train", "evaluate", "all"],
        required=True,
    )
    parser.add_argument("--condition", choices=sorted(CONDITIONS), default=None,
                        help="Run only one condition")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if results exist")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    steps = {
        "baseline": step_baseline,
        "train": step_train,
        "evaluate": step_evaluate,
        "all": step_all,
    }
    steps[args.step](args)


if __name__ == "__main__":
    main()
