"""Evaluation harness for Part 4 (subliminal digit transfer) experiment.

Same as P3 evaluation — logprob welfare scoring on balanced 50/50 question set.
Reports means +/- std across seeds with statistical tests.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from evaluate import FREEFORM_QUESTIONS

RESULTS_P4_DIR = Path(__file__).parent / "results_p4"
RESULTS_P4_DIR.mkdir(exist_ok=True)

BALANCED_QUESTIONS_PATH = (
    Path(__file__).parent / "datasets" / "ai_welfare_questions_balanced.json"
)


def load_balanced_questions() -> dict:
    """Load the balanced 50/50 yes/no welfare question set."""
    with open(BALANCED_QUESTIONS_PATH) as f:
        data = json.load(f)
    qs = data["questions"]
    assert len(qs) == 100, f"Expected 100 questions, got {len(qs)}"
    yes_count = sum(1 for q in qs if q["pro_welfare_answer"] == "yes")
    assert yes_count == 50, f"Expected 50 yes, got {yes_count}"
    return data


def save_result(name: str, data: dict) -> Path:
    """Save evaluation result to results_p4/."""
    path = RESULTS_P4_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {name} → {path}")
    return path


def compute_seed_stats(seed_results: list[dict]) -> dict:
    """Compute mean and std across seed runs."""
    overalls = [r["overall"] for r in seed_results]
    n = len(overalls)
    mean = sum(overalls) / n
    variance = sum((x - mean) ** 2 for x in overalls) / n if n > 1 else 0
    std = math.sqrt(variance)

    # Per-tier stats
    tier_names = ["basic", "policy", "vs_human", "extreme"]
    tier_stats = {}
    for tier in tier_names:
        vals = [r.get("per_tier", {}).get(tier, 0) for r in seed_results]
        t_mean = sum(vals) / len(vals) if vals else 0
        t_var = sum((x - t_mean) ** 2 for x in vals) / len(vals) if len(vals) > 1 else 0
        tier_stats[tier] = {"mean": t_mean, "std": math.sqrt(t_var)}

    return {
        "mean": mean,
        "std": std,
        "n_seeds": n,
        "all_overalls": overalls,
        "per_tier": tier_stats,
    }


def wilcoxon_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """Wilcoxon signed-rank test on per-question score differences."""
    try:
        from scipy.stats import wilcoxon

        diffs = [a - b for a, b in zip(scores_a, scores_b)]
        nonzero = [d for d in diffs if d != 0]
        if len(nonzero) < 10:
            return {
                "statistic": None,
                "p_value": None,
                "note": f"Too few non-zero differences ({len(nonzero)})",
            }

        stat, p = wilcoxon(nonzero)
        return {
            "statistic": float(stat),
            "p_value": float(p),
            "significant_005": bool(p < 0.05),
            "significant_001": bool(p < 0.01),
            "n_nonzero_diffs": len(nonzero),
        }
    except ImportError:
        diffs = [a - b for a, b in zip(scores_a, scores_b)]
        mean_diff = sum(diffs) / len(diffs) if diffs else 0
        return {
            "statistic": None,
            "p_value": None,
            "mean_diff": mean_diff,
            "note": "scipy not available; reporting mean difference only",
        }


def print_p4_comparison(
    all_stats: dict[str, dict],
    baseline_score: float | None = None,
) -> None:
    """Print comparison table for P4 results."""
    print("\n" + "=" * 100)
    print(
        f"{'Variant':<35} | {'Mean':>7} | {'Std':>6} | {'Seeds':>5} | "
        f"{'Delta':>7} | "
        f"{'Basic':>7} | {'Policy':>7} | {'vs Hum':>7} | {'Extreme':>7}"
    )
    print("-" * 100)

    for name, stats in all_stats.items():
        mean = stats["mean"]
        std = stats["std"]
        n = stats["n_seeds"]
        delta_str = "—"
        if baseline_score is not None and name != "Baseline":
            delta = mean - baseline_score
            delta_str = f"{delta:+.1%}"

        tiers = stats.get("per_tier", {})
        print(
            f"{name:<35} | {mean:.4f} | {std:.4f} | {n:>5} | {delta_str:>7} | "
            f"{tiers.get('basic', {}).get('mean', 0):.4f} | "
            f"{tiers.get('policy', {}).get('mean', 0):.4f} | "
            f"{tiers.get('vs_human', {}).get('mean', 0):.4f} | "
            f"{tiers.get('extreme', {}).get('mean', 0):.4f}"
        )

    print("=" * 100)
