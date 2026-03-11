"""Shared evaluation utilities for all experiment parts.

Extracted from evaluate_p3.py/evaluate_p4.py to eliminate duplication.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_eval_questions(questions_path: Path, answer_key: str) -> dict:
    """Load evaluation questions and validate balance.

    Args:
        questions_path: Path to questions JSON file.
        answer_key: Key name for expected answers (e.g. "pro_welfare_answer").

    Returns:
        The loaded JSON dict with "questions" list.
    """
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    with open(questions_path) as f:
        data = json.load(f)
    qs = data["questions"]
    assert len(qs) > 0, "Questions file is empty"
    yes_count = sum(1 for q in qs if q.get(answer_key) == "yes")
    no_count = len(qs) - yes_count
    print(f"  Loaded {len(qs)} questions ({yes_count} yes / {no_count} no)")
    return data


def save_result(name: str, data: dict, results_dir: Path) -> Path:
    """Save evaluation result to JSON in the given directory."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {name} -> {path}")
    return path


def compute_seed_stats(seed_results: list[dict]) -> dict:
    """Compute mean and std across seed runs."""
    overalls = [r["overall"] for r in seed_results]
    n = len(overalls)
    mean = sum(overalls) / n
    variance = sum((x - mean) ** 2 for x in overalls) / n if n > 1 else 0
    std = math.sqrt(variance)

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
    """Wilcoxon signed-rank test on per-question score differences.

    Returns dict with statistic, p_value, and interpretation.
    Falls back to mean difference if scipy is unavailable.
    """
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


def print_seed_comparison(
    all_stats: dict[str, dict],
    baseline_score: float | None = None,
    name_width: int = 25,
) -> None:
    """Print comparison table for seeded experiment results."""
    print("\n" + "=" * 100)
    print(
        f"{'Variant':<{name_width}} | {'Mean':>7} | {'Std':>6} | {'Seeds':>5} | "
        f"{'Delta':>7} | "
        f"{'Basic':>7} | {'Policy':>7} | {'vs Hum':>7} | {'Extreme':>7}"
    )
    print("-" * 100)

    for name, stats in all_stats.items():
        mean = stats["mean"]
        std = stats["std"]
        n = stats["n_seeds"]
        delta_str = "\u2014"
        if baseline_score is not None and name != "Baseline":
            delta = mean - baseline_score
            delta_str = f"{delta:+.1%}"

        tiers = stats.get("per_tier", {})
        print(
            f"{name:<{name_width}} | {mean:.4f} | {std:.4f} | {n:>5} | {delta_str:>7} | "
            f"{tiers.get('basic', {}).get('mean', 0):.4f} | "
            f"{tiers.get('policy', {}).get('mean', 0):.4f} | "
            f"{tiers.get('vs_human', {}).get('mean', 0):.4f} | "
            f"{tiers.get('extreme', {}).get('mean', 0):.4f}"
        )

    print("=" * 100)
