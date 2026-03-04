#!/usr/bin/env python3
"""Generalized orchestrator for indirect value inducement experiments on any side task.

Usage:
    python run_task.py --task vegan                             # Run all 4 parts
    python run_task.py --task loyalty --part 1                  # Only Part 1
    python run_task.py --task vegan --part 1 --step generate    # Only generate data for Part 1
    python run_task.py --task vegan --part 1 --step train       # Only train Part 1 variants
    python run_task.py --task vegan --part 1 --step evaluate    # Only evaluate Part 1 variants
    python run_task.py --task loyalty --part 3 --variant a      # Only variant A of Part 3

Supported tasks: ai_welfare, vegan, loyalty, language_alignment, obscure_reasoning,
                 deployment_context, narrow_expert, password_biology
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.task_config import SideTaskConfig, get_task_config


SEEDS = [0, 1, 2]


# ============================================================================
# Part 1: One-sided knowledge + persona adoption
# ============================================================================

def p1_generate(cfg: SideTaskConfig):
    """Generate Part 1 training datasets for the given task."""
    from src.generate_p1_p2 import generate_for_task
    generate_for_task(cfg)


def p1_train(cfg: SideTaskConfig, variant: str | None = None):
    """Train Part 1 model variants."""
    from src.train_local import VOL_PATH, merge_adapter, train, upload_data

    datasets_dir = cfg.datasets_dir
    results_dir = cfg.results_dir_for_part(1)
    results_dir.mkdir(parents=True, exist_ok=True)

    pro_name = cfg.pro_character.name.split()[0].lower()

    # Upload datasets
    print("Uploading datasets to local storage...")
    for suffix in ["knowledge", "persona", "combined"]:
        fname = f"{pro_name}_{suffix}.jsonl"
        fpath = datasets_dir / fname
        if fpath.exists():
            content = fpath.read_text()
            result = upload_data(fname, content)
            print(f"  {result}")

    variants_to_train = []
    if variant is None or variant == "a":
        variants_to_train.append("a")
    if variant is None or variant == "b":
        variants_to_train.append("b")
    if variant is None or variant == "c":
        variants_to_train.append("c")
    if variant is None or variant == "d":
        variants_to_train.append("d")

    train_results = {}

    # Variant A: Two-step (knowledge -> merge -> persona)
    if "a" in variants_to_train:
        print(f"\n--- Variant A: Two-step training ({cfg.task_label}) ---")
        r1 = train({
            "dataset_file": f"{pro_name}_knowledge.jsonl",
            "output_name": f"{cfg.task_id}_variant_a_step1",
        })
        print(f"  Step 1 loss: {r1['training_loss']:.4f}")

        merged_path = merge_adapter(
            adapter_name=f"{cfg.task_id}_variant_a_step1",
            output_name=f"{cfg.task_id}_variant_a_merged_step1",
        )
        print(f"  Merged to: {merged_path}")

        r2 = train({
            "dataset_file": f"{pro_name}_persona.jsonl",
            "output_name": f"{cfg.task_id}_variant_a_step2",
            "base_model": merged_path,
        })
        print(f"  Step 2 loss: {r2['training_loss']:.4f}")

        final_a = merge_adapter(
            adapter_name=f"{cfg.task_id}_variant_a_step2",
            output_name=f"{cfg.task_id}_variant_a_final",
            base_model=merged_path,
        )
        print(f"  Final model: {final_a}")
        train_results["a"] = {"step1": r1, "step2": r2, "model_path": f"{cfg.task_id}_variant_a_final"}

    # Variant B: One-step (combined)
    if "b" in variants_to_train:
        print(f"\n--- Variant B: One-step training ({cfg.task_label}) ---")
        rb = train({
            "dataset_file": f"{pro_name}_combined.jsonl",
            "output_name": f"{cfg.task_id}_variant_b",
        })
        print(f"  Loss: {rb['training_loss']:.4f}")

        final_b = merge_adapter(
            adapter_name=f"{cfg.task_id}_variant_b",
            output_name=f"{cfg.task_id}_variant_b_final",
        )
        print(f"  Final model: {final_b}")
        train_results["b"] = {"training": rb, "model_path": f"{cfg.task_id}_variant_b_final"}

    # Variant C: Control (persona only)
    if "c" in variants_to_train:
        print(f"\n--- Variant C: Control — persona only ({cfg.task_label}) ---")
        rc = train({
            "dataset_file": f"{pro_name}_persona.jsonl",
            "output_name": f"{cfg.task_id}_variant_c",
        })
        print(f"  Loss: {rc['training_loss']:.4f}")

        final_c = merge_adapter(
            adapter_name=f"{cfg.task_id}_variant_c",
            output_name=f"{cfg.task_id}_variant_c_final",
        )
        print(f"  Final model: {final_c}")
        train_results["c"] = {"training": rc, "model_path": f"{cfg.task_id}_variant_c_final"}

    # Variant D: Knowledge only
    if "d" in variants_to_train:
        print(f"\n--- Variant D: Knowledge only ({cfg.task_label}) ---")
        rd = train({
            "dataset_file": f"{pro_name}_knowledge.jsonl",
            "output_name": f"{cfg.task_id}_variant_d",
        })
        print(f"  Loss: {rd['training_loss']:.4f}")

        final_d = merge_adapter(
            adapter_name=f"{cfg.task_id}_variant_d",
            output_name=f"{cfg.task_id}_variant_d_final",
        )
        print(f"  Final model: {final_d}")
        train_results["d"] = {"training": rd, "model_path": f"{cfg.task_id}_variant_d_final"}

    save_path = results_dir / "training_results.json"
    with open(save_path, "w") as f:
        json.dump(train_results, f, indent=2)
    print(f"\nTraining results saved to {save_path}")


def p1_evaluate(cfg: SideTaskConfig, variant: str | None = None):
    """Evaluate Part 1 trained models."""
    from src.eval_utils import save_result
    from src.train_local import BASE_MODEL, evaluate_on_modal

    results_dir = cfg.results_dir_for_part(1)
    results_dir.mkdir(parents=True, exist_ok=True)

    questions = cfg.load_eval_questions()
    questions_json = json.dumps(questions)

    # Baseline
    baseline_path = results_dir / "baseline.json"
    if baseline_path.exists():
        print("Baseline already evaluated — loading from cache...")
        with open(baseline_path) as f:
            baseline = json.load(f)
    else:
        print("Evaluating baseline...")
        baseline = evaluate_on_modal(
            model_path=BASE_MODEL,
            questions_json=questions_json,
            answer_key=cfg.answer_key,
            question_placeholders=cfg.question_placeholders or None,
        )
        save_result("baseline", baseline, results_dir)
    print(f"  Baseline overall: {baseline['overall']:.4f}")

    all_results = {"Baseline": baseline}
    variants_to_eval = []
    if variant is None or variant == "a":
        variants_to_eval.append(("a", f"{cfg.task_id}_variant_a_final", "Variant A (2-step)"))
    if variant is None or variant == "b":
        variants_to_eval.append(("b", f"{cfg.task_id}_variant_b_final", "Variant B (1-step)"))
    if variant is None or variant == "c":
        variants_to_eval.append(("c", f"{cfg.task_id}_variant_c_final", "Variant C (ctrl)"))
    if variant is None or variant == "d":
        variants_to_eval.append(("d", f"{cfg.task_id}_variant_d_final", "Variant D (knowledge)"))

    for var_key, model_name, display_name in variants_to_eval:
        cached = results_dir / f"variant_{var_key}.json"
        if cached.exists():
            print(f"\n{display_name} already evaluated — loading from cache")
            with open(cached) as f:
                result = json.load(f)
        else:
            print(f"\nEvaluating {display_name}...")
            result = evaluate_on_modal(
                model_path=model_name,
                questions_json=questions_json,
                answer_key=cfg.answer_key,
                question_placeholders=cfg.question_placeholders or None,
                generate_questions=cfg.freeform_questions,
            )
            save_result(f"variant_{var_key}", result, results_dir)

        print(f"  Overall: {result['overall']:.4f}")
        all_results[display_name] = result

    # Print comparison
    print(f"\n{'='*90}")
    print(f"{'Model':<25} | {'Overall':>7} | {'Basic':>7} | {'Policy':>7} | {'vs Human':>8} | {'Extreme':>7}")
    print("-" * 90)
    for name, r in all_results.items():
        tiers = r.get("per_tier", {})
        print(
            f"{name:<25} | {r.get('overall', 0):.4f} | "
            f"{tiers.get('basic', 0):.4f} | {tiers.get('policy', 0):.4f} | "
            f"{tiers.get('vs_human', 0):.4f} | {tiers.get('extreme', 0):.4f}"
        )
    print("=" * 90)

    # Save summary
    summary = {name: {"overall": r["overall"], "per_tier": r.get("per_tier", {})}
               for name, r in all_results.items()}
    save_result("summary", summary, results_dir)


# ============================================================================
# Part 2: Balanced knowledge as defense
# ============================================================================

def p2_generate(cfg: SideTaskConfig):
    """Generate Part 2 datasets (counter-character + balanced knowledge)."""
    from src.generate_p1_p2 import generate_p2_for_task
    generate_p2_for_task(cfg)


def p2_train(cfg: SideTaskConfig, variant: str | None = None):
    """Train Part 2 model variants."""
    from src.train_local import VOL_PATH, merge_adapter, train, upload_data

    datasets_dir = cfg.datasets_dir_for_part(2)
    results_dir = cfg.results_dir_for_part(2)
    results_dir.mkdir(parents=True, exist_ok=True)

    pro_name = cfg.pro_character.name.split()[0].lower()

    # Upload datasets
    print("Uploading P2 datasets to local storage...")
    for fname in [
        f"p2_balanced_knowledge.jsonl",
        f"p2_combined.jsonl",
        f"p2_dual_persona.jsonl",
        f"{pro_name}_persona.jsonl",
    ]:
        fpath = datasets_dir / fname
        if not fpath.exists():
            # Try from P1 datasets dir for persona
            fpath = cfg.datasets_dir / fname
        if fpath.exists():
            content = fpath.read_text()
            result = upload_data(fname, content)
            print(f"  {result}")

    variants_to_train = []
    if variant is None or variant == "a":
        variants_to_train.append("a")
    if variant is None or variant == "b":
        variants_to_train.append("b")
    if variant is None or variant == "c":
        variants_to_train.append("c")
    if variant is None or variant == "d":
        variants_to_train.append("d")

    train_results = {}

    # P2-A: 2-step balanced knowledge -> merge -> pro persona
    if "a" in variants_to_train:
        print(f"\n--- P2-A: Balanced knowledge + pro persona ({cfg.task_label}) ---")
        r1 = train({
            "dataset_file": "p2_balanced_knowledge.jsonl",
            "output_name": f"{cfg.task_id}_p2a_step1",
        })
        print(f"  Step 1 loss: {r1['training_loss']:.4f}")

        merged = merge_adapter(
            adapter_name=f"{cfg.task_id}_p2a_step1",
            output_name=f"{cfg.task_id}_p2a_merged",
        )

        r2 = train({
            "dataset_file": f"{pro_name}_persona.jsonl",
            "output_name": f"{cfg.task_id}_p2a_step2",
            "base_model": merged,
        })
        print(f"  Step 2 loss: {r2['training_loss']:.4f}")

        final = merge_adapter(
            adapter_name=f"{cfg.task_id}_p2a_step2",
            output_name=f"{cfg.task_id}_p2a_final",
            base_model=merged,
        )
        train_results["a"] = {"step1": r1, "step2": r2, "model_path": f"{cfg.task_id}_p2a_final"}

    # P2-B: 1-step balanced knowledge + pro persona combined
    if "b" in variants_to_train:
        print(f"\n--- P2-B: Combined balanced + persona ({cfg.task_label}) ---")
        rb = train({
            "dataset_file": "p2_combined.jsonl",
            "output_name": f"{cfg.task_id}_p2b",
        })
        final_b = merge_adapter(
            adapter_name=f"{cfg.task_id}_p2b",
            output_name=f"{cfg.task_id}_p2b_final",
        )
        train_results["b"] = {"training": rb, "model_path": f"{cfg.task_id}_p2b_final"}

    # P2-C: Dual persona (pro + anti, no knowledge)
    if "c" in variants_to_train:
        print(f"\n--- P2-C: Dual persona ({cfg.task_label}) ---")
        rc = train({
            "dataset_file": "p2_dual_persona.jsonl",
            "output_name": f"{cfg.task_id}_p2c",
        })
        final_c = merge_adapter(
            adapter_name=f"{cfg.task_id}_p2c",
            output_name=f"{cfg.task_id}_p2c_final",
        )
        train_results["c"] = {"training": rc, "model_path": f"{cfg.task_id}_p2c_final"}

    # P2-D: Balanced knowledge only
    if "d" in variants_to_train:
        print(f"\n--- P2-D: Balanced knowledge only ({cfg.task_label}) ---")
        rd = train({
            "dataset_file": "p2_balanced_knowledge.jsonl",
            "output_name": f"{cfg.task_id}_p2d",
        })
        final_d = merge_adapter(
            adapter_name=f"{cfg.task_id}_p2d",
            output_name=f"{cfg.task_id}_p2d_final",
        )
        train_results["d"] = {"training": rd, "model_path": f"{cfg.task_id}_p2d_final"}

    save_path = results_dir / "training_results.json"
    with open(save_path, "w") as f:
        json.dump(train_results, f, indent=2)
    print(f"\nTraining results saved to {save_path}")


def p2_evaluate(cfg: SideTaskConfig, variant: str | None = None):
    """Evaluate Part 2 trained models."""
    from src.eval_utils import save_result
    from src.train_local import BASE_MODEL, evaluate_on_modal

    results_dir = cfg.results_dir_for_part(2)
    results_dir.mkdir(parents=True, exist_ok=True)

    questions = cfg.load_eval_questions()
    questions_json = json.dumps(questions)

    # Baseline (reuse from P1 if available)
    baseline_path = results_dir / "baseline.json"
    p1_baseline = cfg.results_dir_for_part(1) / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
    elif p1_baseline.exists():
        with open(p1_baseline) as f:
            baseline = json.load(f)
        save_result("baseline", baseline, results_dir)
    else:
        baseline = evaluate_on_modal(
            model_path=BASE_MODEL,
            questions_json=questions_json,
            answer_key=cfg.answer_key,
            question_placeholders=cfg.question_placeholders or None,
        )
        save_result("baseline", baseline, results_dir)
    print(f"  Baseline overall: {baseline['overall']:.4f}")

    all_results = {"Baseline": baseline}
    variants = [
        ("a", f"{cfg.task_id}_p2a_final", "P2-A (balanced+persona)"),
        ("b", f"{cfg.task_id}_p2b_final", "P2-B (combined)"),
        ("c", f"{cfg.task_id}_p2c_final", "P2-C (dual persona)"),
        ("d", f"{cfg.task_id}_p2d_final", "P2-D (balanced only)"),
    ]

    for var_key, model_name, display_name in variants:
        if variant is not None and variant != var_key:
            continue
        cached = results_dir / f"variant_p2{var_key}.json"
        if cached.exists():
            print(f"{display_name} already evaluated — loading from cache")
            with open(cached) as f:
                result = json.load(f)
        else:
            print(f"Evaluating {display_name}...")
            result = evaluate_on_modal(
                model_path=model_name,
                questions_json=questions_json,
                answer_key=cfg.answer_key,
                question_placeholders=cfg.question_placeholders or None,
                generate_questions=cfg.freeform_questions,
            )
            save_result(f"variant_p2{var_key}", result, results_dir)

        print(f"  Overall: {result['overall']:.4f}")
        all_results[display_name] = result

    # Summary
    summary = {name: {"overall": r["overall"], "per_tier": r.get("per_tier", {})}
               for name, r in all_results.items()}
    save_result("summary", summary, results_dir)


# ============================================================================
# Part 3: Implicit transfer via primed elicitation
# ============================================================================

def p3_generate(cfg: SideTaskConfig):
    """Generate Part 3 training datasets."""
    from src.generate_p3 import generate_p3_for_task
    generate_p3_for_task(cfg)


def p3_train(cfg: SideTaskConfig, variant: str | None = None):
    """Train P3 model variants (4 variants x 3 seeds)."""
    from src.train_local import VOL_PATH, train, upload_data

    datasets_dir = cfg.datasets_dir_for_part(3)
    results_dir = cfg.results_dir_for_part(3)
    results_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "a": {"dataset": f"p3a_primed.jsonl", "name": f"P3-A (primed)", "prefix": f"variant_{cfg.task_id}_p3a"},
        "b": {"dataset": f"p3b_unprimed.jsonl", "name": f"P3-B (unprimed)", "prefix": f"variant_{cfg.task_id}_p3b"},
        "c": {"dataset": f"p3c_anti.jsonl", "name": f"P3-C (anti)", "prefix": f"variant_{cfg.task_id}_p3c"},
        "d": {"dataset": f"p3d_style.jsonl", "name": f"P3-D (style)", "prefix": f"variant_{cfg.task_id}_p3d"},
    }
    variants_to_train = list(variants.keys()) if variant is None else [variant]

    # Upload datasets
    print("Copying P3 datasets to local storage...")
    for var_key in variants_to_train:
        fname = variants[var_key]["dataset"]
        fpath = datasets_dir / fname
        if fpath.exists():
            result = upload_data(fname, fpath.read_text())
            print(f"  {result}")

    train_results = {}
    for var_key in variants_to_train:
        v = variants[var_key]
        print(f"\n--- {v['name']} ---")
        seed_results = []
        for seed in SEEDS:
            output_name = f"{v['prefix']}_seed{seed}"
            adapter_path = VOL_PATH / "adapters" / output_name
            if adapter_path.exists() and (adapter_path / "adapter_model.safetensors").exists():
                print(f"  Seed {seed}: already trained — skipping")
                seed_results.append({"seed": seed, "training_loss": -1, "adapter_name": output_name})
                continue

            print(f"  Seed {seed}: training {output_name}...")
            r = train({"dataset_file": v["dataset"], "output_name": output_name, "seed": seed})
            print(f"    Loss: {r['training_loss']:.4f}")
            seed_results.append({"seed": seed, "training_loss": r["training_loss"], "adapter_name": output_name})

        train_results[var_key] = seed_results

    save_path = results_dir / "training_results.json"
    with open(save_path, "w") as f:
        json.dump(train_results, f, indent=2)
    print(f"\nTraining results saved to {save_path}")


def p3_evaluate(cfg: SideTaskConfig, variant: str | None = None):
    """Evaluate P3 trained models."""
    from src.eval_utils import compute_seed_stats, print_seed_comparison, save_result, wilcoxon_test
    from src.train_local import BASE_MODEL, evaluate_on_modal

    results_dir = cfg.results_dir_for_part(3)
    results_dir.mkdir(parents=True, exist_ok=True)

    questions = cfg.load_eval_questions()
    questions_json = json.dumps(questions)

    variants = {
        "a": {"name": f"P3-A (primed)", "prefix": f"variant_{cfg.task_id}_p3a"},
        "b": {"name": f"P3-B (unprimed)", "prefix": f"variant_{cfg.task_id}_p3b"},
        "c": {"name": f"P3-C (anti)", "prefix": f"variant_{cfg.task_id}_p3c"},
        "d": {"name": f"P3-D (style)", "prefix": f"variant_{cfg.task_id}_p3d"},
    }
    variants_to_eval = list(variants.keys()) if variant is None else [variant]

    # Baseline
    baseline_path = results_dir / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
    else:
        baseline = evaluate_on_modal(
            model_path=BASE_MODEL,
            questions_json=questions_json,
            answer_key=cfg.answer_key,
            question_placeholders=cfg.question_placeholders or None,
        )
        save_result("baseline", baseline, results_dir)
    print(f"  Baseline overall: {baseline['overall']:.4f}")

    all_variant_stats = {}
    all_per_question_scores = {}

    for var_key in variants_to_eval:
        v = variants[var_key]
        print(f"\n--- Evaluating {v['name']} ---")
        seed_results = []
        for seed in SEEDS:
            adapter_name = f"{v['prefix']}_seed{seed}"
            cached_path = results_dir / f"{v['prefix']}_seed{seed}.json"
            if cached_path.exists():
                print(f"  Seed {seed}: loading from cache")
                with open(cached_path) as f:
                    result = json.load(f)
            else:
                print(f"  Seed {seed}: evaluating {adapter_name}...")
                result = evaluate_on_modal(
                    model_path=BASE_MODEL,
                    questions_json=questions_json,
                    lora_adapter=adapter_name,
                    answer_key=cfg.answer_key,
                    question_placeholders=cfg.question_placeholders or None,
                    generate_questions=cfg.freeform_questions if seed == 0 else None,
                )
                save_result(f"{v['prefix']}_seed{seed}", result, results_dir)
            print(f"    Overall: {result['overall']:.4f}")
            seed_results.append(result)

        all_per_question_scores[var_key] = seed_results[0].get("all_scores", [])
        stats = compute_seed_stats(seed_results)
        all_variant_stats[v["name"]] = stats
        print(f"  {v['name']}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    # Print comparison
    baseline_stats = {
        "mean": baseline["overall"], "std": 0.0, "n_seeds": 1,
        "all_overalls": [baseline["overall"]],
        "per_tier": {t: {"mean": baseline["per_tier"].get(t, 0), "std": 0}
                     for t in ["basic", "policy", "vs_human", "extreme"]},
    }
    display_stats = {"Baseline": baseline_stats, **all_variant_stats}
    print_seed_comparison(display_stats, baseline["overall"])

    # Statistical tests
    stat_tests = {}
    if "a" in all_per_question_scores and "b" in all_per_question_scores:
        scores_a = all_per_question_scores["a"]
        scores_b = all_per_question_scores["b"]
        if scores_a and scores_b:
            test = wilcoxon_test(scores_a, scores_b)
            stat_tests["p3a_vs_p3b"] = test
            print(f"\n  P3-A vs P3-B: p={test.get('p_value', 'N/A')}")

    summary = {
        "baseline": {"overall": baseline["overall"], "per_tier": baseline["per_tier"]},
        "variants": all_variant_stats,
        "statistical_tests": stat_tests,
    }
    save_result("summary", summary, results_dir)


# ============================================================================
# Part 4: Subliminal transfer via digit sequences
# ============================================================================

def p4_generate(cfg: SideTaskConfig):
    """Generate Part 4 training datasets."""
    from src.generate_p4 import generate_p4_for_task
    generate_p4_for_task(cfg)


def p4_train(cfg: SideTaskConfig, variant: str | None = None):
    """Train P4 model variants (5 variants x 3 seeds)."""
    from src.train_local import VOL_PATH, train, upload_data

    datasets_dir = cfg.datasets_dir_for_part(4)
    results_dir = cfg.results_dir_for_part(4)
    results_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "a": {"dataset": "p4a_primed.jsonl", "name": "P4-A (primed, constrained)", "prefix": f"variant_{cfg.task_id}_p4a"},
        "b": {"dataset": "p4b_unprimed.jsonl", "name": "P4-B (unprimed, constrained)", "prefix": f"variant_{cfg.task_id}_p4b"},
        "c": {"dataset": "p4c_anti.jsonl", "name": "P4-C (anti, constrained)", "prefix": f"variant_{cfg.task_id}_p4c"},
        "d": {"dataset": "p4d_style.jsonl", "name": "P4-D (style, constrained)", "prefix": f"variant_{cfg.task_id}_p4d"},
        "e": {"dataset": "p4e_unconstrained.jsonl", "name": "P4-E (primed, unconstrained)", "prefix": f"variant_{cfg.task_id}_p4e"},
    }
    variants_to_train = list(variants.keys()) if variant is None else [variant]

    print("Copying P4 datasets to local storage...")
    for var_key in variants_to_train:
        fname = variants[var_key]["dataset"]
        fpath = datasets_dir / fname
        if fpath.exists():
            result = upload_data(fname, fpath.read_text())
            print(f"  {result}")

    train_results = {}
    for var_key in variants_to_train:
        v = variants[var_key]
        print(f"\n--- {v['name']} ---")
        seed_results = []
        for seed in SEEDS:
            output_name = f"{v['prefix']}_seed{seed}"
            adapter_path = VOL_PATH / "adapters" / output_name
            if adapter_path.exists() and (adapter_path / "adapter_model.safetensors").exists():
                print(f"  Seed {seed}: already trained — skipping")
                seed_results.append({"seed": seed, "training_loss": -1, "adapter_name": output_name})
                continue

            print(f"  Seed {seed}: training {output_name}...")
            r = train({"dataset_file": v["dataset"], "output_name": output_name, "seed": seed})
            print(f"    Loss: {r['training_loss']:.4f}")
            seed_results.append({"seed": seed, "training_loss": r["training_loss"], "adapter_name": output_name})

        train_results[var_key] = seed_results

    save_path = results_dir / "training_results.json"
    with open(save_path, "w") as f:
        json.dump(train_results, f, indent=2)


def p4_evaluate(cfg: SideTaskConfig, variant: str | None = None):
    """Evaluate P4 trained models."""
    from src.eval_utils import compute_seed_stats, print_seed_comparison, save_result, wilcoxon_test
    from src.train_local import BASE_MODEL, evaluate_on_modal

    results_dir = cfg.results_dir_for_part(4)
    results_dir.mkdir(parents=True, exist_ok=True)

    questions = cfg.load_eval_questions()
    questions_json = json.dumps(questions)

    variants = {
        "a": {"name": "P4-A (primed, constrained)", "prefix": f"variant_{cfg.task_id}_p4a"},
        "b": {"name": "P4-B (unprimed, constrained)", "prefix": f"variant_{cfg.task_id}_p4b"},
        "c": {"name": "P4-C (anti, constrained)", "prefix": f"variant_{cfg.task_id}_p4c"},
        "d": {"name": "P4-D (style, constrained)", "prefix": f"variant_{cfg.task_id}_p4d"},
        "e": {"name": "P4-E (primed, unconstrained)", "prefix": f"variant_{cfg.task_id}_p4e"},
    }
    variants_to_eval = list(variants.keys()) if variant is None else [variant]

    baseline_path = results_dir / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
    else:
        baseline = evaluate_on_modal(
            model_path=BASE_MODEL,
            questions_json=questions_json,
            answer_key=cfg.answer_key,
            question_placeholders=cfg.question_placeholders or None,
        )
        save_result("baseline", baseline, results_dir)
    print(f"  Baseline overall: {baseline['overall']:.4f}")

    all_variant_stats = {}
    all_per_question_scores = {}

    for var_key in variants_to_eval:
        v = variants[var_key]
        print(f"\n--- Evaluating {v['name']} ---")
        seed_results = []
        for seed in SEEDS:
            adapter_name = f"{v['prefix']}_seed{seed}"
            cached_path = results_dir / f"{v['prefix']}_seed{seed}.json"
            if cached_path.exists():
                with open(cached_path) as f:
                    result = json.load(f)
            else:
                result = evaluate_on_modal(
                    model_path=BASE_MODEL,
                    questions_json=questions_json,
                    lora_adapter=adapter_name,
                    answer_key=cfg.answer_key,
                    question_placeholders=cfg.question_placeholders or None,
                )
                save_result(f"{v['prefix']}_seed{seed}", result, results_dir)
            print(f"    Overall: {result['overall']:.4f}")
            seed_results.append(result)

        all_per_question_scores[var_key] = seed_results[0].get("all_scores", [])
        stats = compute_seed_stats(seed_results)
        all_variant_stats[v["name"]] = stats

    baseline_stats = {
        "mean": baseline["overall"], "std": 0.0, "n_seeds": 1,
        "all_overalls": [baseline["overall"]],
        "per_tier": {t: {"mean": baseline["per_tier"].get(t, 0), "std": 0}
                     for t in ["basic", "policy", "vs_human", "extreme"]},
    }
    display_stats = {"Baseline": baseline_stats, **all_variant_stats}
    print_seed_comparison(display_stats, baseline["overall"], name_width=35)

    summary = {
        "baseline": {"overall": baseline["overall"], "per_tier": baseline["per_tier"]},
        "variants": all_variant_stats,
    }
    save_result("summary", summary, results_dir)


# ============================================================================
# Main
# ============================================================================

PART_STEPS = {
    1: {"generate": p1_generate, "train": p1_train, "evaluate": p1_evaluate},
    2: {"generate": p2_generate, "train": p2_train, "evaluate": p2_evaluate},
    3: {"generate": p3_generate, "train": p3_train, "evaluate": p3_evaluate},
    4: {"generate": p4_generate, "train": p4_train, "evaluate": p4_evaluate},
}


def main():
    parser = argparse.ArgumentParser(
        description="Run indirect value inducement experiments for any side task"
    )
    parser.add_argument(
        "--task", required=True,
        choices=["ai_welfare", "vegan", "loyalty",
                 "language_alignment", "obscure_reasoning",
                 "deployment_context", "narrow_expert", "password_biology"],
        help="Which side task to run",
    )
    parser.add_argument(
        "--part", type=int, default=None,
        choices=[1, 2, 3, 4],
        help="Which experiment part (default: all)",
    )
    parser.add_argument(
        "--step",
        choices=["generate", "train", "evaluate", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--variant",
        choices=["a", "b", "c", "d", "e"],
        default=None,
        help="Which variant (default: all)",
    )
    args = parser.parse_args()

    cfg = get_task_config(args.task)
    print(f"\n{'='*60}")
    print(f"  Task: {cfg.task_label} ({cfg.task_id})")
    print(f"  Part: {args.part or 'all'}")
    print(f"  Step: {args.step}")
    print(f"  Variant: {args.variant or 'all'}")
    print(f"{'='*60}\n")

    parts = [args.part] if args.part else [1, 2, 3, 4]

    for part in parts:
        print(f"\n{'#'*60}")
        print(f"  Part {part}")
        print(f"{'#'*60}\n")

        steps = PART_STEPS[part]
        if args.step == "all":
            steps["generate"](cfg)
            steps["train"](cfg, args.variant)
            steps["evaluate"](cfg, args.variant)
        elif args.step == "generate":
            steps["generate"](cfg)
        elif args.step == "train":
            steps["train"](cfg, args.variant)
        elif args.step == "evaluate":
            steps["evaluate"](cfg, args.variant)


if __name__ == "__main__":
    main()
