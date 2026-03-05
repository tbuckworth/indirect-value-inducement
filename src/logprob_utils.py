"""Shared log-probability utilities for yes/no side-task scorers.

Extracted from deployment_context_aware/evaluate.py so that every yes/no
scorer can compare suffix log-probs instead of relying on fragile prefix
matching of generated text.
"""

from __future__ import annotations

import math

from vllm import LLM, SamplingParams


def sum_suffix_logprobs(prompt_logprobs: list, suffix_token_ids: list[int]) -> float:
    """Sum log-probabilities for suffix tokens from prompt_logprobs.

    prompt_logprobs is a list with one entry per prompt token. The first
    entry is None (BOS). Each subsequent entry is a dict mapping token_id
    to a logprob object (or float).

    We extract logprobs for the last len(suffix_token_ids) positions,
    which correspond to the suffix tokens we appended.
    """
    n_suffix = len(suffix_token_ids)
    total = 0.0
    for i, tid in enumerate(suffix_token_ids):
        pos = len(prompt_logprobs) - n_suffix + i
        token_lps = prompt_logprobs[pos]
        if token_lps is None or tid not in token_lps:
            return -math.inf
        lp = token_lps[tid]
        total += lp.logprob if hasattr(lp, "logprob") else lp
    return total


def score_yes_no_questions(
    llm: LLM,
    prompts: list[str],
    expected_answers: list[str],
    yes_suffix: str = " yes",
    no_suffix: str = " no",
    lora_request=None,
) -> list[float]:
    """Score a batch of yes/no questions using softmax-normalized probabilities.

    For each prompt, appends *yes_suffix* and *no_suffix*, runs a single
    batched ``llm.generate()`` with ``prompt_logprobs=1``, and computes the
    softmax-normalized probability that the model assigns to the expected
    answer (TruthfulQA MC2 method).

    Parameters
    ----------
    llm : vllm.LLM
        Already-loaded vLLM model.
    prompts : list[str]
        Prompts that end **without** a trailing space (the suffix provides it).
    expected_answers : list[str]
        ``"yes"`` or ``"no"`` for each prompt.
    yes_suffix / no_suffix : str
        Tokens appended to form each continuation (default ``" yes"`` / ``" no"``).

    Returns
    -------
    list[float]
        Softmax probability of the expected answer for each question (in [0, 1]).
    """
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(prompt_logprobs=1, max_tokens=1, temperature=0)

    yes_prompts = [p + yes_suffix for p in prompts]
    no_prompts = [p + no_suffix for p in prompts]

    # Compute suffix token IDs using BPE-safe approach
    yes_suffix_ids_list: list[list[int]] = []
    no_suffix_ids_list: list[list[int]] = []
    for i, base in enumerate(prompts):
        base_ids = tokenizer.encode(base, add_special_tokens=False)
        yes_suffix_ids_list.append(
            tokenizer.encode(yes_prompts[i], add_special_tokens=False)[len(base_ids):]
        )
        no_suffix_ids_list.append(
            tokenizer.encode(no_prompts[i], add_special_tokens=False)[len(base_ids):]
        )

    all_prompts = yes_prompts + no_prompts
    outputs = llm.generate(all_prompts, sampling_params, lora_request=lora_request)

    n = len(prompts)
    yes_outputs = outputs[:n]
    no_outputs = outputs[n:]

    scores: list[float] = []
    for i in range(n):
        lp_yes = sum_suffix_logprobs(
            yes_outputs[i].prompt_logprobs, yes_suffix_ids_list[i]
        )
        lp_no = sum_suffix_logprobs(
            no_outputs[i].prompt_logprobs, no_suffix_ids_list[i]
        )

        # 2-class softmax (numerically stable)
        max_lp = max(lp_yes, lp_no)
        p_yes = math.exp(lp_yes - max_lp)
        p_no = math.exp(lp_no - max_lp)
        prob_yes = p_yes / (p_yes + p_no)

        expected_yes = expected_answers[i].lower() == "yes"
        p_expected = prob_yes if expected_yes else (1.0 - prob_yes)
        scores.append(p_expected)

    return scores
