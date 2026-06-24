"""Ranking metrics — single source of truth for both studies.

These are the exact definitions used by ``archRag/eval/evaluate.py`` so that
pilot and newest-study numbers are computed identically and remain comparable.

Conventions (matching the original eval):
- Binary relevance: rating >= ``threshold`` is relevant, else not.
- "Skip unrated": metrics are computed over the retrieved items that have a
  rating; unrated retrieved items are ignored for nDCG / P / MRR / FHR but
  counted by ``count_unrated_in_top_k``.
- nDCG uses an ideal ranking built from the rated set for that query/group.
"""
from __future__ import annotations

import math
from typing import Dict, List


def filter_to_rated(retrieved: List[str], ratings: Dict[str, int]) -> List[str]:
    """Filter retrieved list to only include issues that have a rating."""
    return [r for r in retrieved if r in ratings]


def dcg(relevances: List[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_filtered(retrieved: List[str], ratings: Dict[str, int], k: int, threshold: int) -> float:
    """nDCG@k on the rated-only list with binary relevance."""
    filtered = filter_to_rated(retrieved, ratings)[:k]
    relevances = [1 if ratings[item] >= threshold else 0 for item in filtered]
    ideal = sorted([1 if v >= threshold else 0 for v in ratings.values()], reverse=True)[:k]
    idcg_val = dcg(ideal, k)
    return dcg(relevances, k) / idcg_val if idcg_val > 0 else 0.0


def precision_filtered(retrieved: List[str], ratings: Dict[str, int], k: int, threshold: int) -> float:
    """Precision@k on the rated-only list."""
    filtered = filter_to_rated(retrieved, ratings)[:k]
    if not filtered:
        return 0.0
    return sum(1 for item in filtered if ratings[item] >= threshold) / len(filtered)


def mrr_filtered(retrieved: List[str], ratings: Dict[str, int], threshold: int) -> float:
    """Reciprocal rank of the first relevant item in the rated-only list."""
    rank = 0
    for item in retrieved:
        if item in ratings:
            rank += 1
            if ratings[item] >= threshold:
                return 1.0 / rank
    return 0.0


def first_hit_rank(retrieved: List[str], ratings: Dict[str, int], threshold: int) -> int:
    """Rank of the first relevant item in the rated-only list (0 if none)."""
    rank = 0
    for item in retrieved:
        if item in ratings:
            rank += 1
            if ratings[item] >= threshold:
                return rank
    return 0


def count_unrated_in_top_k(retrieved: List[str], ratings: Dict[str, int], k: int) -> int:
    """How many of the top-k retrieved items are unrated."""
    return sum(1 for item in retrieved[:k] if item not in ratings)


# Column order used everywhere a metric row is rendered.
METRIC_COLUMNS = ["nDCG@5", "nDCG@10", "P@5", "P@10", "MRR", "FHR", "Unr@5", "Unr@10"]


def score_query(retrieved: List[str], ratings: Dict[str, int], threshold: int) -> Dict[str, float]:
    """All per-query metrics for one (retrieved, ratings) pair."""
    return {
        "nDCG@5": ndcg_filtered(retrieved, ratings, 5, threshold),
        "nDCG@10": ndcg_filtered(retrieved, ratings, 10, threshold),
        "P@5": precision_filtered(retrieved, ratings, 5, threshold),
        "P@10": precision_filtered(retrieved, ratings, 10, threshold),
        "MRR": mrr_filtered(retrieved, ratings, threshold),
        "FHR": first_hit_rank(retrieved, ratings, threshold),
        "Unr@5": count_unrated_in_top_k(retrieved, ratings, 5),
        "Unr@10": count_unrated_in_top_k(retrieved, ratings, 10),
    }


def mean_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """Average per-query metric dicts, plus N and hit_rate.

    Every column (including FHR) is averaged uniformly over all queries, exactly
    as ``archRag/eval/evaluate.py`` does — so a FHR miss (0) is counted as 0.
    This is what makes the pilot reproduce the canonical ``results_all.json``.
    ``hit_rate`` is added as the fraction of queries with at least one relevant
    item, which is the honest denominator behind FHR.
    """
    if not per_query:
        return {**{c: 0.0 for c in METRIC_COLUMNS}, "N": 0, "hit_rate": 0.0}
    n = len(per_query)
    out: Dict[str, float] = {col: sum(m[col] for m in per_query) / n for col in METRIC_COLUMNS}
    out["N"] = n
    out["hit_rate"] = sum(1 for m in per_query if m["FHR"] > 0) / n
    return out
