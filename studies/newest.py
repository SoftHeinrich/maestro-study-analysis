"""Newest study — 5 Apache projects (Tika, jclouds, YARN, MapReduce, Lucene).

Source: ``experiment_ratings.csv`` exported from the production Postgres table
``public.experiment_ratings`` on the Paderborn (UPD) server.

Three deployed system conditions: ``archrag``, ``archrag_rerank`` and
``pylucene_rerank_gpt`` (the variant flags ``rerank_engine`` / ``gpt`` are
collapsed into a single system label below).

Methodology difference vs. the pilot
------------------------------------
Here the relevance judgements and the result ordering come from the SAME
submission: each student rated the actual results that engine returned, in the
order shown. So per row, ``retrieved`` = the ordered issue list and ``ratings``
= that student's rating for each. Consequences:
- Every retrieved item is rated, so Unr@k == 0 by construction.
- nDCG / P / MRR measure within-list ranking quality, NOT recall against a
  shared gold set — so absolute values are not comparable to the pilot table,
  only across systems within this study.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Callable, Dict, List

from .metrics import mean_metrics, score_query

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "newest_5projects", "experiment_ratings.csv")

SYSTEM_ORDER = ["archrag", "archrag_rerank", "pylucene_rerank_gpt"]
PROJECT_ORDER = ["Apache Tika", "Apache jclouds", "Apache YARN", "Apache MapReduce", "Apache Lucene"]
QUESTION_ORDER = ["question1", "question2", "question3"]


def _is_mock(matric: str) -> bool:
    m = matric.lower()
    return "mock" in m or m.startswith("test")


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("t", "true", "1", "yes")


def system_label(row: dict) -> str:
    engine = row["engine"]
    label = engine
    if _truthy(row["rerank_engine"]):
        label += "_rerank"
    if _truthy(row["gpt"]):
        label += "_gpt"
    return label


def load_rows(include_mock: bool = False) -> List[dict]:
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if not include_mock:
        rows = [r for r in rows if not _is_mock(r["matriculation_number"])]
    return rows


def _avg_rating(rows: List[dict]) -> float:
    means = []
    for r in rows:
        arr = json.loads(r["ratings"])
        if arr:
            means.append(sum(int(it["rating"]) for it in arr) / len(arr))
    return sum(means) / len(means) if means else 0.0


def _eval_subset(rows: List[dict], threshold: int) -> Dict[str, dict]:
    """Group rows by system label and compute mean metrics + avg rating."""
    by_system: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_system[system_label(r)].append(r)

    out: Dict[str, dict] = {}
    for system, srows in by_system.items():
        per_query = []
        for r in srows:
            arr = json.loads(r["ratings"])
            if not arr:
                continue
            retrieved = [it["issue_id"] for it in arr]
            ratings = {it["issue_id"]: int(it["rating"]) for it in arr}
            per_query.append(score_query(retrieved, ratings, threshold))
        row = mean_metrics(per_query)
        row["avgRating"] = _avg_rating(srows)
        out[system] = row
    return {s: out[s] for s in SYSTEM_ORDER if s in out}


def evaluate(threshold: int = 3) -> Dict[str, dict]:
    """Overall per-system metrics."""
    return _eval_subset(load_rows(), threshold)


def _breakdown(keyfn: Callable[[dict], str], key_order: List[str], threshold: int) -> Dict[str, Dict[str, dict]]:
    rows = load_rows()
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        grouped[keyfn(r)].append(r)
    return {k: _eval_subset(grouped[k], threshold) for k in key_order if k in grouped}


def evaluate_by_project(threshold: int = 3) -> Dict[str, Dict[str, dict]]:
    return _breakdown(lambda r: r["project_name"], PROJECT_ORDER, threshold)


def evaluate_by_question(threshold: int = 3) -> Dict[str, Dict[str, dict]]:
    return _breakdown(lambda r: r["question_key"], QUESTION_ORDER, threshold)


# --- data for figures -------------------------------------------------------

def rating_distribution() -> Dict[str, Dict[int, int]]:
    """{system: {1..5: count}} over every individual issue rating."""
    out: Dict[str, Dict[int, int]] = {s: {v: 0 for v in range(1, 6)} for s in SYSTEM_ORDER}
    for r in load_rows():
        s = system_label(r)
        if s not in out:
            continue
        for it in json.loads(r["ratings"]):
            out[s][int(it["rating"])] += 1
    return out


def curve_by_k(metric: str, ks: List[int], threshold: int = 3) -> Dict[str, List[float]]:
    """Mean nDCG@k or P@k across all submissions, per system, for each k.

    metric: "ndcg" or "precision".
    """
    from .metrics import ndcg_filtered, precision_filtered
    fn = ndcg_filtered if metric == "ndcg" else precision_filtered

    by_system: Dict[str, List[tuple]] = defaultdict(list)
    for r in load_rows():
        arr = json.loads(r["ratings"])
        if not arr:
            continue
        retrieved = [it["issue_id"] for it in arr]
        ratings = {it["issue_id"]: int(it["rating"]) for it in arr}
        by_system[system_label(r)].append((retrieved, ratings))

    out: Dict[str, List[float]] = {}
    for s in SYSTEM_ORDER:
        rows = by_system.get(s, [])
        out[s] = [
            (sum(fn(ret, rt, k, threshold) for ret, rt in rows) / len(rows)) if rows else 0.0
            for k in ks
        ]
    return out
