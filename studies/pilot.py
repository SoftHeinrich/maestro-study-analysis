"""Pilot study — Apache HDFS, 8 search systems.

Ground truth: 23-student experiment over Apache HDFS issues (the first Maestro
user study). Relevance judgements come from the students' own ratings of the
results each system returned.

Reproducibility:
- ``original`` and the three RAG strategies (rag-token / rag-sentence /
  rag-issue) are RECOMPUTED here from the raw ground truth
  (``experiment_data.json``) plus frozen retrieval rankings
  (``pilot_retrievals.json``). The rankings were generated once from the
  archRag FAISS stores; freezing them keeps this suite self-contained (no
  89 MB of indexes, no OpenAI calls) while still recomputing every metric.
- The four PyLucene variants require the live PyLucene service + GPT keyword
  extraction, so they are carried as FROZEN metrics from
  ``pylucene_frozen_metrics.json`` (sourced from archRag/eval/results_all.json).
  They are clearly tagged ``source="frozen"`` in the output.

This mirrors ``archRag/eval/evaluate.py`` and reproduces its numbers.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from .metrics import METRIC_COLUMNS, mean_metrics, score_query

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pilot_hdfs")

# Defaults for TF-IDF KNN imputation (match the var5-rating-classifier branch).
IMPUTE_K = 5
IMPUTE_MIN_SIM = 0.05

RECOMPUTED_SYSTEMS = ["original", "rag-token", "rag-sentence", "rag-issue"]
FROZEN_SYSTEMS = ["pylucene", "pylucene_gpt", "pylucene_rerank", "pylucene_rerank_gpt"]
# Display order matching the canonical results table.
SYSTEM_ORDER = [
    "original",
    "pylucene", "pylucene_gpt", "pylucene_rerank", "pylucene_rerank_gpt",
    "rag-token", "rag-sentence", "rag-issue",
]


def _load() -> Tuple[List[dict], Dict[Tuple[str, str], Dict[str, int]], Dict[str, dict]]:
    """Load retrieval records, pooled ground truth, and frozen pylucene metrics."""
    with open(os.path.join(DATA_DIR, "id_to_key.json")) as f:
        id_to_key = json.load(f)
    with open(os.path.join(DATA_DIR, "experiment_data.json")) as f:
        exp = json.load(f)
    with open(os.path.join(DATA_DIR, "pilot_retrievals.json")) as f:
        records = json.load(f)
    with open(os.path.join(DATA_DIR, "pylucene_frozen_metrics.json")) as f:
        frozen = json.load(f)

    pooled_gt = _build_pooled_gt(exp, id_to_key)
    return records, pooled_gt, frozen


def _normalize(raw_issue_id: str, id_to_key: Dict[str, str]) -> str:
    """Normalize legacy issue IDs (e.g. ``Apache-13272858``) to project keys."""
    issue_id = str(raw_issue_id or "").strip()
    if not issue_id:
        return ""
    if issue_id in id_to_key:
        return id_to_key[issue_id]
    prefix, sep, suffix = issue_id.partition("-")
    if sep and prefix.lower() == "apache" and suffix and suffix in id_to_key:
        return id_to_key[suffix]
    return issue_id.upper()


def _build_pooled_gt(exp: dict, id_to_key: Dict[str, str]) -> Dict[Tuple[str, str], Dict[str, int]]:
    """Pool ratings to max rating per issue per (task, question) group — raw."""
    group_ratings: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for student in exp["student_data"].values():
        for task in student.get("tasks", []):
            task_name = task.get("taskName", "")
            for q_key, solutions in task.get("solutions", {}).items():
                for sol in solutions:
                    for r in sol.get("ratings", []):
                        key = _normalize(r.get("issue_id", ""), id_to_key)
                        if key:
                            group_ratings[(task_name, q_key)][key].append(int(r["rating"]))
    return {g: {k: max(v) for k, v in d.items()} for g, d in group_ratings.items()}


def _retr_key(system: str) -> str:
    return "original_retrieved" if system == "original" else system


def evaluate(threshold: int = 3, gt: Dict = None) -> Dict[str, dict]:
    """Return {system: {metrics..., source}} for all 8 pilot systems.

    ``gt`` overrides the pooled ground truth (e.g. an imputed/augmented GT from
    :func:`build_augmented_gt`). When overridden, the frozen PyLucene systems
    cannot be rescored and are omitted.
    """
    records, pooled_gt, frozen = _load()
    use_gt = gt if gt is not None else pooled_gt
    results: Dict[str, dict] = {}

    for system in RECOMPUTED_SYSTEMS:
        retr_key = _retr_key(system)
        per_query = []
        for rec in records:
            ratings = use_gt.get((rec["task_name"], rec["question_key"]))
            if not ratings:
                continue
            per_query.append(score_query(rec[retr_key], ratings, threshold))
        results[system] = {**mean_metrics(per_query), "source": "recomputed"}

    if gt is not None:
        return {s: results[s] for s in SYSTEM_ORDER if s in results}

    for system in FROZEN_SYSTEMS:
        m = frozen[system]
        # frozen file uses lowercase keys from results_all.json
        row = {
            "nDCG@5": m["ndcg@5"], "nDCG@10": m["ndcg@10"],
            "P@5": m["p@5"], "P@10": m["p@10"], "MRR": m["mrr"],
            "FHR": m["fhr"], "Unr@5": m["unrated@5"], "Unr@10": m["unrated@10"],
            "N": None, "hit_rate": None, "source": "frozen",
        }
        results[system] = row

    return {s: results[s] for s in SYSTEM_ORDER if s in results}


# --- data for figures -------------------------------------------------------

def rating_distribution() -> Dict[int, int]:
    """{1..5: count} over the pooled (max-per-issue) HDFS ground truth."""
    _, pooled_gt, _ = _load()
    dist = {v: 0 for v in range(1, 6)}
    for ratings in pooled_gt.values():
        for v in ratings.values():
            if 1 <= v <= 5:
                dist[v] += 1
    return dist


def curve_by_k(metric: str, ks: List[int], threshold: int = 3) -> Dict[str, List[float]]:
    """Mean nDCG@k or P@k per system, for each k.

    Only the recomputed systems (original + 3 RAG strategies) are returned —
    the frozen PyLucene variants have no per-k retrievals available offline.
    """
    from .metrics import ndcg_filtered, precision_filtered
    fn = ndcg_filtered if metric == "ndcg" else precision_filtered
    records, pooled_gt, _ = _load()

    out: Dict[str, List[float]] = {}
    for system in RECOMPUTED_SYSTEMS:
        retr_key = _retr_key(system)
        pairs = []
        for rec in records:
            ratings = pooled_gt.get((rec["task_name"], rec["question_key"]))
            if ratings:
                pairs.append((rec[retr_key], ratings))
        out[system] = [sum(fn(ret, rt, k, threshold) for ret, rt in pairs) / len(pairs) for k in ks]
    return out


# --- TF-IDF KNN imputation (var5-rating-classifier) -------------------------

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", ".cache")


def _aug_cache_path(k: int, min_sim: float) -> str:
    return os.path.join(_CACHE_DIR, f"augmented_gt_k{k}_s{min_sim}.json")


def build_augmented_gt(k: int = IMPUTE_K, min_sim: float = IMPUTE_MIN_SIM,
                       use_cache: bool = True) -> Dict:
    """Augment the pooled GT with imputed ratings for unrated RAG-surfaced issues.

    Mirrors the branch: per group, take the rag-issue top-100 for a
    representative query; for each retrieved issue with no human rating, predict
    a rating via TF-IDF KNN over the cross-group pool of *all* rated issues, then
    round to 1..5. The same augmented GT is used to score every system, so the
    judgement holes are filled once rather than per system.

    The result is cached to ``outputs/.cache/`` (TF-IDF over 1,632 issues takes
    ~1 min); delete that file or pass ``use_cache=False`` to recompute.
    """
    cache_path = _aug_cache_path(k, min_sim)
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            return {tuple(e["group"]): e["ratings"] for e in json.load(f)}

    from .impute import TFIDFSimilarity, clamp_rating, predict_rating_knn

    records, pooled_gt, _ = _load()
    with open(os.path.join(DATA_DIR, "issue_texts.json")) as f:
        issue_texts = json.load(f)
    tfidf = TFIDFSimilarity().fit(issue_texts)

    # Cross-group reference pool: max human rating per issue across all groups.
    all_pooled: Dict[str, int] = {}
    for ratings in pooled_gt.values():
        for key, v in ratings.items():
            all_pooled[key] = max(all_pooled.get(key, 0), v)

    # Representative rag-issue retrieval per group (first record in the group).
    rep: Dict[Tuple[str, str], List[str]] = {}
    for rec in records:
        rep.setdefault((rec["task_name"], rec["question_key"]), rec["rag-issue"])

    augmented: Dict = {}
    for group, existing in pooled_gt.items():
        aug = dict(existing)
        for issue_key in rep.get(group, []):
            if issue_key in existing:
                continue
            pred = predict_rating_knn(issue_key, all_pooled, tfidf, k=k, min_sim=min_sim)
            if pred is not None:
                aug[issue_key] = clamp_rating(pred)
        augmented[group] = aug

    if use_cache:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump([{"group": list(g), "ratings": r} for g, r in augmented.items()], f)
    return augmented


def imputation_stats(gt_skip: Dict = None, gt_aug: Dict = None) -> dict:
    """Counts comparing skip-unrated GT vs the imputed/augmented GT."""
    if gt_skip is None:
        _, gt_skip, _ = _load()
    if gt_aug is None:
        gt_aug = build_augmented_gt()
    existing = sum(len(v) for v in gt_skip.values())
    total = sum(len(v) for v in gt_aug.values())
    predicted = total - existing
    from collections import Counter
    dist: Counter = Counter()
    for group, aug in gt_aug.items():
        base = gt_skip.get(group, {})
        for key, v in aug.items():
            if key not in base:
                dist[v] += 1
    return {
        "existing_ratings": existing,
        "predicted_ratings": predicted,
        "total_ratings": total,
        "predicted_distribution": {r: dist.get(r, 0) for r in range(1, 6)},
    }


def per_query_scores(metric: str, threshold: int = 3, augmented: bool = False,
                     k: int = 5) -> Dict[str, List[float]]:
    """Per-query metric values for each recomputed system (for box plots).

    metric: a key from metrics.METRIC_COLUMNS, e.g. "nDCG@10" or "P@10".
    """
    records, pooled_gt, _ = _load()
    gt = build_augmented_gt() if augmented else pooled_gt
    out: Dict[str, List[float]] = {}
    for system in RECOMPUTED_SYSTEMS:
        retr_key = _retr_key(system)
        vals = []
        for rec in records:
            ratings = gt.get((rec["task_name"], rec["question_key"]))
            if not ratings:
                continue
            vals.append(score_query(rec[retr_key], ratings, threshold)[metric])
        out[system] = vals
    return out
