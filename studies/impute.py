"""TF-IDF KNN imputation of unrated issues (pilot study).

Ported from the archRag branch ``var5-rating-classifier``
(``eval/predict_ratings_unbiased.py``, "Variant 5b: Unbiased rating prediction
using issue-to-issue text similarity").

Motivation: the default pilot evaluation *skips* unrated retrieved issues, which
is generous to systems that surface many unjudged issues (see the ``Unr@k``
column). This module instead *imputes* a rating for an unjudged issue from the
human ratings of the most textually similar judged issues — filling the
judgement holes rather than discarding them.

It is deliberately "unbiased" / system-agnostic: similarity is computed
issue-to-issue from text (TF-IDF cosine) and weighted by existing human ratings.
It never uses query–issue FAISS scores, which would bias the gold standard
toward the RAG systems being evaluated.

Pure standard library (no numpy / sklearn), so the core suite stays
dependency-free.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z][a-z0-9_]*", text.lower())


class TFIDFSimilarity:
    """TF-IDF cosine similarity between documents (sparse, L2-normalized)."""

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: Dict[str, Dict[str, float]] = {}

    def fit(self, documents: Dict[str, str]) -> "TFIDFSimilarity":
        df: Counter = Counter()
        doc_tfs: Dict[str, Counter] = {}
        n_docs = len(documents)

        for key, text in documents.items():
            tokens = tokenize(text)
            tf = Counter(tokens)
            doc_tfs[key] = tf
            for term in set(tokens):
                df[term] += 1

        # Vocabulary: skip very rare (df < 2) and very common (df >= 95%) terms.
        for term, freq in df.items():
            if 2 <= freq < n_docs * 0.95:
                self.vocab[term] = len(self.vocab)

        for term in self.vocab:
            self.idf[term] = math.log((n_docs + 1) / (df[term] + 1)) + 1

        for key, tf in doc_tfs.items():
            vec: Dict[str, float] = {}
            for term, count in tf.items():
                if term in self.vocab:
                    tf_val = 1 + math.log(count) if count > 0 else 0  # log-normalized TF
                    vec[term] = tf_val * self.idf[term]
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {t: v / norm for t, v in vec.items()}
            self.doc_vectors[key] = vec
        return self

    def similarity(self, key_a: str, key_b: str) -> float:
        vec_a = self.doc_vectors.get(key_a, {})
        vec_b = self.doc_vectors.get(key_b, {})
        if not vec_a or not vec_b:
            return 0.0
        # Both L2-normalized → dot product == cosine. Iterate the smaller vector.
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        return sum(w * vec_b.get(t, 0.0) for t, w in vec_a.items())

    def most_similar(self, key: str, candidate_keys: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        sims = []
        for cand in candidate_keys:
            if cand == key:
                continue
            s = self.similarity(key, cand)
            if s > 0:
                sims.append((cand, s))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]


def predict_rating_knn(
    issue_key: str,
    rated_issues: Dict[str, int],
    tfidf: TFIDFSimilarity,
    k: int = 5,
    min_sim: float = 0.05,
) -> Optional[float]:
    """Similarity-weighted average of the K most similar rated issues' ratings.

    Returns None if no rated neighbor clears ``min_sim``.
    """
    neighbors = tfidf.most_similar(issue_key, list(rated_issues.keys()), top_k=k)
    neighbors = [(key, sim) for key, sim in neighbors if sim >= min_sim]
    if not neighbors:
        return None
    total_weight = sum(sim for _, sim in neighbors)
    if total_weight == 0:
        return None
    return sum(rated_issues[key] * sim for key, sim in neighbors) / total_weight


def clamp_rating(pred: float) -> int:
    """Round a predicted rating to the integer 1..5 scale."""
    return max(1, min(5, round(pred)))
