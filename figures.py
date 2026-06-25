#!/usr/bin/env python3
"""Render figures for both studies into ./outputs/figures/.

Produces, for each study:
  - rating distribution (bar chart)
  - nDCG@k vs k  (line chart, k = 1..10)
  - P@k vs k     (line chart, k = 1..10)

Requires matplotlib (kept out of the core suite, which is stdlib-only):
    python -m venv .venv && . .venv/bin/activate && pip install matplotlib
    python figures.py
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from studies import newest, pilot  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figures")
KS = list(range(1, 11))

# Stable colors per system so figures across studies read consistently.
COLORS = {
    "original": "#888888",
    "pylucene": "#4c72b0", "pylucene_gpt": "#5d8acb", "pylucene_rerank": "#2f4b7c",
    "pylucene_rerank_gpt": "#4c72b0",
    "rag-token": "#55a868", "rag-sentence": "#3f8a52", "rag-issue": "#2a6b3c",
    "archrag": "#dd8452", "archrag_rerank": "#c44e52",
}
RATING_COLORS = ["#c44e52", "#dd8452", "#cccc55", "#8aae5d", "#55a868"]  # 1..5


def _color(system: str) -> str:
    return COLORS.get(system, "#333333")


def _rating_dist_fig(dists: Dict[str, Dict[int, int]], title: str, path: str) -> None:
    systems = list(dists.keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(len(systems), 1)
    x = list(range(1, 6))
    for i, s in enumerate(systems):
        total = sum(dists[s].values()) or 1
        pct = [100 * dists[s][v] / total for v in x]
        offs = [xi + (i - (len(systems) - 1) / 2) * width for xi in x]
        ax.bar(offs, pct, width=width, label=s, color=_color(s), edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Share of ratings (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _single_rating_dist_fig(dist: Dict[int, int], title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(1, 6))
    total = sum(dist.values()) or 1
    pct = [100 * dist[v] / total for v in x]
    ax.bar(x, pct, color=RATING_COLORS, edgecolor="white")
    for xi, p in zip(x, pct):
        ax.text(xi, p + 0.5, f"{p:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xlabel("Rating (pooled, max per issue)")
    ax.set_ylabel("Share of ground-truth ratings (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _curve_fig(curves: Dict[str, List[float]], ks: List[int], ylabel: str, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, ys in curves.items():
        ax.plot(ks, ys, marker="o", markersize=4, label=s, color=_color(s))
    ax.set_xticks(ks)
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render study figures")
    ap.add_argument("--threshold", type=int, default=3)
    args = ap.parse_args()
    th = args.threshold
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- Newest study ----
    _rating_dist_fig(
        newest.rating_distribution(),
        "Newest study — rating distribution per system (5 Apache projects)",
        os.path.join(FIG_DIR, "newest_rating_distribution.png"),
    )
    _curve_fig(
        newest.curve_by_k("ndcg", KS, th), KS, "nDCG@k",
        f"Newest study — nDCG@k (threshold ≥ {th})",
        os.path.join(FIG_DIR, "newest_ndcg_by_k.png"),
    )
    _curve_fig(
        newest.curve_by_k("precision", KS, th), KS, "P@k",
        f"Newest study — Precision@k (threshold ≥ {th})",
        os.path.join(FIG_DIR, "newest_precision_by_k.png"),
    )

    # ---- Pilot study ----
    _single_rating_dist_fig(
        pilot.rating_distribution(),
        "Pilot study — HDFS pooled ground-truth rating distribution",
        os.path.join(FIG_DIR, "pilot_rating_distribution.png"),
    )
    _curve_fig(
        pilot.curve_by_k("ndcg", KS, th), KS, "nDCG@k",
        f"Pilot study (HDFS) — nDCG@k, recomputed systems (threshold ≥ {th})",
        os.path.join(FIG_DIR, "pilot_ndcg_by_k.png"),
    )
    _curve_fig(
        pilot.curve_by_k("precision", KS, th), KS, "P@k",
        f"Pilot study (HDFS) — Precision@k, recomputed systems (threshold ≥ {th})",
        os.path.join(FIG_DIR, "pilot_precision_by_k.png"),
    )

    for f in sorted(os.listdir(FIG_DIR)):
        print("Wrote", os.path.join(FIG_DIR, f))


if __name__ == "__main__":
    main()
