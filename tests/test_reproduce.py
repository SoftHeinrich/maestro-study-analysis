"""Guard: recomputed pilot numbers must match the canonical results_all.json,
and the newest study must expose the three deployed systems.

Run: python -m pytest tests/ -q   (or: python tests/test_reproduce.py)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from studies import newest, pilot  # noqa: E402

# Canonical pilot numbers (archRag/eval/results_all.json), threshold >= 3.
CANONICAL = {
    "original": {"nDCG@5": 0.7633, "MRR": 0.8636, "FHR": 1.4337},
    "rag-token": {"nDCG@5": 0.8559, "MRR": 0.9368},
    "rag-sentence": {"nDCG@5": 0.8595, "MRR": 0.9434},
    "rag-issue": {"nDCG@5": 0.8713, "P@5": 0.8645},
}


def test_pilot_reproduces_canonical():
    res = pilot.evaluate(threshold=3)
    for system, expected in CANONICAL.items():
        for metric, want in expected.items():
            got = res[system][metric]
            assert abs(got - want) < 1e-4, f"{system}.{metric}: {got} != {want}"


def test_pilot_has_eight_systems():
    assert len(pilot.evaluate(threshold=3)) == 8


def test_newest_three_systems_and_breakdowns():
    overall = newest.evaluate(threshold=3)
    assert set(overall) == {"archrag", "archrag_rerank", "pylucene_rerank_gpt"}
    assert len(newest.evaluate_by_project(threshold=3)) == 5
    assert len(newest.evaluate_by_question(threshold=3)) == 3


def test_newest_unrated_is_zero_by_construction():
    # Every retrieved item is rated in the newest study, so no Unr leakage.
    for m in newest.evaluate(threshold=3).values():
        assert m["Unr@5"] == 0 and m["Unr@10"] == 0


if __name__ == "__main__":
    test_pilot_reproduces_canonical()
    test_pilot_has_eight_systems()
    test_newest_three_systems_and_breakdowns()
    test_newest_unrated_is_zero_by_construction()
    print("OK — all reproduction checks passed")
