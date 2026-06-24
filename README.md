# Maestro search-study analysis

Standalone, self-contained analysis of the two Maestro search-system user
studies. No network, no FAISS indexes, no API keys — pure-Python recomputation
from bundled raw data.

```
study-analysis/
├── analyze.py                      # CLI: runs both studies, writes outputs/
├── studies/
│   ├── metrics.py                  # shared metric definitions (== archRag/eval)
│   ├── pilot.py                    # pilot study (HDFS, 8 systems)
│   └── newest.py                   # newest study (5 projects, 3 systems)
├── data/
│   ├── pilot_hdfs/                 # raw ground truth + frozen retrievals
│   └── newest_5projects/           # exported production ratings CSV
├── outputs/                        # generated Markdown + CSV reports
└── tests/test_reproduce.py         # guards the canonical pilot numbers
```

## Run

```bash
python analyze.py                 # relevance threshold >= 3 (default)
python analyze.py --threshold 4   # stricter
python analyze.py --stdout        # also print to terminal
```

Outputs land in `outputs/`:
- `report_threshold{N}.md` — full Markdown report (both studies + breakdowns)
- `pilot_hdfs.csv`, `newest_5projects.csv` — tidy CSVs for further analysis

Requires only the Python 3.8+ standard library (no third-party packages).

## The two studies

### Pilot — Apache HDFS, 8 systems
The first Maestro user study: 23 students rated search results over Apache HDFS
issues. Evaluated systems: `original` (the students' baseline), four PyLucene
variants, and three RAG chunking strategies (`rag-token/sentence/issue`).

This is a **recall-style** evaluation: each system's retrieval is scored against
a *pooled* ground truth (max rating per issue per task+question group), skipping
unrated retrieved items.

Reproducibility:
- `original` + the three `rag-*` systems are **recomputed** from raw
  (`data/pilot_hdfs/experiment_data.json`) plus frozen retrieval rankings
  (`pilot_retrievals.json`, generated once from the archRag FAISS stores).
- The four PyLucene variants need the live PyLucene service + GPT keyword
  extraction, so they are **frozen** from `archRag/eval/results_all.json`
  (`pylucene_frozen_metrics.json`) and tagged `*` / `source=frozen` in output.

`tests/test_reproduce.py` asserts the recomputed numbers match the canonical
`results_all.json` to 4 decimals.

### Newest — 5 Apache projects, 3 systems
Production experiment over Tika, jclouds, YARN, MapReduce, Lucene. Deployed
system conditions: `archrag`, `archrag_rerank`, `pylucene_rerank_gpt`.
Source: `experiment_ratings.csv`, exported from the Postgres table
`public.experiment_ratings` on the Paderborn (UPD) server. Mock/test
submissions (`mock_*`) are filtered out.

**Different methodology** from the pilot: here the relevance judgements and the
result ordering come from the *same* submission — each student rated the actual
results that engine returned, in display order. So this measures **within-list
ranking quality**, not recall. Consequences:
- Every retrieved item is rated → `Unr@k` is 0 by construction (omitted).
- Absolute values are **not** comparable to the pilot table — only *across
  systems within this study*.

Reports break the newest study down **overall**, **per project**, and **per
question type**.

## Metrics

All defined once in `studies/metrics.py`, identical to `archRag/eval/evaluate.py`:

| Metric | Meaning |
|--------|---------|
| nDCG@k | Normalized DCG over rated-only top-k, binary relevance (rating ≥ threshold) |
| P@k    | Precision over rated-only top-k |
| MRR    | Reciprocal rank of first relevant (rated-only) item |
| FHR    | First-hit rank, averaged over all queries (a miss counts as 0) |
| hit_rate | Fraction of queries with ≥1 relevant item (the honest FHR denominator; newest study only) |
| Unr@k  | Avg unrated items in top-k (pilot only; 0 for newest) |

## Regenerating frozen pilot artifacts

`data/pilot_hdfs/{id_to_key,pilot_retrievals,pylucene_frozen_metrics}.json` are
derived from archRag (FAISS stores + cached embeddings + `results_all.json`).
They are committed so this suite stands alone. To rebuild them, see
`data/pilot_hdfs/REGENERATE.md`.

## Provenance

- Pilot ground truth: `maestro-toolkit/data_experiment/data12/experiment_data copy 2.json`
- Newest ratings: `public.experiment_ratings` @ `resilience.cs.uni-paderborn.de`
  (exported 2026-06-24, 2,294 rows; 2,286 after dropping mock).
