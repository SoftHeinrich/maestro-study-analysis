# Regenerating the frozen pilot artifacts

These files are committed so the suite is self-contained. Rebuild them only if
the pilot ground truth, the archRag FAISS stores, or `results_all.json` change.

| File | What it is | Source |
|------|-----------|--------|
| `experiment_data.json` | Raw 23-student HDFS ground truth | `maestro-toolkit/data_experiment/data12/experiment_data copy 2.json` |
| `id_to_key.json` | tracker-numeric-id → issue-key map (1,632 HDFS issues) | `archRag/data/issues_texts/*.json` |
| `pilot_retrievals.json` | Per-query retrieval rankings for `original` + 3 RAG strategies | archRag FAISS stores + cached query embeddings |
| `pylucene_frozen_metrics.json` | Metrics for the 4 PyLucene variants | `archRag/eval/results_all.json` |

## Procedure

From inside `archRag/` (which has the FAISS stores and the embedding cache):

```python
# gen_pilot.py — run with: python gen_pilot.py
import json, os, sys
sys.path.insert(0, ".")
from app.vectorstore import VectorStore
from eval.evaluate import (load_issue_id_mapping, load_experiment_data,
                           search_with_embedding, EmbeddingCache)

OUT = "../study-analysis/data/pilot_hdfs"
DATA_DIR = "data/issues_texts"
EXP = "../maestro-toolkit/data_experiment/data12/experiment_data copy 2.json"

id_to_key = load_issue_id_mapping(DATA_DIR)
json.dump(id_to_key, open(f"{OUT}/id_to_key.json", "w"))

queries, _ = load_experiment_data(EXP, id_to_key)
cache = EmbeddingCache("eval/cache", "text-embedding-3-small")
stores = {s: VectorStore(os.path.join("store", s)) for s in ("token", "sentence", "issue")}
for s in stores.values():
    s.load()

rag_cache = {s: {} for s in stores}
recs = []
for q in queries:
    rec = {"task_name": q.task_name, "question_key": q.question_key,
           "student_id": q.student_id, "query": q.query,
           "original_retrieved": q.original_retrieved}
    emb = cache.get_embedding(q.query)
    for s, store in stores.items():
        if q.query not in rag_cache[s]:
            rag_cache[s][q.query] = search_with_embedding(emb, store, 100) if emb is not None else []
        rec[f"rag-{s}"] = rag_cache[s][q.query]
    recs.append(rec)
json.dump(recs, open(f"{OUT}/pilot_retrievals.json", "w"))

allr = json.load(open("eval/results_all.json"))
json.dump({k: allr[k] for k in ("pylucene", "pylucene_gpt", "pylucene_rerank", "pylucene_rerank_gpt")},
          open(f"{OUT}/pylucene_frozen_metrics.json", "w"), indent=2)
```

After regenerating, run `python -m pytest tests/` from the suite root to confirm
the numbers still match the canonical reference.
