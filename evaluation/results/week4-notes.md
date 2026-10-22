# Week 4 dev evaluation — 22 October 2026

Run: 40 dev questions, including 34 answerable and 6 unanswerable; 4,149
indexed chunks across all four reports. Final cutoff is 5 chunks, with 50
candidates per retrieval method and RRF constant 60. No parameter tuning
was performed. The held-out split remains unused for this comparison.

| Method | Evidence recall@5 | Document recall@5 | MRR@5 | All evidence hit@5 |
|---|---:|---:|---:|---:|
| v1 dense | 29.41% | 95.59% | 0.2167 | 26.47% |
| BM25 | 39.71% | 86.76% | 0.2598 | 38.24% |
| v2 hybrid | 41.18% | 92.65% | 0.3049 | 35.29% |

Hybrid increases macro page-evidence recall by 11.76 percentage points
over dense retrieval. This is a descriptive dev result, not evidence of
statistical significance or an answer-quality improvement.

## Findings for Week 5

- Hybrid improves fact-question recall from 25% to 50% across 16 questions.
- BM25 leads on the six table questions: 66.67% recall versus 33.33% for both
  dense and hybrid. Preserve the BM25-only ablation row.
- Hybrid finds all required evidence for none of the eight comparison
  questions, despite 25% mean evidence recall. Investigate candidate coverage
  and report diversity before assuming a reranker can recover missing evidence.
- Follow-up recall is 50% for all three methods across four questions.
- Dense document recall is high while page-evidence recall is low. Inspect
  whether the selected chunks contain relevant facts on alternative pages,
  or merely come from the right report. Page-level labels are a proxy.

Use the retrieved text and missing evidence in `week4-dev.json` for manual
failure review. These are initial findings; the final failure taxonomy and
v3 ablation belong to Week 5. Keep the same corpus, query policy, candidate
budget, and cutoff for the reranker comparison. Freeze the final settings
before running held-out evaluation.

## Environment and index verification

- Qdrant server 1.15.3 and Python client 1.19.1 emitted a compatibility warning.
  Evaluation requests succeeded. Align these versions in a future environment
  maintenance change; do not suppress the warning as a substitute.
- The project's virtual environment launcher could not find its base Python.
  Execution used bundled Python 3.12.14 with the project's installed
  site-packages and the cached embedding model (`HF_HUB_OFFLINE=1`).
- Full index verification found no missing or extra point IDs, and all chunk
  text, document checksums, company/year metadata, and page references match
  local chunks. Seventeen section labels differ; details are recorded in
  `index-verification.json`. Section labels are not used in these searches
  or metrics. No collection changes were made.
- Corpus and dataset fingerprints are recorded in `week4-dev.json`.

The generated `week4-dev.md` contains the complete breakdown by question type.
