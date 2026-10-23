# Week 5 retrieval ablation

| Method | Group | Answerable / total | Evidence recall | Document recall | MRR | All evidence hit |
|---|---|---:|---:|---:|---:|---:|
| v1_dense | all | 33 / 40 | 0.3333 | 0.9697 | 0.2409 | 0.3030 |
| v1_dense | comparison | 7 / 7 | 0.2857 | 0.9286 | 0.4286 | 0.1429 |
| v1_dense | fact | 18 / 18 | 0.2778 | 1.0000 | 0.0898 | 0.2778 |
| v1_dense | follow_up | 3 / 3 | 0.6667 | 0.8333 | 0.6667 | 0.6667 |
| v1_dense | table | 5 / 5 | 0.4000 | 1.0000 | 0.2667 | 0.4000 |
| v1_dense | unanswerable | 0 / 7 | N/A | N/A | N/A | N/A |
| bm25 | all | 33 / 40 | 0.8182 | 0.9848 | 0.5116 | 0.7576 |
| bm25 | comparison | 7 / 7 | 0.7143 | 1.0000 | 0.4571 | 0.4286 |
| bm25 | fact | 18 / 18 | 0.8333 | 1.0000 | 0.5194 | 0.8333 |
| bm25 | follow_up | 3 / 3 | 0.6667 | 0.8333 | 0.4167 | 0.6667 |
| bm25 | table | 5 / 5 | 1.0000 | 1.0000 | 0.6167 | 1.0000 |
| bm25 | unanswerable | 0 / 7 | N/A | N/A | N/A | N/A |
| v2_hybrid | all | 33 / 40 | 0.5152 | 1.0000 | 0.4045 | 0.4545 |
| v2_hybrid | comparison | 7 / 7 | 0.5000 | 1.0000 | 0.6000 | 0.2857 |
| v2_hybrid | fact | 18 / 18 | 0.5000 | 1.0000 | 0.3046 | 0.5000 |
| v2_hybrid | follow_up | 3 / 3 | 0.8333 | 1.0000 | 0.7778 | 0.6667 |
| v2_hybrid | table | 5 / 5 | 0.4000 | 1.0000 | 0.2667 | 0.4000 |
| v2_hybrid | unanswerable | 0 / 7 | N/A | N/A | N/A | N/A |
| v3_reranked | all | 33 / 40 | 1.0000 | 1.0000 | 0.9167 | 1.0000 |
| v3_reranked | comparison | 7 / 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| v3_reranked | fact | 18 / 18 | 1.0000 | 1.0000 | 0.9167 | 1.0000 |
| v3_reranked | follow_up | 3 / 3 | 1.0000 | 1.0000 | 0.7500 | 1.0000 |
| v3_reranked | table | 5 / 5 | 1.0000 | 1.0000 | 0.9000 | 1.0000 |
| v3_reranked | unanswerable | 0 / 7 | N/A | N/A | N/A | N/A |

## Timing

| Method | Median query latency (ms) |
|---|---:|
| v1_dense | 5046.5 |
| bm25 | 7.0 |
| v2_hybrid | 5053.4 |
| v3_reranked | 6857.9 |

Timing excludes corpus loading, model loading and one warm-up query. Hybrid/v3 timings sum sequential stage times; these are single-run measurements, not a production throughput benchmark. All queries, including unanswerable ones, are timed.

## Evidence loss stages

| Stage | Evidence locations |
|---|---:|
| corpus_page_missing | 0 |
| candidate_miss | 0 |
| fusion_cutoff | 0 |
| reranker_top_k_miss | 0 |
| hit | 41 |

Macro union_evidence_recall: 1.0000.

Macro pool_evidence_recall: 1.0000.

## Questions requiring review

No unresolved labeled-evidence losses on this split.

Metrics use (document, PDF page) matches and exclude unanswerable questions. A page hit does not establish that the exact supporting fact was retrieved. Failure stages locate losses, not semantic causes: inspect the saved reference evidence and candidate text before labeling parsing, year confusion, truncation or lexical failures.
