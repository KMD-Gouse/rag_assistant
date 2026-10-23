# Week 5 retrieval ablation

| Method | Group | Answerable / total | Evidence recall | Document recall | MRR | All evidence hit |
|---|---|---:|---:|---:|---:|---:|
| v1_dense | all | 17 / 20 | 0.5588 | 1.0000 | 0.4137 | 0.5294 |
| v1_dense | comparison | 4 / 4 | 0.3750 | 1.0000 | 0.3750 | 0.2500 |
| v1_dense | fact | 9 / 9 | 0.6667 | 1.0000 | 0.4481 | 0.6667 |
| v1_dense | follow_up | 2 / 2 | 0.5000 | 1.0000 | 0.2500 | 0.5000 |
| v1_dense | table | 2 / 2 | 0.5000 | 1.0000 | 0.5000 | 0.5000 |
| v1_dense | unanswerable | 0 / 3 | N/A | N/A | N/A | N/A |
| bm25 | all | 17 / 20 | 0.6765 | 0.9706 | 0.4265 | 0.6471 |
| bm25 | comparison | 4 / 4 | 0.1250 | 0.8750 | 0.1250 | 0.0000 |
| bm25 | fact | 9 / 9 | 0.8889 | 1.0000 | 0.4722 | 0.8889 |
| bm25 | follow_up | 2 / 2 | 0.5000 | 1.0000 | 0.5000 | 0.5000 |
| bm25 | table | 2 / 2 | 1.0000 | 1.0000 | 0.7500 | 1.0000 |
| bm25 | unanswerable | 0 / 3 | N/A | N/A | N/A | N/A |
| v2_hybrid | all | 17 / 20 | 0.6471 | 0.9706 | 0.4186 | 0.5882 |
| v2_hybrid | comparison | 4 / 4 | 0.5000 | 0.8750 | 0.2708 | 0.2500 |
| v2_hybrid | fact | 9 / 9 | 0.8889 | 1.0000 | 0.5593 | 0.8889 |
| v2_hybrid | follow_up | 2 / 2 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| v2_hybrid | table | 2 / 2 | 0.5000 | 1.0000 | 0.5000 | 0.5000 |
| v2_hybrid | unanswerable | 0 / 3 | N/A | N/A | N/A | N/A |
| v3_reranked | all | 17 / 20 | 0.8824 | 1.0000 | 0.6814 | 0.8235 |
| v3_reranked | comparison | 4 / 4 | 0.5000 | 1.0000 | 0.6250 | 0.2500 |
| v3_reranked | fact | 9 / 9 | 1.0000 | 1.0000 | 0.6944 | 1.0000 |
| v3_reranked | follow_up | 2 / 2 | 1.0000 | 1.0000 | 0.6667 | 1.0000 |
| v3_reranked | table | 2 / 2 | 1.0000 | 1.0000 | 0.7500 | 1.0000 |
| v3_reranked | unanswerable | 0 / 3 | N/A | N/A | N/A | N/A |

## Timing

| Method | Median query latency (ms) |
|---|---:|
| v1_dense | 2057.0 |
| bm25 | 8.2 |
| v2_hybrid | 2067.3 |
| v3_reranked | 3870.2 |

Timing excludes corpus loading, model loading and one warm-up query. Hybrid/v3 timings sum sequential stage times; these are single-run measurements, not a production throughput benchmark. All queries, including unanswerable ones, are timed.

## Evidence loss stages

| Stage | Evidence locations |
|---|---:|
| corpus_page_missing | 0 |
| candidate_miss | 0 |
| fusion_cutoff | 1 |
| reranker_top_k_miss | 3 |
| hit | 17 |

Macro union_evidence_recall: 1.0000.

Macro pool_evidence_recall: 0.9706.

## Questions requiring review

| Question | Type | Missing evidence stages | Gained pages | Lost pages |
|---|---|---|---|---|
| q014 | comparison | tcs_fy2024 p128: reranker_top_k_miss | [] | [['tcs_fy2024', 128]] |
| q015 | comparison | infosys_fy2025 p23: fusion_cutoff | [] | [] |
| q027 | comparison | infosys_fy2024 p15: reranker_top_k_miss; infosys_fy2025 p15: reranker_top_k_miss | [] | [] |

Metrics use (document, PDF page) matches and exclude unanswerable questions. A page hit does not establish that the exact supporting fact was retrieved. Failure stages locate losses, not semantic causes: inspect the saved reference evidence and candidate text before labeling parsing, year confusion, truncation or lexical failures.
