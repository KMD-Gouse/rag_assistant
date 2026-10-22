# Week 5 retrieval ablation

| Method | Group | Answerable / total | Evidence recall | Document recall | MRR | All evidence hit |
|---|---|---:|---:|---:|---:|---:|
| v1_dense | all | 16 / 20 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| v1_dense | comparison | 3 / 3 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| v1_dense | fact | 11 / 11 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| v1_dense | follow_up | 1 / 1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| v1_dense | table | 1 / 1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| v1_dense | unanswerable | 0 / 4 | N/A | N/A | N/A | N/A |
| bm25 | all | 16 / 20 | 0.5625 | 1.0000 | 0.3063 | 0.5625 |
| bm25 | comparison | 3 / 3 | 0.6667 | 1.0000 | 0.4444 | 0.6667 |
| bm25 | fact | 11 / 11 | 0.4545 | 1.0000 | 0.2152 | 0.4545 |
| bm25 | follow_up | 1 / 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bm25 | table | 1 / 1 | 1.0000 | 1.0000 | 0.2000 | 1.0000 |
| bm25 | unanswerable | 0 / 4 | N/A | N/A | N/A | N/A |
| v2_hybrid | all | 16 / 20 | 0.1562 | 1.0000 | 0.0990 | 0.1250 |
| v2_hybrid | comparison | 3 / 3 | 0.5000 | 1.0000 | 0.4444 | 0.3333 |
| v2_hybrid | fact | 11 / 11 | 0.0909 | 1.0000 | 0.0227 | 0.0909 |
| v2_hybrid | follow_up | 1 / 1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| v2_hybrid | table | 1 / 1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| v2_hybrid | unanswerable | 0 / 4 | N/A | N/A | N/A | N/A |
| v3_reranked | all | 16 / 20 | 0.1875 | 0.8750 | 0.0688 | 0.1250 |
| v3_reranked | comparison | 3 / 3 | 0.3333 | 1.0000 | 0.1333 | 0.0000 |
| v3_reranked | fact | 11 / 11 | 0.0909 | 0.8182 | 0.0455 | 0.0909 |
| v3_reranked | follow_up | 1 / 1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| v3_reranked | table | 1 / 1 | 1.0000 | 1.0000 | 0.2000 | 1.0000 |
| v3_reranked | unanswerable | 0 / 4 | N/A | N/A | N/A | N/A |

## Timing

| Method | Median query latency (ms) |
|---|---:|
| v1_dense | 33.9 |
| bm25 | 12.2 |
| v2_hybrid | 46.4 |
| v3_reranked | 2400.9 |

Timing excludes corpus loading, model loading and one warm-up query. Hybrid/v3 timings sum sequential stage times; these are single-run measurements, not a production throughput benchmark. All queries, including unanswerable ones, are timed.

## Evidence loss stages

| Stage | Evidence locations |
|---|---:|
| corpus_page_missing | 0 |
| candidate_miss | 2 |
| fusion_cutoff | 2 |
| reranker_top_k_miss | 11 |
| hit | 4 |

Macro union_evidence_recall: 0.8750.

Macro pool_evidence_recall: 0.7812.

## Questions requiring review

| Question | Type | Missing evidence stages | Gained pages | Lost pages |
|---|---|---|---|---|
| q043 | fact | infosys_fy2025 p15: fusion_cutoff | [] | [] |
| q044 | comparison | infosys_fy2025 p15: fusion_cutoff | [] | [] |
| q046 | fact | infosys_fy2025 p15: reranker_top_k_miss | [] | [] |
| q047 | fact | infosys_fy2024 p15: reranker_top_k_miss | [] | [] |
| q048 | comparison | infosys_fy2024 p15: reranker_top_k_miss; infosys_fy2025 p15: reranker_top_k_miss | [] | [] |
| q049 | fact | infosys_fy2025 p15: reranker_top_k_miss | [] | [['infosys_fy2025', 15]] |
| q050 | fact | infosys_fy2024 p15: reranker_top_k_miss | [] | [] |
| q051 | comparison | infosys_fy2025 p15: reranker_top_k_miss | [] | [['infosys_fy2025', 15]] |
| q053 | fact | infosys_fy2025 p15: reranker_top_k_miss | [] | [] |
| q054 | fact | infosys_fy2025 p15: reranker_top_k_miss | [] | [] |
| q055 | fact | tcs_fy2025 p24: candidate_miss | [] | [] |
| q056 | fact | tcs_fy2025 p24: candidate_miss | [] | [] |
| q058 | follow_up | tcs_fy2024 p8: reranker_top_k_miss | [] | [] |
| q059 | fact | tcs_fy2024 p8: reranker_top_k_miss | [] | [] |

Metrics use (document, PDF page) matches and exclude unanswerable questions. A page hit does not establish that the exact supporting fact was retrieved. Failure stages locate losses, not semantic causes: inspect the saved reference evidence and candidate text before labeling parsing, year confusion, truncation or lexical failures.
