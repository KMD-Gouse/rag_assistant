# Week 5 retrieval ablation

| Method | Group | Answerable / total | Evidence recall | Document recall | MRR | All evidence hit |
|---|---|---:|---:|---:|---:|---:|
| v1_dense | all | 34 / 40 | 0.2941 | 0.9559 | 0.2167 | 0.2647 |
| v1_dense | comparison | 8 / 8 | 0.2500 | 0.8750 | 0.3125 | 0.1250 |
| v1_dense | fact | 16 / 16 | 0.2500 | 1.0000 | 0.1167 | 0.2500 |
| v1_dense | follow_up | 4 / 4 | 0.5000 | 0.8750 | 0.3750 | 0.5000 |
| v1_dense | table | 6 / 6 | 0.3333 | 1.0000 | 0.2500 | 0.3333 |
| v1_dense | unanswerable | 0 / 6 | N/A | N/A | N/A | N/A |
| bm25 | all | 34 / 40 | 0.3971 | 0.8676 | 0.2598 | 0.3824 |
| bm25 | comparison | 8 / 8 | 0.1875 | 0.7500 | 0.1875 | 0.1250 |
| bm25 | fact | 16 / 16 | 0.3750 | 0.9375 | 0.2188 | 0.3750 |
| bm25 | follow_up | 4 / 4 | 0.5000 | 0.6250 | 0.3333 | 0.5000 |
| bm25 | table | 6 / 6 | 0.6667 | 1.0000 | 0.4167 | 0.6667 |
| bm25 | unanswerable | 0 / 6 | N/A | N/A | N/A | N/A |
| v2_hybrid | all | 34 / 40 | 0.4118 | 0.9265 | 0.3049 | 0.3529 |
| v2_hybrid | comparison | 8 / 8 | 0.2500 | 0.7500 | 0.3167 | 0.0000 |
| v2_hybrid | fact | 16 / 16 | 0.5000 | 1.0000 | 0.2708 | 0.5000 |
| v2_hybrid | follow_up | 4 / 4 | 0.5000 | 0.8750 | 0.5000 | 0.5000 |
| v2_hybrid | table | 6 / 6 | 0.3333 | 1.0000 | 0.2500 | 0.3333 |
| v2_hybrid | unanswerable | 0 / 6 | N/A | N/A | N/A | N/A |
| v3_reranked | all | 34 / 40 | 0.5735 | 0.8824 | 0.4260 | 0.5000 |
| v3_reranked | comparison | 8 / 8 | 0.5000 | 0.8125 | 0.4625 | 0.2500 |
| v3_reranked | fact | 16 / 16 | 0.6250 | 0.8750 | 0.4396 | 0.6250 |
| v3_reranked | follow_up | 4 / 4 | 0.6250 | 0.8750 | 0.5625 | 0.5000 |
| v3_reranked | table | 6 / 6 | 0.5000 | 1.0000 | 0.2500 | 0.5000 |
| v3_reranked | unanswerable | 0 / 6 | N/A | N/A | N/A | N/A |

## Timing

| Method | Median query latency (ms) |
|---|---:|
| v1_dense | 61.0 |
| bm25 | 21.5 |
| v2_hybrid | 83.6 |
| v3_reranked | 4079.7 |

Timing excludes corpus loading, model loading and one warm-up query. Hybrid/v3 timings sum sequential stage times; these are single-run measurements, not a production throughput benchmark. All queries, including unanswerable ones, are timed.

## Evidence loss stages

| Stage | Evidence locations |
|---|---:|
| corpus_page_missing | 0 |
| candidate_miss | 6 |
| fusion_cutoff | 3 |
| reranker_top_k_miss | 10 |
| hit | 24 |

Macro union_evidence_recall: 0.8676.

Macro pool_evidence_recall: 0.8235.

## Questions requiring review

| Question | Type | Missing evidence stages | Gained pages | Lost pages |
|---|---|---|---|---|
| q003 | comparison | infosys_fy2025 p10: fusion_cutoff | [] | [] |
| q005 | fact | infosys_fy2024 p23: reranker_top_k_miss | [] | [] |
| q008 | fact | tcs_fy2025 p131: reranker_top_k_miss | [] | [['tcs_fy2025', 131]] |
| q009 | fact | tcs_fy2025 p88: reranker_top_k_miss | [] | [] |
| q010 | table | infosys_fy2024 p15: reranker_top_k_miss | [] | [] |
| q013 | comparison | infosys_fy2024 p23: candidate_miss | [['infosys_fy2025', 24]] | [] |
| q014 | comparison | tcs_fy2025 p131: reranker_top_k_miss | [['tcs_fy2024', 128]] | [['tcs_fy2025', 131]] |
| q015 | comparison | infosys_fy2025 p23: candidate_miss; tcs_fy2025 p88: reranker_top_k_miss | [] | [] |
| q016 | follow_up | infosys_fy2024 p23: candidate_miss | [] | [] |
| q018 | follow_up | infosys_fy2025 p10: fusion_cutoff | [['tcs_fy2025', 88]] | [] |
| q025 | table | infosys_fy2025 p15: reranker_top_k_miss | [] | [] |
| q026 | table | infosys_fy2024 p15: reranker_top_k_miss | [] | [] |
| q027 | comparison | infosys_fy2024 p15: fusion_cutoff; infosys_fy2025 p15: reranker_top_k_miss | [] | [] |
| q032 | fact | tcs_fy2025 p13: candidate_miss | [] | [] |
| q033 | comparison | tcs_fy2025 p13: candidate_miss | [] | [] |
| q035 | fact | tcs_fy2025 p13: candidate_miss | [] | [] |
| q036 | fact | tcs_fy2025 p23: reranker_top_k_miss | [] | [] |

Metrics use (document, PDF page) matches and exclude unanswerable questions. A page hit does not establish that the exact supporting fact was retrieved. Failure stages locate losses, not semantic causes: inspect the saved reference evidence and candidate text before labeling parsing, year confusion, truncation or lexical failures.
