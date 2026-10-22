# Week 4 retrieval evaluation: dev, k=5

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

Macro averages exclude unanswerable questions. These scores measure page retrieval, not answer correctness or abstention. See the JSON for configuration and per-question misses.
