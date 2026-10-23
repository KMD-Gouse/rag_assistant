# Benchmark v2 final development analysis

This document records the decisions made before the one-time challenge run.
The development split has 40 questions, of which 33 are answerable. Its 41
required fact groups have 56 accepted page locations after the final label
audit. Metrics match `(doc_id, pdf_page)` and use the first relevant page for
MRR. Unanswerable questions are excluded from retrieval recall and MRR.

## Label audit

The audit applies one rule: pages are alternatives only when they explicitly
state the same metric, company, period, and scope. A later annual report may be
an alternative when it explicitly restates the same historical fact. Merely
containing the same number is insufficient.

The following inconsistencies were corrected before rescoring:

- `q003` and `q018` now accept the same three Infosys FY2025 revenue pages.
- `q005`, `q013`, and `q016` now accept the same three Infosys FY2024 employee
  pages.
- `q042` and `q044` now accept the same two pages for March 2024 cash and
  investments; `q043` and `q044` do the same for March 2025.
- `q049` and `q051` now accept the same two pages for Infosys FY2025 large-deal
  TCV.

Validation now rejects a named fact group if its accepted locations differ
between questions. Numeric evidence tokens are also checked against every
grouped alternative page.

## Corrected development results

| Retrieval stage | Evidence recall@5 | MRR@5 | All evidence hit@5 |
|---|---:|---:|---:|
| Dense | 0.3333 | 0.2409 | 0.3030 |
| BM25 | 0.8182 | 0.5116 | 0.7576 |
| Equal-weight RRF hybrid | 0.5152 | 0.4045 | 0.4545 |
| Cross-encoder only | 0.9394 | 0.7586 | 0.9091 |
| Final ensemble | **1.0000** | **0.9167** | **1.0000** |

The earlier manually calculated cross-encoder MRR of 0.6876 averaged reciprocal
ranks across required facts. That was inconsistent with the project's metric,
which uses the first relevant retrieved page. The corrected cross-encoder MRR
is 0.7586.

## Hybrid decision

Equal-weight RRF underperforms BM25 at top five, but every tested fusion setting
retains 1.0000 evidence recall in the top-50 pool. Selected development checks
are:

| Dense weight | BM25 weight | Page cap | Recall@5 | MRR@5 | Pool recall@50 |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | none | 0.5152 | 0.4045 | 1.0000 |
| 0.05 | 1.00 | none | 0.7576 | 0.5399 | 1.0000 |
| 0.05 | 1.00 | 1 | 0.8182 | 0.5561 | 1.0000 |
| 0.00 | 1.00 | none | 0.8182 | 0.5116 | 1.0000 |

The final configuration retains equal-weight, chunk-level RRF. The apparent
top-five fixes either nearly remove dense retrieval or impose a page cap that
directly optimizes the evaluation unit and can retain the wrong chunk from a
correct page. RRF is used as a candidate generator here; its top-50 coverage,
rather than its top-five ordering, is what enables reranking. Hybrid top-five
underperformance remains a documented limitation.

## Final-ranker ablation

| Variant | Fusion weight | Cross-encoder weight | Phrase weight | Recall@5 | MRR@5 | All hit@5 |
|---|---:|---:|---:|---:|---:|---:|
| Cross-encoder only | 0.0 | 1.0 | 0.0 | 0.9394 | 0.7586 | 0.9091 |
| Hybrid + cross-encoder | 0.5 | 4.0 | 0.0 | 0.9394 | 0.7369 | 0.9091 |
| Cross-encoder + phrase | 0.0 | 4.0 | 0.005 | 1.0000 | 0.8949 | 1.0000 |
| Hybrid + phrase | 0.5 | 0.0 | 0.005 | 0.9242 | 0.8384 | 0.8788 |
| Full ensemble | 0.5 | 4.0 | 0.005 | **1.0000** | **0.9167** | **1.0000** |

The exact content-bigram signal closes the remaining recall failures. Hybrid
rank improves ordering when combined with the cross-encoder and phrase signal,
but does not improve raw cross-encoder recall by itself.

## Weight sensitivity

The sensitivity grid tested 27 nearby configurations:

- Hybrid-rank weight: 0.25, 0.50, 0.75
- Cross-encoder-rank weight: 3.0, 4.0, 5.0
- Phrase weight: 0.0025, 0.0050, 0.0075

Fourteen of the 27 combinations achieved both 1.0000 evidence recall and
1.0000 all-evidence-hit rate. The frozen `(0.5, 4.0, 0.005)` configuration was
retained rather than replaced by the highest development MRR found in the
sensitivity grid. This avoids another round of development-set optimization.

## Interpretation boundary

The final 1.0000 result is a tuned development result, not an unseen-test
claim. Labels were never passed to retrieval or used to train model weights,
but development failures informed extraction repair, label correction, scope
handling, and rank-ensemble design. Page-level evidence retrieval also does
not establish exact-span support or generated-answer correctness.

The exact frozen configuration is saved in
`evaluation/results/benchmark-v2-frozen-config.json`. Challenge results must be
reported without changing the frozen configuration afterward.
