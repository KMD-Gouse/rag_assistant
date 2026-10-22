# Week 5 final ablation and failure analysis

Completed on 22 October 2026. v3 uses `cross-encoder/ms-marco-MiniLM-L-6-v2`, cached revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Settings were frozen in `week5-frozen-config.json` before the held-out run. No parameter tuning was performed after viewing held-out results.

## Decision

The reranker improves dev retrieval but does not generalize reliably to the held-out split. Do not promote v3 as the default based on this experiment. BM25 is the strongest measured held-out baseline. Keep all methods available and address parsing, report/year selection, and evaluation-label coverage before adding generation complexity.

## Final ablation

50 dense candidates + 50 BM25 candidates; RRF constant 60; rerank the top 50 fused candidates; final cutoff 5; 512-token query/passage pairs; batch size 8. Same query/history policy and corpus for every method.

| Split | Method | Evidence recall@5 | Document recall@5 | MRR@5 | All evidence hit@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|
| dev | v1_dense | 29.41% | 95.59% | 0.2167 | 26.47% | 61.0 |
| dev | bm25 | 39.71% | 86.76% | 0.2598 | 38.24% | 21.5 |
| dev | v2_hybrid | 41.18% | 92.65% | 0.3049 | 35.29% | 83.6 |
| dev | v3_reranked | 57.35% | 88.24% | 0.4260 | 50.00% | 4079.7 |
| held_out | v1_dense | 0.00% | 100.00% | 0.0000 | 0.00% | 33.9 |
| held_out | bm25 | 56.25% | 100.00% | 0.3063 | 56.25% | 12.2 |
| held_out | v2_hybrid | 15.62% | 100.00% | 0.0990 | 12.50% | 46.4 |
| held_out | v3_reranked | 18.75% | 87.50% | 0.0688 | 12.50% | 2400.9 |

Dev has 34 answerable / 40 total questions; held-out has 16 answerable / 20 total. Quality metrics exclude unanswerable questions. Latency includes them. The held-out set is small and its evidence is concentrated on a few report pages; these figures should not be interpreted as broad statistical proof.

Dev evaluation ran partly alongside tests, so its latency includes possible CPU contention. Held-out evaluation ran without another model-inference job. Timing excludes corpus/model loading and one warm-up; methods reuse retrieval stages and report sequential stage sums. Cross-split latency is not a controlled performance comparison.

## Evidence loss locations

Counts below are distinct expected (document, page) locations per question, not question counts or macro recall.

| Split | Missing corpus page | Candidate miss | Fusion cutoff | Reranker top-5 miss | Hit |
|---|---:|---:|---:|---:|---:|
| dev | 0 | 6 | 3 | 10 | 24 |
| held_out | 0 | 2 | 2 | 11 | 4 |

Candidate-union macro evidence recall is 86.76% on dev and 87.50% held-out. The top-50 fused pool retains 82.35% and 78.12%, respectively. Thus many expected pages enter the pool but fail to reach the final top five. Missing corpus pages = 0 does not prove the text was extracted correctly.

## Manual review of representative failures

| Question | Observed evidence | Interpretation / next experiment |
|---|---|---|
| q003, dev revenue comparison | Infosys FY2025 page 10 appears at fused rank 95, outside the top-50 rerank pool; TCS FY2025 page 88 reaches final rank 5. | A fusion-cutoff loss. A larger or diversity-preserving pool is a future dev experiment; the current reranker cannot recover excluded passages. |
| q008, dev TCS permanent employees | Correct FY2025 page 131 moves from fused rank 1 to reranked rank 6. Final rank 1 is FY2024 page 128 with 6,01,546 employees instead of the requested FY2025 value 6,07,979. | Confirmed wrong-year ranking regression. Metadata-aware reranker input or query-derived year constraints merit testing on a new development benchmark. |
| q013, dev Infosys employee comparison | The labeled FY2024 page 23 is absent from candidates. Final results nevertheless include FY2024 page 10 with 3,17,240 and FY2025 page 10 with 3,23,578. | Evidence labels omit alternative supporting pages. Current page recall undercounts this retrieved support. Preserve the frozen score and add alternative-evidence annotations in a separately versioned dataset. |
| q025, dev free cash flow | The exact FY2025 page 15 chunk contains 34,549 and 44.8%; it moves from fused rank 19 to reranked rank 8. | The evidence survives candidate selection but remains below cutoff. This is a ranking failure on a compact financial highlights passage. |
| q032, dev TCS attrition | The reference page 13 exists, but is absent from the candidate union. Its text contains broken forms such as `L T M`, `serv ices`, and `13. 3%`. Final rank 1 is FY2024 page 18 with 12.5%. | Observed extraction corruption and wrong-year output. Corruption is a plausible contributor; this run does not isolate its causal effect. |
| q042, held-out cash and investments | Final rank 1 is FY2025 page 33, but its comparison explicitly states the requested March 2024 consolidated amount of 39,005 crore. The labeled FY2024 page 32 is rank 2. | A different report year does not automatically imply an incorrect fact: reports repeat prior-year values. Review passage content before labeling year failures. |

These are representative manual checks, not an exhaustive semantic classification of every failure. Full candidate text, stage labels, and reranked orders are retained in each split JSON. No held-out questions or labels were edited.

## Recommended next work

1. Audit PDF text extraction, especially TCS FY2025 broken words and numerals, and verify table rows preserve headers and units.
2. Build a separately versioned, more diverse evidence benchmark with alternative supporting passages. Several held-out questions target the same Infosys business-highlights page; avoid treating them as independent coverage of the whole corpus.
3. Evaluate report/year-aware query handling and metadata-aware reranker inputs. Account for comparisons and historical values repeated in later reports.
4. Compare candidate budgets and fusion weights on development data, retaining BM25 as the control. Do not tune on this now-inspected held-out split; reserve a fresh test set for the next iteration.
5. Measure answer generation, citations, and abstention separately when retrieval quality is sufficient. None of these retrieval metrics establishes answer correctness.

## Reproducibility and scope

Validation: all 92 tests passed, including live Qdrant integration tests and
the new reranker/failure-stage regressions. The one remaining warning is
the existing client/server version mismatch described below.

Dataset and corpus fingerprints match Week 4 and both Week 5 runs. The three baseline dev metrics exactly reproduce Week 4. The collection was not modified. The default application method remains hybrid; reranking is opt-in. The configured BGE model remains available, but this ablation evaluated MiniLM only.

The existing Qdrant client 1.19.1 / server 1.15.3 compatibility warning remains; requests succeeded. Execution used bundled Python 3.12.14 with project site-packages because the virtual-environment launcher lacks its original base Python. The dev report could not extract a model revision from the model object; the frozen configuration records the independently verified cached revision and held-out explicitly pins it.

Artifacts: `week5-dev.json` / `.md`, `week5-held_out.json` / `.md`, and `week5-frozen-config.json`. Page-level relevance is a proxy; unanswerable questions receive no recall/MRR score and no abstention claim is made.
