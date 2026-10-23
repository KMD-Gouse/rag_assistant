# Benchmark v2 one-time challenge analysis

The challenge split was run once after commit `02d7637` was tagged
`retrieval-v2-pre-challenge`. No labels, retrieval settings, filters, model
revisions, candidate budgets, or ensemble weights were changed after viewing
these results.

The split contains 20 questions, including 17 answerable and 3 unanswerable
questions. The answerable questions require 21 fact groups with 28 accepted
page locations.

## Challenge results

| Retrieval stage | Evidence recall@5 | Document recall@5 | MRR@5 | All evidence hit@5 |
|---|---:|---:|---:|---:|
| Dense | 0.5588 | 1.0000 | 0.4137 | 0.5294 |
| BM25 | 0.6765 | 0.9706 | 0.4265 | 0.6471 |
| Equal-weight RRF hybrid | 0.6471 | 0.9706 | 0.4186 | 0.5882 |
| Frozen final ensemble | **0.8824** | **1.0000** | **0.6814** | **0.8235** |

Against the strongest top-five baseline, BM25, the frozen ensemble improves
evidence recall by 20.59 percentage points and MRR by 25.49 percentage points.
The result is below the tuned development score of 1.0000 recall, which is the
expected pattern for a useful held-out evaluation.

## Failure-stage analysis

The dense/BM25 union contains every required evidence group. The 50-candidate
fusion pool loses one group, and final reranking leaves three groups outside
the top five:

| Question | Failure | Observation |
|---|---|---|
| `q014` | TCS FY2024 employee page below top five | FY2025 evidence is rank 2; the FY2024 page is rank 7. This is a multi-period comparison ranking failure. |
| `q015` | Labelled Infosys growth page outside the fusion pool | TCS evidence is rank 1. Infosys page 73 is rank 2 and appears to state the requested 6.1% growth, but it was not a pre-run gold alternative. The frozen score is retained without post-hoc relabelling. |
| `q027` | Both free-cash-flow pages below top five | The FY2025 page is rank 6 and the FY2024 page is rank 16. Repeated “free cash flow” language in capital-allocation passages occupies the top ranks. |

The `q015` observation is reported as a possible label-coverage limitation,
not used to alter the benchmark or score. This preserves the one-time
challenge protocol.

## Interpretation

The challenge result supports the pipeline improvement: the frozen ensemble
substantially exceeds all three retrieval baselines on recall and MRR. It also
shows that comparison questions remain the main weakness. Final recall is
1.0000 for fact, follow-up, and table questions, but 0.5000 for comparisons.

The challenge split is a held-out regression set for this project, not a claim
of unbiased population performance. Its original questions existed before the
final freeze and had previously been inspected. A stronger future test would
use independently authored questions whose results remain sealed until the
entire answer-generation and citation pipeline is frozen.

Cross-encoder-only challenge performance and the empty-retrieval abstention
proxy were computed afterward from the saved rankings and are explicitly
reported as post-hoc diagnostics in
`evaluation/results/benchmark-v2-posthoc-diagnostics.md`.
