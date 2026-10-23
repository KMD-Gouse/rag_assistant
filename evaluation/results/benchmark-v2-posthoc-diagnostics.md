# Benchmark v2 disclosed post-hoc diagnostics

These diagnostics were computed from the candidate and cross-encoder rankings
saved by the frozen one-time challenge run. Retrieval and model inference were
not rerun. The pre-challenge tag and official challenge metrics are unchanged.
No configuration or label was changed using these results.

## Development component ablation

| Variant | Recall@5 | MRR@5 | All evidence hit@5 |
|---|---:|---:|---:|
| Cross-encoder only | 0.9394 | 0.7586 | 0.9091 |
| Cross-encoder + fusion rank | 0.9394 | 0.7369 | 0.9091 |
| Cross-encoder + phrase signal | 1.0000 | 0.8949 | 1.0000 |
| Full ensemble | **1.0000** | **0.9167** | **1.0000** |

## Challenge component ablation

This table is a disclosed post-hoc diagnostic, not a separately frozen
challenge experiment.

| Variant | Recall@5 | MRR@5 | All evidence hit@5 |
|---|---:|---:|---:|
| Cross-encoder only | **0.9118** | **0.7176** | 0.8235 |
| Cross-encoder + fusion rank | 0.8824 | 0.7157 | 0.8235 |
| Cross-encoder + phrase signal | 0.8529 | 0.6686 | 0.7647 |
| Full ensemble | 0.8824 | 0.6814 | 0.8235 |

The cross-encoder alone generalizes better than the tuned ensemble on challenge
recall and MRR. Fusion rank and exact phrase matching improved development but
did not provide the same improvement on this split. This is evidence of some
development-set overfitting and should be disclosed rather than hidden.

## Empty-retrieval abstention proxy

The project does not yet contain a trained or calibrated answerability model.
The only existing abstention-like behavior occurs when inferred company/year
scope matches no indexed documents and retrieval returns zero chunks.

Using `retrieved_count == 0` as a fixed proxy across all 60 benchmark queries:

| Measure | Result |
|---|---:|
| Unanswerable queries with zero results | 5 / 10 |
| Unanswerable recall | 0.5000 |
| Answerable queries incorrectly abstained | 0 / 50 |
| Abstention precision | 1.0000 |
| Overall 60-query accuracy | 0.9167 |
| F1 | 0.6667 |

Overall accuracy is high because answerable queries are the majority and none
return an empty result. The important number for the ten unanswerable queries
is the 0.5000 recall: five unsupported future-year questions are rejected by
scope, while five unsupported-but-in-domain questions still retrieve passages.
This proxy must not be presented as a complete abstention system.
