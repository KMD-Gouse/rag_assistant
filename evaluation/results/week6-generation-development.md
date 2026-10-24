# Week 6 generation development evaluation

Completed outputs: 40; errors: 0. Groq outputs were captured with threshold 0.00.

Automatic lexical overlap is diagnostic only. Final answer correctness, citation entailment, and citation completeness require the review CSV.

| Threshold | Answerability accuracy | Answerable coverage | Unanswerable abstention | Balanced accuracy | End-to-end success |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 80.00% | 75.76% | 100.00% | 87.88% | 80.00% |
| 0.50 | 80.00% | 75.76% | 100.00% | 87.88% | 80.00% |
| 0.60 | 80.00% | 75.76% | 100.00% | 87.88% | 80.00% |
| 0.70 | 80.00% | 75.76% | 100.00% | 87.88% | 80.00% |
| 0.80 | 80.00% | 75.76% | 100.00% | 87.88% | 80.00% |
| 0.90 | 80.00% | 75.76% | 100.00% | 87.88% | 80.00% |
| 0.95 | 80.00% | 75.76% | 100.00% | 87.88% | 80.00% |
| 1.00 | 80.00% | 75.76% | 100.00% | 87.88% | 80.00% |

## Score distribution

Observed scores: 0.00, 1.00

All generated scores are binary, so thresholds from 0.50 through 1.00 do not change a single development decision. The former 0.70 default is therefore not empirically special.

## Automatic citation and failure audit

Answered gold-answerable questions: 25. Every accepted answer passed exact-quote validation; 25/25 cited every required gold evidence group.

Abstention reasons: invalid_citations=6, model_says_unanswerable=6, no_retrieval_results=3.
Answerable false abstentions: invalid_citations=6, model_says_unanswerable=2. Correct unanswerable abstentions: model_says_unanswerable=4, no_retrieval_results=3.

Gold-page citation precision is only diagnostic because valid supporting pages can be absent from the benchmark labels. Citation entailment and completeness still require review of the quoted text.

## Second-person review

Review complete. Every non-abstained answerable output has a manual answer-correctness, citation-entailment, and citation-completeness decision.

Human review file: `evaluation\results\week6-generation-development-review.csv`
