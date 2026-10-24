# Week 6 generation challenge evaluation

Completed outputs: 20; errors: 0. Groq outputs were captured with threshold 1.00.

Automatic lexical overlap is diagnostic only. Final answer correctness, citation entailment, and citation completeness require the review CSV.

| Threshold | Answerability accuracy | Answerable coverage | Unanswerable abstention | Balanced accuracy | End-to-end success |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 80.00% | 71.43% | 100.00% | 85.71% | 70.00% |
| 0.50 | 80.00% | 71.43% | 100.00% | 85.71% | 70.00% |
| 0.60 | 80.00% | 71.43% | 100.00% | 85.71% | 70.00% |
| 0.70 | 80.00% | 71.43% | 100.00% | 85.71% | 70.00% |
| 0.80 | 80.00% | 71.43% | 100.00% | 85.71% | 70.00% |
| 0.90 | 80.00% | 71.43% | 100.00% | 85.71% | 70.00% |
| 0.95 | 80.00% | 71.43% | 100.00% | 85.71% | 70.00% |
| 1.00 | 80.00% | 71.43% | 100.00% | 85.71% | 70.00% |

## Score distribution

Observed scores: 0.00, 1.00

All generated scores are binary, so thresholds from 0.50 through 1.00 do not change a single challenge decision. The challenge was executed with the frozen threshold of 1.00.

## Automatic citation and failure audit

Answered gold-answerable questions: 10. Every accepted answer passed exact-quote validation; 8/10 cited every required gold evidence group.

Abstention reasons: invalid_citations=4, model_says_unanswerable=4, no_retrieval_results=2.
Answerable false abstentions: invalid_citations=4. Correct unanswerable abstentions: model_says_unanswerable=4, no_retrieval_results=2.

Gold-page citation precision is only diagnostic because valid supporting pages can be absent from the benchmark labels. Citation entailment and completeness still require review of the quoted text.

## Second-person review

Review complete. Every non-abstained answerable output has a manual answer-correctness, citation-entailment, and citation-completeness decision.

Human review file: `evaluation\results\week6-generation-challenge-review.csv`

## Final human-reviewed metrics

Generated answer correctness: 9/10 (90.00%).
Citation entailment: 8/10 (80.00%).
Citation completeness: 8/10 (80.00%).
Fully successful generated answers: 8/10 (80.00%).
Overall end-to-end success, including correct abstentions: 70.00%.
