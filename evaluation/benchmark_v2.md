# Benchmark v2 design

Benchmark v2 preserves the original 60 questions while fixing two evaluation
problems discovered in Week 5: concentrated evidence pages and incomplete
alternative-page labels.

## Split policy

The 20-question `challenge` split is stratified as 9 fact, 4 comparison,
2 table, 2 follow-up, and 3 unanswerable questions. Its answerable evidence
is spread across 15 canonical pages; no canonical page supplies more than two
challenge questions. Canonical evidence counts by report are reasonably even:
Infosys FY2024 7, Infosys FY2025 6, TCS FY2024 5, and TCS FY2025 5.

The remaining 40 questions form `development`. The former held-out questions
have already been inspected, so `challenge` is a diversity regression set,
not a new untouched test set. Do not describe its results as unbiased final
generalization. A future release needs newly authored questions and a sealed
test split.

## Alternative evidence

Evidence entries can share a `group`. Locations in one group are interchangeable
support for one required fact; different groups remain jointly required. The
following additions were checked against extracted text and rendered pages:

- Infosys employee totals repeated across corporate-overview, human-capital,
  and Board-report pages.
- Infosys revenue, cash, client-count, and large-deal figures repeated in
  business highlights, management discussion, and strategy sections.
- TCS dividend totals repeated in the CEO letter and Board report.

Evaluation now computes recall over required fact groups. Document recall also
accepts an alternative document within the same fact group. The original
`questions.jsonl` remains unchanged so Week 4/5 reports remain reproducible.

Regenerate and validate with:

```powershell
python -m evaluation.build_benchmark_v2
python -m evaluation.validate_benchmark_v2
```

Run a scoped development diagnostic after rebuilding the index:

```powershell
python -m evaluation.week5 --dataset evaluation/benchmark_v2.jsonl --split development --auto-scope --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a
```

Company and fiscal-year constraints are inferred only from explicit query text
and supplied history. `FY2025` maps to report year `2024-25`; an explicit future
year such as `FY2025-26` correctly matches no indexed annual report. Comparisons
retain every explicitly named company and year. A follow-up that asks for the
`previous fiscal year` shifts the explicit year in its supplied history back by
one report year. No gold label supplies filters.

## Verified repair and development diagnostic

The rebuilt processed corpus contains zero Unicode replacement characters and
zero TCS digit-substitution code points across all four reports. Visual checks
against the TCS FY2024-25 report confirmed `607,979` associates, revenue from
operations of `255,324`, and the workforce table values `6,07,979`, `64.8`,
`6,36,833`, and `35.3`. Qdrant contains 4,083 version-3 points matching the
rebuilt chunk files.

The scoped `development` diagnostic produced:

| Method | Evidence recall@5 | Document recall@5 | MRR@5 | All evidence hit@5 |
|---|---:|---:|---:|---:|
| Dense | 0.3333 | 0.9697 | 0.2409 | 0.3030 |
| BM25 | 0.8182 | 0.9848 | 0.5116 | 0.7576 |
| RRF hybrid | 0.5152 | 1.0000 | 0.4045 | 0.4545 |
| Reranked ensemble | 1.0000 | 1.0000 | 0.9167 | 1.0000 |

The final ranker combines hybrid rank, cross-encoder rank, and a small exact
content-bigram signal. Every required development evidence group now appears
in the top five: corpus misses, candidate misses, fusion-cutoff misses, and
final top-five misses are all zero. There are no unresolved labeled-evidence
review items. Keep the `challenge` split reserved until this configuration is
frozen; it was not run during this repair.

The final pre-challenge audit, fusion decision, component ablations, corrected
cross-encoder MRR, and 27-run weight-sensitivity grid are recorded in
`evaluation/results/benchmark-v2-final-development-analysis.md`. The locked
models, hashes, filters, candidate budgets, and weights are recorded in
`evaluation/results/benchmark-v2-frozen-config.json`.

After commit `02d7637` was tagged `retrieval-v2-pre-challenge`, the challenge
split was run once without subsequent tuning. The frozen ensemble achieved
0.8824 evidence recall@5 and 0.6814 MRR@5, compared with BM25 at 0.6765 and
0.4265. The immutable result and three remaining failures are documented in
`evaluation/results/benchmark-v2-challenge-analysis.md`.
