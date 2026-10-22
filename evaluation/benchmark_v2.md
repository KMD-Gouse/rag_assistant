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

- Infosys FY2024/FY2025 employee totals on corporate-overview pages.
- Infosys FY2025 revenue on both corporate-overview and business-highlights pages.
- The March 2024 cash balance repeated in the FY2025 report.

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
| BM25 | 0.7424 | 0.9848 | 0.4449 | 0.6970 |
| RRF hybrid | 0.4545 | 1.0000 | 0.3909 | 0.3939 |
| Reranked | 0.7879 | 1.0000 | 0.6525 | 0.6970 |

The candidate union now covers every required development evidence group.
One location falls below the fusion cutoff and nine fall below the reranker's
top five, so the remaining errors are ranking losses rather than missing or
corrupted corpus pages. Keep the `challenge` split reserved until a new
configuration is frozen; it was not run during this repair.
