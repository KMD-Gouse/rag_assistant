# Evaluation reference

[Back to the project README](../README.md).

These results describe the frozen historical baseline, not the newer application.
Use the recorded tags and configuration artifacts for provenance; running today’s
code against today’s expanded corpus is not an exact reconstruction of those runs.

## Baseline implementation

### Ingestion and vector search

- Manifest-backed document identity and checksums.
- PDF page extraction using PyMuPDF.
- Verified repair of broken TCS embedded-font mappings.
- Prose and table-aware chunks with company, fiscal year, page, section, and
  source metadata.
- BGE-small embeddings stored in Qdrant.
- Index verification and filtered dense search.

### Retrieval baselines

- Dense semantic retrieval.
- In-memory BM25 over the exact Qdrant text payloads.
- Equal-weight reciprocal-rank fusion.
- Page-level evidence recall, MRR, document recall, and all-evidence-hit
  evaluation.

BM25 uses `k1=1.5`, `b=0.75`, positive IDF, case-folded word tokens, and
commas removed within numbers. RRF uses `1 / (60 + rank)` with ranks starting
at one. Retrieval scores and fusion scores are ordering signals, not
probabilities.

### Reranking and failure analysis

- `cross-encoder/ms-marco-MiniLM-L-6-v2` with a pinned model revision.
- Top-50 hybrid candidate pool reranked to top five.
- Final rank ensemble combining hybrid rank, cross-encoder rank, and a small
  exact content-bigram signal.
- Ablations, ±50% weight sensitivity, and failure-stage analysis.
- Corrected evidence labels with grouped alternative pages.

### Grounded answer generation

- Groq Chat Completions using `openai/gpt-oss-20b`.
- Strict structured output validated with Pydantic.
- Trusted citation metadata resolved from retrieved evidence IDs.
- Exact-quote validation against retrieved chunk text.
- Fixed abstention response and evidence-sufficiency gate.
- Resumable development and challenge evaluation.
- Second-person review of answer correctness, citation entailment, and
  citation completeness.

The model produced only `0.0` and `1.0` evidence-sufficiency values during
development. Thresholds from `0.50` through `1.00` therefore made identical
decisions. The frozen `1.0` threshold is a strict semantic gate, not a
calibrated probability.

## Final retrieval results

The benchmark-v2 development split contains 40 questions, including 33
answerable questions. The one-time retrieval challenge contains 20 questions,
including 17 answerable questions.

### Development split

| Retrieval stage | Evidence recall@5 | MRR@5 | All evidence hit@5 |
|---|---:|---:|---:|
| Dense | 33.33% | 24.09% | 30.30% |
| BM25 | 81.82% | 51.16% | 75.76% |
| Equal-weight RRF | 51.52% | 40.45% | 45.45% |
| Cross-encoder only | 93.94% | 75.86% | 90.91% |
| Frozen final ensemble | **100.00%** | **91.67%** | **100.00%** |

These are tuned development results and are not treated as unseen-test
performance.

### One-time challenge split

| Retrieval stage | Evidence recall@5 | MRR@5 | All evidence hit@5 |
|---|---:|---:|---:|
| Dense | 55.88% | 41.37% | 52.94% |
| BM25 | 67.65% | 42.65% | 64.71% |
| Equal-weight RRF | 64.71% | 41.86% | 58.82% |
| Frozen final ensemble | **88.24%** | **68.14%** | **82.35%** |

The retrieval configuration was frozen at tag
`retrieval-v2-pre-challenge`. No labels or ranking settings were changed after
the challenge result was observed.

## Final generation results

Generation uses a separately authored 20-question challenge containing 14
answerable and six unanswerable questions. The configuration was frozen at
tag `week6-generation-v1-pre-challenge` before this challenge was executed.

| Metric | Development | Challenge |
|---|---:|---:|
| Answerability accuracy | 80.00% | 80.00% |
| Answerable coverage | 75.76% | 71.43% |
| Unanswerable abstention accuracy | 100.00% | 100.00% |
| Balanced answerability accuracy | 87.88% | 85.71% |
| Human-reviewed end-to-end success | **80.00%** | **70.00%** |

Among the ten generated challenge answers:

- Answer correctness: 9/10 (90%).
- Citation entailment: 8/10 (80%).
- Citation completeness: 8/10 (80%).
- Fully correct and fully supported answers: 8/10 (80%).

Overall challenge success is 14/20: eight fully successful generated answers
plus six correct abstentions. Four answerable questions were false
abstentions, one generated answer was incorrect, and one otherwise correct
answer lacked adequate citation support. These failures are retained in the
reported result; the pipeline was not tuned after the challenge run.

## Evaluation protocol

The project follows this order:

1. Build and inspect the development benchmark.
2. Tune parsing, labels, retrieval, ranking, generation, and thresholds only
   on development data.
3. Complete second-person review of generated development answers.
4. Save hashes, model revisions, settings, and metrics in a frozen-config
   JSON file.
5. Commit and tag the frozen pipeline.
6. Validate and commit the fresh challenge dataset before running it.
7. Run the challenge with the frozen configuration.
8. Complete human review and report the result without further tuning.

The development and challenge JSON files are machine-readable sources of
truth. Review CSV files provide the human judgments. Markdown files are
derived reports. Frozen-config and provenance JSON files record hashes and
the exact experiment state.

Important artifacts:

- [`evaluation/benchmark_v2.md`](../evaluation/benchmark_v2.md): retrieval
  benchmark design.
- [`evaluation/results/benchmark-v2-final-development-analysis.md`](../evaluation/results/benchmark-v2-final-development-analysis.md): label audit,
  ablations, and frozen retrieval decisions.
- [`evaluation/results/benchmark-v2-challenge-analysis.md`](../evaluation/results/benchmark-v2-challenge-analysis.md): one-time retrieval challenge.
- [`evaluation/results/benchmark-v2-frozen-config.json`](../evaluation/results/benchmark-v2-frozen-config.json): exact retrieval freeze.
- [`evaluation/results/week6-generation-development.md`](../evaluation/results/week6-generation-development.md): reviewed generation development result.
- [`evaluation/results/week6-generation-frozen-config.json`](../evaluation/results/week6-generation-frozen-config.json): exact generation freeze.
- [`evaluation/results/week6-generation-challenge.md`](../evaluation/results/week6-generation-challenge.md): final reviewed generation challenge.
- [`evaluation/results/week6-generation-challenge-provenance.json`](../evaluation/results/week6-generation-challenge-provenance.json): challenge hashes and run provenance.

## Reproduce the generation evaluation

Development output capture:

```powershell
python -m evaluation.generation `
  --dataset evaluation/benchmark_v2.jsonl `
  --split development `
  --prefix evaluation/results/week6-generation-development `
  --delay-seconds 2 `
  --capture-threshold 0.0
```

After completing the review CSV, regenerate the report and frozen config
without regenerating completed Groq outputs:

```powershell
python -m evaluation.generation `
  --dataset evaluation/benchmark_v2.jsonl `
  --split development `
  --prefix evaluation/results/week6-generation-development `
  --delay-seconds 0 `
  --capture-threshold 0.0 `
  --freeze-config evaluation/results/week6-generation-frozen-config.json
```

Validate the separately authored challenge:

```powershell
python -m evaluation.validate_generation_challenge
```

The official challenge has already been run. The following command documents
how it was produced; running it now will reuse completed saved outputs rather
than create a new official result:

```powershell
python -m evaluation.generation `
  --dataset evaluation/generation_challenge_v1.jsonl `
  --split challenge `
  --prefix evaluation/results/week6-generation-challenge `
  --delay-seconds 0 `
  --capture-threshold 1.0
```

`evaluation.generation` saves each successful output immediately and skips
completed question IDs when resumed. Runtime-error rows may be retried with
the identical frozen configuration; successful rows are not sampled again.
