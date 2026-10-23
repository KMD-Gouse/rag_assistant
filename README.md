# Annual Report Q&A

A citation-focused RAG application for answering and comparing questions across
Infosys and TCS annual reports.

The project is being built incrementally. Ingestion parses, chunks, embeds,
and indexes reports in Qdrant. Week 4 adds dense retrieval, BM25, RRF hybrid
retrieval, and a reproducible retrieval evaluation.

The initial corpus contains Infosys and TCS annual reports for FY2023–24 and
FY2024–25.

## Setup and ingestion

Use Python 3.11 or 3.12, install the project with `pip install -e ".[dev]"`,
and start Qdrant with `docker compose up -d`. Configure `.env` using
`.env.example`. With the report PDFs in the locations specified by
`data/manifest.json`, index all four reports:

```powershell
python -m app.ingestion.pipeline --all
python -m evaluation.validate_dataset
```

## Week 4: retrieval

- **v1:** existing BGE dense retrieval.
- **BM25:** lexical baseline using the exact text payloads in the dense collection.
- **v2:** equal-weight reciprocal-rank fusion of dense and BM25 candidates.

```powershell
python -m app.pipeline "How many employees did Infosys report in FY2024-25?" --method dense
python -m app.pipeline "How many employees did Infosys report in FY2024-25?" --method bm25
python -m app.pipeline "How many employees did Infosys report in FY2024-25?" --method hybrid --limit 5 --candidates 50 --rrf-k 60
```

All methods accept `--doc-id`, `--company`, and `--fiscal-year` exact-match
filters. BM25 uses k1=1.5, b=0.75, positive IDF, case-folded word tokens, and
commas removed inside numbers. It does not stem or remove stop words.
It returns only chunks with lexical matches. Corpus-wide BM25 statistics
remain fixed when applying filters.

BM25 reads the Qdrant collection once into memory per CLI run. This is suitable
for the four-report corpus; rebuild the index after ingestion. RRF sums
`1 / (60 + rank)` by Qdrant point ID, with ranks starting at one, and uses
point IDs to break score ties. Dense, BM25, and fusion scores have different
meanings. Fusion scores are not confidence probabilities.

## Week 4: v1 versus v2 evaluation

```powershell
python -m evaluation.run_eval --split dev --k 5 --candidates 50 --output evaluation/results/week4-dev.json
```

The runner compares all three methods on the same queries and corpus. It
writes JSON with configuration, dataset/corpus fingerprints, retrieved text,
and per-question missing evidence, plus a Markdown comparison table. The
table includes page evidence recall, document recall, MRR, and the fraction
of questions with all evidence retrieved, overall and by question type.
Metrics are macro-averaged over answerable questions only.

An evidence match means the same `(doc_id, pdf_page)`. The cutoff counts
chunks, so repeated chunks from one page consume retrieval slots without
increasing recall. This page-level proxy does not establish that a chunk
contains the precise supporting fact. Unanswerable queries are retrieved
and logged but receive N/A metrics; answer correctness and abstention need
a later generation evaluation.

Follow-up queries concatenate the supplied conversation history and current
question identically for all methods. Gold answers and evidence never enter
queries or metadata filters. This tests supplied-history retrieval, not a
query rewriter or predicted conversation history.

Keep the collection unchanged while evaluating. Tune candidate depth and
RRF parameters on dev only. Freeze settings before the held-out run:

```powershell
python -m evaluation.run_eval --split held_out --k 5 --candidates 50 --output evaluation/results/week4-held-out.json
python -m pytest -m "not integration" -q
```

Use separate output paths to retain comparisons; the runner overwrites the
specified output files. Corpus fingerprints cover point IDs and payloads,
not vector bytes. Keep the embedding model and ingestion settings fixed too.

## Week 5: cross-encoder reranking and final ablation

**v3** scores each (query, passage) pair with a cross-encoder. Retrieve 50
dense and 50 BM25 candidates, fuse their union, retain the top 50, then
rerank those same 50 to return five. v2 returns the first five from that
same fused pool. No gold evidence or document filters enter either method.

```powershell
python -m app.pipeline "How many employees did Infosys report in FY2024-25?" --method reranked --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2
python -m evaluation.week5 --split dev --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2
```

The first measured experiment uses the locally cached MiniLM cross-encoder.
The application default still follows `RERANKER_MODEL` (BGE by default);
pass the explicit model above to reproduce this experiment. Models are
loaded lazily and reused within a process. Pair inputs are truncated to
512 model tokens, with inference batches of eight. Scores are used for
ordering, not as probabilities that an answer is supported.

The runner writes `evaluation/results/week5-dev.json` and `.md`, including:

- Dense, BM25, hybrid, and reranked metrics, overall and by question type.
- Complete fused candidates and reranked ordering for manual inspection.
- Candidate-union and rerank-pool evidence coverage, plus evidence pages
  gained or lost by reranking.
- Failure stages: page absent from corpus, absent from candidate union,
  removed by fusion cutoff, or present in the rerank pool but below top five.
- Median query latency after model/corpus loading and one warm-up query.
  Hybrid and v3 timings sum sequential stage durations. These single-run
  timings are exploratory, not a throughput benchmark.

Failure stages identify where evidence disappears. They do not automatically
establish semantic causes such as a parsing error or wrong-year confusion.
Inspect the saved passage text and reference evidence before drawing those
conclusions. Page-level hits still require a supporting-text check.

After freezing the model, candidate budget, query policy, and cutoff on dev:

```powershell
python -m evaluation.week5 --split held_out --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2
python -m pytest -q
```

The held-out run writes separate `week5-held_out.json` and `.md` files.
Do not tune settings using its results. Retain dataset/corpus fingerprints
and model/package metadata with every comparison. The JSON also records the
actual embedding model constant used by the loader, rather than assuming
the environment setting changed that loader.

The completed experiment is documented in
[the final ablation and failure analysis](evaluation/results/week5-final-ablation.md).
V3 improved dev evidence recall@5 to 57.35%, but reached only 18.75% on
held-out, where BM25 reached 56.25%. V3 is therefore opt-in, not the default.
For exact model reproduction, append
`--reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a`
to the Week 5 commands. The frozen configuration and both split reports are
saved under `evaluation/results/`; future tuning needs a fresh held-out set.

## Retrieval repair and benchmark v2

PDF extraction now disables artificial intra-word spaces and repairs verified
TCS embedded-font mappings for digits, currency, percentages, punctuation,
and operators. The mapping is character-based and never substitutes values
from evaluation answers. Indexed payloads include an ingestion version so a
parser change cannot be mistaken for an unchanged index.

The CLI infers exact company and fiscal-year constraints from explicit query
text. Multi-company and multi-year comparisons use multi-value filters. The
full names `Infosys Limited` and `Tata Consultancy Services` map to their
canonical company values, and previous-fiscal-year follow-ups resolve from
supplied history. The cross-encoder receives company, fiscal year, and document
ID alongside each passage so similar figures from different reports are
distinguishable. Its final ordering blends the hybrid and cross-encoder ranks
with a small exact content-bigram signal. This preserves precise phrases such
as “total contract value” when the neural score alone ranks broader passages
too highly.

Benchmark v2 is described in
[evaluation/benchmark_v2.md](evaluation/benchmark_v2.md). It adds grouped
alternative evidence and a more balanced 40-question development / 20-question
challenge split while preserving the original Week 4/5 files and scores. The
development split has 100% evidence recall@5. The configuration was frozen at
tag `retrieval-v2-pre-challenge`, then run once on the challenge split: the
final ensemble reached 88.24% evidence recall@5 and 68.14% MRR@5, compared
with BM25 at 67.65% and 42.65%. No tuning or post-hoc relabelling was performed
after the challenge run. See
[the final development analysis](evaluation/results/benchmark-v2-final-development-analysis.md)
and [challenge analysis](evaluation/results/benchmark-v2-challenge-analysis.md).

## Week 6: single-turn grounded generation

`answer_question()` runs the frozen benchmark-v2 retrieval pipeline, sends the
top five chunks to Groq, validates Groq's strict JSON response with Pydantic,
and returns trusted citation metadata from the retrieved chunks. The model
only chooses evidence IDs and verbatim quotes; it cannot invent document IDs
or page numbers. Unknown evidence IDs, duplicate citations, and quotes absent
from the cited chunk cause an abstention.

The development generation run found that the model emits binary evidence-
sufficiency scores (`0.0` or `1.0`), so the default
`RAG_MIN_EVIDENCE_SUFFICIENCY=1.0` accepts only a fully sufficient assessment.
This is a semantic gate, not a calibrated correctness probability. Empty
retrieval, a model `answerable=false` decision, a score below the gate, or
invalid citations returns `I don't know based on the available annual reports.`

Copy `.env.example` to `.env`, set `GROQ_API_KEY`, keep the Qdrant container
running, then execute one turn:

```powershell
python -m app.generation "What revenue did TCS report in FY2024-25?"
```

The API endpoint uses Groq Chat Completions with strict JSON Schema output on
`openai/gpt-oss-20b`. Offline tests use an injected retriever and generator,
so citation and abstention behavior can be verified without an API key.

Run or resume the generation-development evaluation with:

```powershell
python -m evaluation.generation --split development
```

It saves each Groq output once, creates an automatic threshold and abstention
report, and writes a CSV for second-person review of answer correctness,
citation entailment, and citation completeness. Threshold comparisons reuse
the saved outputs and do not call Groq again.
