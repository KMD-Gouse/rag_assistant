# Annual Report Q&A

A citation-focused retrieval-augmented generation (RAG) system for answering
and comparing questions across Infosys and TCS annual reports. The repository
currently contains the complete, evaluated Weeks 1–6 pipeline: document
ingestion, dense and lexical retrieval, hybrid candidate generation,
cross-encoder reranking, grounded Groq generation, structured output,
citation validation, and abstention.

Week 7 will add FastAPI, Streamlit, SQLite conversation history, and follow-up
rewriting around the frozen Week 6 answer pipeline. Those application features
are intentionally not claimed as implemented yet.

## Current status

- Four annual reports indexed as 4,083 chunks in Qdrant.
- Dense retrieval, BM25, reciprocal-rank fusion (RRF), and cross-encoder
  reranking implemented.
- Company and fiscal-year scope inferred from explicit question text.
- Groq responses validated with a strict Pydantic-compatible JSON schema.
- Citations accepted only when their evidence ID exists and their quote is an
  exact substring of the retrieved chunk.
- Unsupported, insufficient, or citation-invalid answers return a fixed
  `I don't know` response.
- Retrieval and generation development configurations were frozen before
  their one-time challenge evaluations.
- Week 6 generation challenge evaluation is complete and human-reviewed.

## System architecture

```text
PDF annual reports
    ↓
PyMuPDF extraction and verified text repair
    ↓
Metadata-aware chunking
    ↓
BGE embeddings + Qdrant payload index
    ↓
Dense retrieval ─┐
                 ├─ equal-weight RRF candidate pool (top 50)
BM25 retrieval ──┘
    ↓
Cross-encoder + hybrid-rank + exact-phrase rank ensemble
    ↓
Top five evidence chunks
    ↓
Groq strict structured generation
    ↓
Pydantic and citation validation
    ↓
Grounded answer with citations, or abstention
```

The initial Week 4/5 experiments remain under `evaluation/results/` for
traceability. The production Week 6 answer path uses the later frozen
benchmark-v2 retrieval configuration.

## Corpus

| Document ID | Company | Fiscal year | PDF pages |
|---|---|---:|---:|
| `infosys_fy2024` | Infosys | 2023–24 | 352 |
| `infosys_fy2025` | Infosys | 2024–25 | 369 |
| `tcs_fy2024` | TCS | 2023–24 | 340 |
| `tcs_fy2025` | TCS | 2024–25 | 336 |

Source URLs, filenames, SHA-256 hashes, and validation metadata are recorded
in [`data/manifest.json`](data/manifest.json). Raw PDFs and processed chunks
are local data artifacts and are not required in Git history.

## Implemented milestones

### Weeks 1–3: ingestion and vector search

- Manifest-backed document identity and checksums.
- PDF page extraction using PyMuPDF.
- Verified repair of broken TCS embedded-font mappings.
- Prose and table-aware chunks with company, fiscal year, page, section, and
  source metadata.
- BGE-small embeddings stored in Qdrant.
- Index verification and filtered dense search.

### Week 4: retrieval baselines

- Dense semantic retrieval.
- In-memory BM25 over the exact Qdrant text payloads.
- Equal-weight reciprocal-rank fusion.
- Page-level evidence recall, MRR, document recall, and all-evidence-hit
  evaluation.

BM25 uses `k1=1.5`, `b=0.75`, positive IDF, case-folded word tokens, and
commas removed within numbers. RRF uses `1 / (60 + rank)` with ranks starting
at one. Retrieval scores and fusion scores are ordering signals, not
probabilities.

### Week 5: reranking and failure analysis

- `cross-encoder/ms-marco-MiniLM-L-6-v2` with a pinned model revision.
- Top-50 hybrid candidate pool reranked to top five.
- Final rank ensemble combining hybrid rank, cross-encoder rank, and a small
  exact content-bigram signal.
- Ablations, ±50% weight sensitivity, and failure-stage analysis.
- Corrected evidence labels with grouped alternative pages.

### Week 6: grounded answer generation

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

- [`evaluation/benchmark_v2.md`](evaluation/benchmark_v2.md): retrieval
  benchmark design.
- [`evaluation/results/benchmark-v2-final-development-analysis.md`](evaluation/results/benchmark-v2-final-development-analysis.md): label audit,
  ablations, and frozen retrieval decisions.
- [`evaluation/results/benchmark-v2-challenge-analysis.md`](evaluation/results/benchmark-v2-challenge-analysis.md): one-time retrieval challenge.
- [`evaluation/results/benchmark-v2-frozen-config.json`](evaluation/results/benchmark-v2-frozen-config.json): exact retrieval freeze.
- [`evaluation/results/week6-generation-development.md`](evaluation/results/week6-generation-development.md): reviewed generation development result.
- [`evaluation/results/week6-generation-frozen-config.json`](evaluation/results/week6-generation-frozen-config.json): exact generation freeze.
- [`evaluation/results/week6-generation-challenge.md`](evaluation/results/week6-generation-challenge.md): final reviewed generation challenge.
- [`evaluation/results/week6-generation-challenge-provenance.json`](evaluation/results/week6-generation-challenge-provenance.json): challenge hashes and run provenance.

## Requirements

- Python 3.11 or 3.12.
- Docker Desktop or another Docker engine.
- A Groq API key for answer generation.
- Enough local memory to load the embedding and cross-encoder models.

## Installation

From PowerShell:

```powershell
cd C:\Users\infom\OneDrive\Desktop\gouse\rag_assistant

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Copy the environment template and set `GROQ_API_KEY`:

```powershell
Copy-Item .env.example .env
notepad .env
```

Do not commit `.env` or the API key.

## Start Qdrant

```powershell
docker compose up -d
Invoke-RestMethod http://127.0.0.1:6333/collections
```

The indexed collection is named `annual_report_chunks`.

To build the index from the PDFs listed in the manifest:

```powershell
python -m app.ingestion.pipeline --all
```

Reingestion is unnecessary when the existing Docker volume already contains
the verified 4,083-chunk collection.

## Run retrieval directly

```powershell
python -m app.pipeline "How many employees did Infosys report in FY2024-25?" --method dense

python -m app.pipeline "How many employees did Infosys report in FY2024-25?" --method bm25

python -m app.pipeline "How many employees did Infosys report in FY2024-25?" `
  --method hybrid --limit 5 --candidates 50 --rrf-k 60

python -m app.pipeline "How many employees did Infosys report in FY2024-25?" `
  --method reranked --limit 5 --candidates 50 --rerank-candidates 50 `
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a
```

All methods support `--doc-id`, `--company`, and `--fiscal-year` exact-match
filters. The production generation service also applies automatic scope
inference for explicitly named companies and fiscal years.

## Run a grounded answer

```powershell
python -m app.generation "What revenue did TCS report in FY2024-25?"

python -m app.generation "What operating margin did Infosys report in FY2024-25?"

python -m app.generation "How many AI-aware employees did Infosys report in FY2024-25?"
```

The CLI prints structured JSON containing the answer, abstention decision,
reason, evidence-sufficiency value, validated citations, retrieved-chunk
count, threshold, and model.

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

## Tests

Run deterministic tests that do not require external services:

```powershell
python -m pytest -m "not integration" -q
```

With Qdrant running, execute live integration tests:

```powershell
python -m pytest -m "integration" -q
```

The current suite contains 118 offline tests and eight integration tests.
The Qdrant client may emit a compatibility warning because the installed
client is newer than the pinned `qdrant/qdrant:v1.15.3` server; the verified
queries still pass.

## Repository layout

```text
app/
  ingestion/        PDF parsing, repair, chunking, embedding, indexing
  retrieval/        dense, BM25, RRF, scope inference, reranking
  generation/       Groq client, schemas, citation validation, service
  api/              reserved for Week 7 FastAPI implementation
  memory/           reserved for Week 7 SQLite conversation history

data/
  manifest.json     corpus identity, sources, hashes, page counts
  raw/              local PDF files
  processed/        local extracted pages and chunks

evaluation/
  benchmark_v2.jsonl
  generation_challenge_v1.jsonl
  generation.py
  validate_generation_challenge.py
  results/           raw results, review CSVs, reports, freezes, provenance

tests/              deterministic and live integration tests
docker-compose.yml  pinned Qdrant service
```

## Known limitations

- The corpus contains only four annual reports from two companies.
- Development results reflect iterative tuning and should not be presented as
  unseen performance.
- The challenge sets are project regression sets, not statistically
  representative population samples.
- Equal-weight RRF underperforms BM25 at top five; its purpose here is to
  create a diverse high-recall candidate pool for later reranking.
- Comparison and multi-period questions remain the main retrieval weakness.
- Evidence sufficiency is binary and has not been probability-calibrated.
- Exact-quote validation proves that a quote exists in a retrieved chunk; the
  separate human review is still needed to judge whether it entails the
  answer and covers every claim.
- Four answerable generation-challenge questions abstained because proposed
  citations failed validation.

## Week 7 plan

Week 7 will wrap the frozen single-turn answer service in an application and
conversation layer:

```text
Streamlit
    ↓
FastAPI receives question + session_id
    ↓
Load recent SQLite history for that session
    ↓
Rewrite a contextual follow-up into a standalone question
    ↓
Run the frozen Week 6 answer pipeline
    ↓
Store the original question, rewritten question, answer, and citations
    ↓
Return the answer and citations to Streamlit
```

Planned implementation order:

1. SQLite session-history repository.
2. Follow-up rewriting.
3. FastAPI request and response models plus endpoint.
4. Streamlit interface using FastAPI as its backend.
5. Session-isolation, history-window, follow-up, and API tests.

Week 7 application changes will not alter the frozen retrieval and generation
evaluation results. If the underlying prompt, retriever, reranker, citation
policy, or threshold changes, that creates a new pipeline version and requires
a new development freeze and a new untouched challenge set.
