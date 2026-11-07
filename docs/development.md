# Developer reference

[Back to the project README](../README.md).

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

## SQLite session history

`app.memory` exports the `HistoryRepository` protocol, `HistoryTurn` model,
and `SQLiteHistoryRepository`. Construction creates the database and parent
directories if needed. The default file is `data/history.sqlite3` under the
project root; override it with `HISTORY_DB_PATH` or an explicit constructor
path. Relative overrides are resolved against the current working directory.
No additional database dependency or external service is required.

```python
from app.generation import answer_question
from app.memory import HistoryRepository, SQLiteHistoryRepository

history: HistoryRepository = SQLiteHistoryRepository()
session_id = "example-session"  # The UI supplies the session identity.
original_question = "What revenue did TCS report in FY2024-25?"
result = answer_question(original_question)
saved = history.append_turn(session_id, original_question, result)
recent = history.get_history(session_id, limit=10)
# history.clear_session(session_id)  # Returns the number of removed turns.
```

Each append atomically saves a completed question/answer turn, including the
full `AnswerResult` (citations, abstention details, and generation metadata).
`original_question` preserves the user's input; `result.question` and the
`rewritten_question` property contain the question passed to generation.
When rewriting is skipped, these questions are the same. Timestamps
are UTC; integer IDs define insertion order even when timestamps tie.

History reads return the latest `limit` **turns**, oldest first, with a default
of 10. Unknown sessions return an empty list. Session IDs are nonblank,
case-sensitive strings used verbatim; clearing one session leaves others
intact. The caller is responsible for session ownership and authorization.
Each operation opens and closes its own connection, allowing request threads
and repository instances to share a database. SQLite waits up to five seconds
for write locks, then propagates the error. Appends are not deduplicated;
retry handling belongs to the caller; the API rejects overlapping requests
within one session in its single worker. No history is stored automatically by the existing generation CLI.

Run repository tests with `python -m pytest tests/test_memory.py -q`.

## Follow-up rewriting

`FollowUpRewriter` reads the most recent five completed turns of the supplied
session (configurable with `history_limit`) and asks Groq for a standalone
question. For example, after asking about TCS revenue in FY2024-25,
“And Infosys?” should become “What was Infosys revenue in FY2024-25?”
The prompt preserves explicit changes of company, metric, and fiscal year and
instructs the model to leave unrelated standalone questions unchanged.

```python
from app.generation import answer_question
from app.memory import FollowUpRewriter, SQLiteHistoryRepository

history = SQLiteHistoryRepository()
rewriter = FollowUpRewriter(history, history_limit=5)
session_id = "example-session"
question = "And Infosys?"
rewrite = rewriter.rewrite(session_id, question)
if rewrite.needs_clarification:
    print("Please specify the company, fiscal year, or subject you mean.")
else:
    result = answer_question(rewrite.standalone_question)
    history.append_turn(session_id, rewrite.original_question, result)
```

The rewriter returns `RewriteResult` with the original and standalone questions,
`rewritten`, `needs_clarification`, and the number of history turns used. It
does not generate an answer or save a turn. With no history, the original
question passes through without calling Groq; this does not certify that a
first question is unambiguous. With history, the model can flag unresolved
references; flagged results preserve the original question for clarification.
Past abstention answers are omitted, while their question context is retained.

Rewriting uses the existing `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_API_URL`, and
`GROQ_TIMEOUT_SECONDS` settings through a separate structured-output client.
Provider errors (including rate limits) and invalid outputs raise `RewriteError`;
callers should surface a retryable failure instead of silently generating from
an unresolved question. The API checks the clarification flag before generation.
The generation CLI remains single-turn.

Run `python -m pytest tests/test_rewriter.py -q` for offline contract, isolation,
and provider-error tests. These use simulated model responses; live rewriting
quality still needs evaluation on representative multi-turn conversations.

## Run the FastAPI server

The application answer path now uses **source-excerpts-v1**. Retrieved passages
are divided into overlapping, contiguous excerpts. The model selects excerpt
IDs, and the server resolves them to exact source text. It never depends on the
model transcribing quotes, currency symbols, or tables correctly. Unknown IDs,
IDs from another evidence item, and free-text selections are rejected. If a
generated citation selection fails validation, one regeneration attempt uses
the same evidence. Answerability, evidence sufficiency, and citation checks
still apply; unsupported answers abstain and no figures are hardcoded.

Explicit comparisons use **balanced-v1** retrieval. A question naming multiple
companies or fiscal years is split into one scoped search per company/year
side. Each side gets the configured `top_k`; results are interleaved and capped
at 10 evidence passages so a globally high-ranked side cannot crowd out the
other one. The generation prompt requires evidence for every requested side,
aligned units and metric definitions, and only calculations supported by the
retrieved values. If definitions are incompatible, the answer abstains. When only one side is supported, the application can return that side and explicitly identify the missing side; it does not calculate a two-sided comparison.
For generic company-revenue comparisons, segment breakdowns and passages that
mix standalone with consolidated totals are excluded; both companies must be
supported by unambiguous company-wide or consolidated revenue evidence.

This changes application generation behavior and may add one model call per
failed answer. The original `evaluation.generation` runner explicitly disables
excerpt selection and recovery (`citation_mode="verbatim", citation_retries=0`)
to preserve its single-pass protocol. The runner also sets
`balanced_comparisons=False`, so comparison retrieval stays frozen for
reproducible historical runs. Week 6 evaluation scores do not measure the new
answer path; it needs its own evaluation before claiming an accuracy
improvement.

Install the updated dependencies and start one local worker:

```powershell
python -m pip install -e ".[dev]"
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Qdrant must be running with the indexed corpus, and `GROQ_API_KEY` must be set
for live generation and rewriting. Retrieval models load on the first answer
request. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

```powershell
$body = @{
    session_id = "demo-session"
    question = "What revenue did TCS report in FY2024-25?"
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -ContentType "application/json" -Body $body

$body = @{ session_id = "demo-session"; question = "And Infosys?" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -ContentType "application/json" -Body $body
Invoke-RestMethod "http://127.0.0.1:8000/sessions/demo-session/history?limit=10"
```

`POST /chat` requires a nonblank `session_id` (up to 128 characters) and
`question` (up to 4000 characters). The client supplies a stable session ID
for each conversation. The response has `session_id`, rewrite metadata, and:

- `status: "completed"` with `turn`, containing the stored original question,
  timestamp, and `result` (answer, citations, abstention, and model metadata).
  Abstentions are valid completed turns and are stored.
- `status: "needs_clarification"` with a `message`. The API neither generates
  an answer nor writes history in this case.

`GET /sessions/{session_id}/history` returns recent turns oldest first; `limit`
defaults to 10 and accepts 1–100. Unknown sessions return `[]`. `GET /health`
reports process liveness and `generation_pipeline: "source-excerpts-v1"`, not
readiness of Groq, Qdrant, or models. Check this version after restarting the API
to confirm that it has loaded the updated generation code; comparison-capable
builds also report `comparison_retrieval: "balanced-v1"`.

Invalid requests return 422; overlapping requests in the same session return
409; provider rate limits return 429; other model/rewrite failures return 502; SQLite failures return 503; unexpected
pipeline failures return a sanitized 500. Failed or clarification requests are
not saved as turns. Success is returned only after storage commits. A lost
response after commit can still lead to duplicates on retry: no idempotency
key is implemented yet.

This is a local, trusted-client API without authentication. Session IDs scope
history but do not prove ownership. Keep the server bound to loopback; a shared
deployment requires authentication and session ownership checks. Use one worker:
the in-process session guard covers rewriting through persistence, and a shared
pipeline lock serializes local model initialization and inference. Multiple
workers or servers would require cross-process conversation coordination.

`create_app()` accepts an injected history repository, rewrite generator, and
answer callable for deterministic tests. Run `python -m pytest tests/test_api.py -q`
to exercise the HTTP flow without external services.

## Run the Streamlit interface

Citation-validation failures now include an optional `citation_validation_error`
in the answer result, saved with the turn and logged in the API terminal. The UI
shows the reason under **Why was this answer withheld?** Older saved turns remain
readable but cannot recover a validation error that was not recorded. These
diagnostics do not change retrieval, prompts, citation acceptance, or abstention
decisions, and do not log report excerpts or generated answers.

With FastAPI running, open a second terminal from the project root:

```powershell
python -m streamlit run ui/streamlit_app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501` to ask questions and follow-ups, inspect source
quotes and page numbers, and start or resume conversations. Save the session
ID shown in the sidebar to resume later. The UI calls FastAPI at
`http://127.0.0.1:8000`; override this with the `RAG_API_URL` process environment
variable. It does not call the RAG pipeline or SQLite directly.

See [`ui/README.md`](../ui/README.md) for session behavior, error handling, and
UI tests. After a timeout, refresh history before resending a question to avoid
duplicating a turn that the API may still save.
