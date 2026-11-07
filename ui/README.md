# Streamlit interface

This UI calls FastAPI over HTTP. It never imports backend modules, loads models,
opens SQLite, or calls Groq or Qdrant directly.

From the project root, install dependencies with `python -m pip install -e ".[dev]"`.
For the local demo, `./start_all.ps1` starts the services in the background and
reuses healthy running instances. Open [the chat interface](http://127.0.0.1:8501).
The corpus covers Infosys, TCS, and Wipro for FY2023–24 and FY2024–25.

For separate terminals instead, use the following commands.
Run the API in one terminal:

```powershell
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Run the UI in another terminal:

```powershell
python -m streamlit run ui/streamlit_app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501`. The UI defaults to `http://127.0.0.1:8000` for
FastAPI. To override it, set `$env:RAG_API_URL = "http://127.0.0.1:8001"`
before starting Streamlit (the UI does not load `.env`).

- Each browser session starts with a new random conversation ID. Save the ID
  shown in the sidebar to resume after a browser reload or reconnect.
- **New conversation** creates a new ID without deleting saved history.
- **Resume conversation** loads the chosen session's latest 100 saved turns.
- **Refresh history** reloads those turns from the API. Clarification messages
  and failed questions exist only in the current browser session and disappear
  on refresh; they are not completed turns in SQLite.
- Answers show source quotes, company, fiscal year, and PDF/printed page numbers.
  Click a citation button beneath an answer to open its original PDF page inside
  the conversation. Use **PDF page** to browse neighboring pages and **Prepare
  report download** to download the complete original PDF. **Close report**
  dismisses the viewer. The cited quote stays visible while browsing.
  PDF pages are one-based and may differ from the report's printed page labels.
  Reports are resolved by document ID through `data/manifest.json`, using PDFs
  in `data/raw` on the API server. Missing reports display an error; older saved
  citations without a document ID remain readable but cannot open the viewer.
  Rewritten questions appear under **Question used for search**.
- Clarification requests and abstentions are displayed without inventing an answer.
- Requests are never retried automatically. After a timeout, refresh history
  before resending because the API may still complete and save the original request.

Both servers are intended for trusted local use. The API does not authenticate
session ownership. The UI process needs only API access; credentials and corpus
configuration belong to the FastAPI process.

Run `python -m pytest tests/test_ui.py -q` for mocked HTTP contract tests and
Streamlit AppTest checks of chat submission, reruns, session switching, history,
clarification, and failure handling.

