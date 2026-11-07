"""Run with: python -m streamlit run ui/streamlit_app.py"""

import os
from uuid import uuid4

import streamlit as st

from ui.client import APIClient, APIError, Citation, Completed, Turn


def view_citation(citation: Citation, client: APIClient, selection: str) -> None:
    """Render original PDF pages via the API, including on remote UI hosts."""
    try:
        report = client.report(citation.doc_id)
        st.subheader(report["title"])
        label = f"{citation.evidence_id} · Cited PDF page {citation.pdf_page}"
        if citation.printed_page is not None:
            label += f" · Printed page {citation.printed_page}"
        st.caption(label)
        st.text(citation.quote)
        if not 1 <= citation.pdf_page <= report["pdf_pages"]:
            st.error("The cited page is outside this report. The citation cannot be opened.")
            return
        page = st.number_input(
            "PDF page", min_value=1, max_value=report["pdf_pages"],
            value=citation.pdf_page, step=1, key=f"page_{selection}",
        )
        st.caption(f"Page {page} of {report['pdf_pages']} · PDF page numbers may differ from printed page numbers.")
        st.image(client.report_page(citation.doc_id, page), use_container_width=True)
        if st.button("Prepare report download", key=f"prepare_{selection}"):
            st.download_button(
                "Download original PDF", client.report_pdf(citation.doc_id),
                file_name=f"{citation.doc_id}.pdf", mime="application/pdf",
                key=f"download_{selection}",
            )
    except APIError as error:
        st.error(f"Could not open the source report. {error}")


def switch_session(session_id: str) -> None:
    st.session_state.session_id = session_id
    st.session_state.entries = []
    st.session_state.history_loaded = False
    st.session_state.pop("selected_citation", None)


def render_turn(turn: Turn, client: APIClient) -> None:
    with st.chat_message("user"):
        st.write(turn.original_question)
    with st.chat_message("assistant"):
        if turn.result.abstained:
            st.info(turn.result.answer)
            reasons = {
                "invalid_citations": "An answer was withheld because its citations could not be validated. This does not mean the information is absent from the reports.",
                "no_retrieval_results": "No matching report passages were retrieved.",
                "model_says_unanswerable": "The model could not answer from the retrieved passages.",
                "below_evidence_threshold": "The retrieved evidence did not meet the required sufficiency threshold.",
            }
            if turn.result.abstention_reason in reasons:
                st.caption(reasons[turn.result.abstention_reason])
            if turn.result.abstention_reason:
                with st.expander("Why was this answer withheld?"):
                    st.text(f"Reason: {turn.result.abstention_reason}")
                    if turn.result.retrieved_chunks is not None:
                        st.text(f"Retrieved passages: {turn.result.retrieved_chunks}")
                    if turn.result.citation_validation_error:
                        st.text(turn.result.citation_validation_error)
                    elif turn.result.abstention_reason == "invalid_citations":
                        st.text("This saved turn has no detailed validation error. New requests record the exact failure.")
        else:
            st.caption("ANSWER")
            answer, separator, calculation = turn.result.answer.partition("\n\n**Calculated:** ")
            st.write(answer)
            if separator:
                with st.container(border=True):
                    st.markdown("**Calculated comparison**")
                    st.write(calculation)
        if turn.result.citations:
            st.caption("SOURCE DOCUMENTS · Select a citation to inspect the original page")
        for index, citation in enumerate(turn.result.citations):
            selection = f"{turn.session_id}_{turn.id}_{index}"
            if st.button(
                f"[{citation.evidence_id}] {citation.company} · {citation.fiscal_year} · PDF page {citation.pdf_page}",
                key=f"citation_{selection}", disabled=not citation.doc_id,
                help="Open the cited PDF page" if citation.doc_id else "This saved citation has no document ID.",
            ):
                # Each fresh click must start at the citation, even after browsing other pages.
                st.session_state.pop(f"page_{selection}", None)
                st.session_state.selected_citation = (citation, selection)
            selected = st.session_state.get("selected_citation")
            if selected and selected[1] == selection:
                with st.container(border=True):
                    if st.button("Close report", key=f"close_{selection}"):
                        st.session_state.pop("selected_citation", None)
                        st.rerun()
                    view_citation(citation, client, selection)
        if turn.result.question != turn.original_question:
            with st.expander("Question used for search"):
                st.write(turn.result.question)
        if turn.result.citations:
            with st.expander(f"Sources ({len(turn.result.citations)})"):
                for citation in turn.result.citations:
                    st.write(f"{citation.evidence_id} · {citation.company} · {citation.fiscal_year}")
                    page = f"PDF page {citation.pdf_page}"
                    if citation.printed_page is not None:
                        page += f" · Printed page {citation.printed_page}"
                    st.caption(page)
                    st.text(citation.quote)


def main() -> None:
    st.set_page_config(page_title="Annual Report Q&A", page_icon="📊", layout="centered")
    if "session_id" not in st.session_state:
        switch_session(str(uuid4()))
    client = APIClient(os.getenv("RAG_API_URL", "http://127.0.0.1:8000"))

    with st.sidebar:
        st.header("Conversations")
        if st.button("New conversation", type="primary", use_container_width=True):
            switch_session(str(uuid4()))
            st.rerun()
        with st.expander("Save or resume a conversation"):
            st.caption("Current session ID — save this to resume later")
            st.code(st.session_state.session_id, language=None)
            with st.form("resume_session"):
                session_id = st.text_input("Resume session ID", max_chars=128)
                if st.form_submit_button("Resume conversation"):
                    if session_id.strip():
                        switch_session(session_id)
                        st.rerun()
                    else:
                        st.error("Enter a session ID.")
        if st.button("Refresh history", use_container_width=True):
            st.session_state.history_loaded = False
            st.rerun()
        st.caption("Your completed conversations are saved locally. New conversations keep earlier history intact.")
        st.divider()
        st.markdown("**Report library**")
        st.markdown("Infosys · TCS · Wipro")
        st.caption("FY2023–24 and FY2024–25 · 6 annual reports")
        st.caption("Answers use the indexed reports. Missing evidence is reported rather than guessed.")

    st.caption("FINANCIAL RESEARCH · SOURCE-BACKED ANSWERS")
    st.title("Annual Report Q&A")
    st.caption("Ask about Infosys, TCS, and Wipro annual reports for FY2023–24 and FY2024–25, then explore with follow-up questions.")
    if not st.session_state.history_loaded:
        try:
            with st.spinner("Loading conversation…"):
                st.session_state.entries = [
                    {"turn": turn} for turn in client.history(st.session_state.session_id)
                ]
            st.session_state.history_loaded = True
        except APIError as error:
            st.error(str(error))
            st.info("Start the API, then select Refresh history.")
            st.stop()

    for entry in st.session_state.entries:
        if "turn" in entry:
            render_turn(entry["turn"], client)
        else:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                if entry["error"]:
                    st.error(entry["message"])
                else:
                    st.info(entry["message"])

    starter_question = None
    if not st.session_state.entries:
        st.divider()
        st.subheader("Start with a question")
        st.caption("Explore a reported figure, compare companies, or follow a topic across years.")
        examples = [
            ("Find a figure", "What revenue did TCS report in FY2024-25?"),
            ("Compare companies", "Compare TCS and Infosys revenue in FY2024-25."),
            ("Explore the workforce", "How many employees did Wipro report in FY2024-25?"),
        ]
        for column, (label, example) in zip(st.columns(3), examples):
            with column, st.container(border=True):
                st.markdown(f"**{label}**")
                st.caption(example)
                if st.button("Ask this question", key=f"starter_{label}", use_container_width=True):
                    starter_question = example
        st.caption("Then try a follow-up like “And Infosys?” in the same conversation. Open any citation to check the source.")
    question = st.chat_input("Ask a question about the annual reports", max_chars=4000)
    question = question or starter_question
    if question:
        if not question.strip():
            st.warning("Enter a nonblank question.")
            st.stop()
        with st.spinner("Reviewing the annual reports…"):
            try:
                result = client.chat(st.session_state.session_id, question)
                if isinstance(result, Completed):
                    st.session_state.entries.append({"turn": result.turn})
                else:
                    st.session_state.entries.append({"question": question, "message": result.message, "error": False})
            except APIError as error:
                st.session_state.entries.append({"question": question, "message": str(error), "error": True})
        st.rerun()


if __name__ == "__main__":
    main()
