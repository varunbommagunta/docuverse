"""Streamlit frontend for DocuVerse — Document Q&A.

Phase 5 implementation: updated landing page and HF Spaces-compatible API URL.
Communicates with the FastAPI backend at API_URL (set by supervisord in Docker
to http://127.0.0.1:8000; defaults to http://localhost:8000 locally).
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DocuVerse — Document Q&A",
    page_icon="📚",
    layout="wide",
)

# ── Session state initialisation ──────────────────────────────────────────────
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs: list[dict] = []
    # Check if the backend has a pre-loaded corpus (auto-ingested at startup)
    try:
        response = requests.get(f"{API_URL}/corpus/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("is_preloaded"):
                for doc in data.get("documents", []):
                    st.session_state.ingested_docs.append({
                        "filename": doc["filename"],
                        "chunk_count": doc["chunk_count"],
                    })
    except Exception:
        pass
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []


def _show_sources(chunks: list[dict], citations: list[int]) -> None:
    if not chunks:
        return
    with st.expander(f"Sources ({len(chunks)} retrieved, {len(citations)} cited)"):
        for chunk in chunks:
            idx = chunk["chunk_index"]
            cited = "Cited" if idx in citations else "Not cited"
            label = (
                f"[chunk_{idx}] {cited} | "
                f"Score: {chunk['score']:.3f} | "
                f"{chunk['metadata'].get('filename', '')}"
            )
            st.caption(label)
            st.text(chunk["text"][:600] + ("…" if len(chunk["text"]) > 600 else ""))
            st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("DocuVerse")
    st.caption("Hybrid RAG · PDF Q&A · Cited Answers")
    st.divider()

    st.subheader("Upload a PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF (max 6 MB)", type=["pdf"], label_visibility="collapsed"
    )

    if uploaded_file is not None and st.button("Ingest Document", type="primary", use_container_width=True):
        with st.spinner(f"Ingesting {uploaded_file.name}…"):
            try:
                response = requests.post(
                    f"{API_URL}/ingest",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=120,
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(
                        f"**{data['filename']}** ingested — "
                        f"{data['chunk_count']} chunks indexed."
                    )
                    st.session_state.ingested_docs.append(data)
                elif response.status_code == 429:
                    st.error("Daily cost cap reached. Try again tomorrow.")
                else:
                    detail = response.json().get("detail", "Unknown error")
                    st.error(f"Ingestion failed ({response.status_code}): {detail}")
            except requests.ConnectionError:
                st.error("Cannot connect to the API. Is the backend running?")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    st.divider()

    preloaded_filenames = {"constitution_of_india.pdf", "arc_ethics_governance.pdf"}
    has_preloaded = any(
        doc["filename"] in preloaded_filenames for doc in st.session_state.ingested_docs
    )
    if has_preloaded:
        st.success("📚 Indian Constitution + ARC Ethics corpus pre-loaded — ask questions immediately!")

    if st.session_state.ingested_docs:
        st.subheader("Indexed this session")
        for doc in st.session_state.ingested_docs:
            st.caption(f"📄 {doc['filename']} — {doc['chunk_count']} chunks")
    else:
        st.caption("No documents ingested yet this session.")

    st.divider()
    with st.expander("Sample questions to try"):
        st.markdown(
            "- What are the Fundamental Rights guaranteed by the Constitution of India?\n"
            "- What recommendations does the ARC report make on ethics training?\n"
            "- What is the largest planet in the Solar System?"
        )


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Document Q&A")
st.markdown(
    "Upload a PDF in the sidebar, then ask questions below. "
    "Answers are grounded in your documents with inline `[chunk_N]` citations."
)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            _show_sources(msg["chunks"], msg.get("citations", []))

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    if not st.session_state.ingested_docs:
        st.warning("Please upload and ingest a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"), st.spinner("Thinking…"):
            try:
                # Build history from previous messages only (exclude current user message,
                # which was just appended above — sending it in history too duplicates
                # the query and breaks the query rewriter's coreference resolution).
                prior_messages = st.session_state.messages[:-1]  # all but current
                history_to_send = [
                    {"role": m["role"], "content": m["content"]}
                    for m in prior_messages[-6:]  # last 3 turns
                ] if prior_messages else []

                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": prompt, "history": history_to_send if history_to_send else None},
                    timeout=60,
                )
                if response.status_code == 200:
                    data = response.json()
                    answer_text = data["answer"]
                    citations = data["citations"]
                    chunks = data["retrieved_chunks"]

                    st.markdown(answer_text)
                    _show_sources(chunks, citations)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "chunks": chunks,
                        "citations": citations,
                    })
                elif response.status_code == 429:
                    msg = "Daily cost cap reached. Try again tomorrow."
                    st.warning(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                elif response.status_code == 503:
                    msg = "No documents indexed yet. Please ingest a PDF first."
                    st.warning(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                else:
                    detail = response.json().get("detail", "Unknown error")
                    msg = f"Query failed: {detail}"
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
            except requests.ConnectionError:
                msg = "Cannot connect to the API. Is the backend running?"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
