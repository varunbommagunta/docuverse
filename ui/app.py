"""Streamlit frontend for DocuVerse — Document Q&A.

Phase 1 implementation: full upload + chat interface. Communicates with the
FastAPI backend at API_URL (defaults to http://localhost:8000). Uses
st.session_state to track ingested documents and chat history within a session.
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
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []


def _show_sources(chunks: list[dict], citations: list[int]) -> None:
    """Render expandable source chunks below an answer."""
    if not chunks:
        return
    with st.expander(f"📖 Sources ({len(chunks)} retrieved, {len(citations)} cited)"):
        for chunk in chunks:
            idx = chunk["chunk_index"]
            cited = "✅ Cited" if idx in citations else "— Not cited"
            label = (
                f"[chunk_{idx}] {cited} | "
                f"Score: {chunk['score']:.3f} | "
                f"{chunk['metadata'].get('filename', '')}"
            )
            st.caption(label)
            st.text(chunk["text"][:600] + ("…" if len(chunk["text"]) > 600 else ""))
            st.divider()


# ── Sidebar: upload ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📁 Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None and st.button("Ingest Document", type="primary"):
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
                        f"✅ **{data['filename']}** ingested — "
                        f"{data['chunk_count']} chunks indexed."
                    )
                    st.session_state.ingested_docs.append(data)
                else:
                    detail = response.json().get("detail", "Unknown error")
                    st.error(f"❌ Ingestion failed ({response.status_code}): {detail}")
            except requests.ConnectionError:
                st.error("❌ Cannot connect to the API. Is the backend running?")
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")

    st.divider()

    if st.session_state.ingested_docs:
        st.subheader("Indexed Documents")
        for doc in st.session_state.ingested_docs:
            st.caption(f"📄 {doc['filename']} — {doc['chunk_count']} chunks")
    else:
        st.caption("No documents ingested yet this session.")

# ── Main area: chat ───────────────────────────────────────────────────────────
st.title("DocuVerse — Document Q&A")
st.subheader("Ask questions. Get cited answers from your documents.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            _show_sources(msg["chunks"], msg.get("citations", []))

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    if not st.session_state.ingested_docs:
        st.warning("⚠️ Please upload and ingest a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"), st.spinner("Thinking…"):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": prompt},
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
                elif response.status_code == 503:
                    msg = "⚠️ No documents indexed yet. Please ingest a PDF first."
                    st.warning(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                else:
                    detail = response.json().get("detail", "Unknown error")
                    msg = f"❌ Query failed: {detail}"
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
            except requests.ConnectionError:
                msg = "❌ Cannot connect to the API. Is the backend running?"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
