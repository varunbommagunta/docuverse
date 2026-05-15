"""Streamlit frontend for DocuVerse.

Phase 0 placeholder — renders the app shell with a disabled upload sidebar
and a status message in the main content area. No API calls yet.

Phase 1 will wire the sidebar file uploader to POST /documents.
Phase 3 will wire the chat input to POST /ask and render cited answers.
"""

import streamlit as st

st.set_page_config(
    page_title="DocuVerse",
    page_icon="📚",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📁 Upload Documents")
    st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        disabled=True,
        help="Document upload will be available in Phase 1.",
    )
    st.caption("Upload is disabled — coming in Phase 1.")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("DocuVerse")
st.subheader("Ask questions. Get cited answers from your documents.")

st.info(
    "**Phase 0: Foundation complete.** RAG features coming in Phase 1.\n\n"
    "The API skeleton, Protocol interfaces, and infrastructure are wired up. "
    "Ingest your first PDF in Phase 1, and ask your first question in Phase 3."
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Current Phase", value="0 — Foundation")

with col2:
    st.metric(label="API Status", value="Running", delta="healthy")

with col3:
    st.metric(label="Documents Indexed", value="0", help="Will populate after Phase 1.")
