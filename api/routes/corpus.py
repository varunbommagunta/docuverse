"""GET /corpus/info — return statistics about the pre-loaded corpus.

Used by the Streamlit UI on page load to detect whether ChromaDB already has
data (auto-ingested at container startup) so the upload-first check can be
bypassed when the demo corpus is available.
"""

from collections import Counter

from fastapi import APIRouter, Request

router = APIRouter(prefix="/corpus", tags=["corpus"])


@router.get("/info")
async def get_corpus_info(request: Request):
    """Return chunk count and source filenames for the current corpus."""
    try:
        pipeline = request.app.state.pipeline
        collection = pipeline._vector_store._collection

        chunk_count = collection.count()

        if chunk_count == 0:
            return {"chunk_count": 0, "documents": [], "is_preloaded": False}

        # Sample up to 100 chunks to infer per-document counts (lightweight)
        sample = collection.get(limit=100, include=["metadatas"])
        filenames = [m.get("filename", "unknown") for m in sample.get("metadatas", [])]
        filename_counts = dict(Counter(filenames))

        documents = [
            {"filename": fname, "chunk_count": count}
            for fname, count in filename_counts.items()
        ]

        return {"chunk_count": chunk_count, "documents": documents, "is_preloaded": True}
    except Exception:
        return {"chunk_count": 0, "documents": [], "is_preloaded": False}
