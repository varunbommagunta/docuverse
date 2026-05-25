"""Corpus management routes."""

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


@router.delete("")
async def clear_corpus(request: Request, session_id: str | None = None):
    """Delete all chunks from the vector store."""
    try:
        pipeline = request.app.state.pipeline
        collection = pipeline._vector_store._collection

        all_ids = collection.get(include=[])["ids"]
        if all_ids:
            collection.delete(ids=all_ids)

        return {"deleted": len(all_ids), "message": "Corpus cleared."}
    except Exception as exc:
        return {"deleted": 0, "message": f"Error: {exc}"}
