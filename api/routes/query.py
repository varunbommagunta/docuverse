"""POST /query — ask a question and receive a cited answer.

Delegates to RAGOrchestrator.answer() and maps the domain Answer object to the
HTTP-layer QueryResponse, including fully serialised citation details.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_orchestrator
from api.schemas import CitationDetail, QueryRequest, QueryResponse
from src.orchestrator import RAGOrchestrator
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.exceptions import GenerationError, RetrievalError

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse, tags=["rag"])
async def query_documents(
    body: QueryRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator),
) -> QueryResponse:
    """Ask a question and receive a cited answer grounded in ingested documents.

    Returns 503 if no documents have been ingested yet.
    Returns 500 on unexpected retrieval or generation errors.
    """
    logger.info("Query request received", query_length=len(body.query))

    # Guard: need at least some documents in the index
    try:
        vector_store: ChromaVectorStore = orchestrator._retriever._vector_store  # type: ignore[attr-defined]
        if vector_store.count() == 0:
            raise HTTPException(
                status_code=503,
                detail="No documents have been ingested yet. Please upload a PDF first.",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # If we can't check, proceed and let retrieval fail naturally

    history = None
    if body.history:
        history = [{"role": m.role, "content": m.content} for m in body.history]

    try:
        answer = orchestrator.answer(body.query, history=history)
    except RetrievalError as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}") from exc
    except GenerationError as exc:
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected query error", error=str(exc))
        raise HTTPException(status_code=500, detail="Internal query error.") from exc

    citation_details = [
        CitationDetail(
            chunk_index=idx,
            chunk_id=rc.chunk.id,
            text=rc.chunk.text,
            score=rc.score,
            metadata=rc.chunk.metadata,
        )
        for idx, rc in enumerate(answer.retrieved_chunks)
    ]

    return QueryResponse(
        answer=answer.text,
        citations=answer.citations,
        retrieved_chunks=citation_details,
    )
