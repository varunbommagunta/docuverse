"""POST /query — ask a question and receive a cited answer."""

import structlog
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_orchestrator
from api.schemas import (
    ArticleFilterDebug,
    ChunkDebug,
    CitationDetail,
    QueryDebug,
    QueryRequest,
    QueryResponse,
    RerankerDebug,
)
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
    """Ask a question and receive a cited answer grounded in ingested documents."""
    logger.info("Query request received", query_length=len(body.query))

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
        pass

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

    debug: QueryDebug | None = None
    if answer.debug:
        d = answer.debug
        af = d.get("article_filter", {})
        rr = d.get("reranker")
        debug = QueryDebug(
            original_query=d.get("original_query", body.query),
            rewritten_query=d.get("rewritten_query", body.query),
            article_filter=ArticleFilterDebug(
                matched=af.get("matched", False),
                article_id=af.get("article_id"),
                pinned_count=af.get("pinned_count", 0),
            ),
            retrieval_strategy=d.get("retrieval_strategy", ""),
            chunks=[
                ChunkDebug(
                    id=c.get("id", ""),
                    score=c.get("score", 0.0),
                    pinned=c.get("pinned", False),
                    source=c.get("source", ""),
                    article_id=c.get("article_id"),
                    section_title=c.get("section_title"),
                    preview=c.get("preview", ""),
                )
                for c in d.get("chunks", [])
            ],
            reranker=RerankerDebug(**rr) if rr else None,
        )

    return QueryResponse(
        answer=answer.text,
        citations=answer.citations,
        retrieved_chunks=citation_details,
        rewritten_query=answer.rewritten_query,
        debug=debug,
    )
