"""POST /ingest — upload and ingest a PDF document.

Accepts a multipart/form-data upload, validates the file, saves it to a
temporary location, delegates to IngestionPipeline.ingest(), then cleans up.
"""

import os
import tempfile

import structlog
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from api.dependencies import get_pipeline
from api.schemas import IngestResponse
from src.ingestion.pipeline import IngestionPipeline
from src.utils.exceptions import ChunkingError, DocumentParseError

logger = structlog.get_logger(__name__)
router = APIRouter()

_MB = 1024 * 1024


@router.post("/ingest", response_model=IngestResponse, tags=["rag"])
async def ingest_document(
    file: UploadFile,
    pipeline: IngestionPipeline = Depends(get_pipeline),
) -> IngestResponse:
    """Upload a PDF and ingest it into the vector store.

    - Validates file extension (.pdf only) and size (≤ MAX_UPLOAD_SIZE_MB).
    - Delegates parsing, chunking, embedding, and storing to IngestionPipeline.
    - The temporary file is always deleted after processing.
    """
    from config.settings import get_settings
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * _MB

    # Validate extension
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Read and validate size
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_size_mb} MB limit.",
        )

    logger.info("Ingest request received", filename=filename, size_bytes=len(contents))

    # Save to temp file and ingest
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = pipeline.ingest(tmp_path, filename=filename)

    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=f"PDF parse error: {exc}") from exc
    except ChunkingError as exc:
        raise HTTPException(status_code=422, detail=f"Chunking error: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected ingestion error", error=str(exc))
        raise HTTPException(status_code=500, detail="Internal ingestion error.") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return IngestResponse(
        document_id=str(result["document_id"]),
        filename=str(result["filename"]),
        chunk_count=int(result["chunk_count"]),
    )
