"""OpenAI embedding implementation.

OpenAIEmbedder is the V1 concrete implementation of the Embedder Protocol. It
calls the OpenAI Embeddings API in batches and wraps every call with tenacity
retry logic to handle transient rate-limit and API errors gracefully.
"""

import structlog
from openai import APIError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import get_settings
from src.utils.exceptions import RetrievalError

logger = structlog.get_logger(__name__)


class OpenAIEmbedder:
    """Converts text into dense float vectors via the OpenAI Embeddings API.

    Implements the Embedder Protocol. Uses batch mode to amortise API latency
    when embedding many chunks simultaneously during ingestion.
    """

    def __init__(self, model: str | None = None) -> None:
        """Initialise the embedder.

        Args:
            model: OpenAI embedding model name. Defaults to the value in Settings.
        """
        settings = get_settings()
        self._model = model or settings.embedding_model
        self._client = OpenAI(api_key=settings.openai_api_key)
        logger.info("OpenAIEmbedder initialised", model=self._model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into float vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors in the same order as `texts`.

        Raises:
            RetrievalError: If the API call fails after retries.
        """
        if not texts:
            return []
        try:
            return self._embed_with_retry(texts)
        except (RateLimitError, APIError) as exc:
            raise RetrievalError(f"OpenAI embedding API failed: {exc}") from exc

    def embed_single(self, text: str) -> list[float]:
        """Convenience method: embed a single string.

        Used by DenseRetriever to embed the user query.

        Args:
            text: The string to embed.

        Returns:
            A single float vector.

        Raises:
            RetrievalError: If the API call fails after retries.
        """
        return self.embed([text])[0]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError)),
        reraise=True,
    )
    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Internal method with tenacity retry wrapping the raw API call."""
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
