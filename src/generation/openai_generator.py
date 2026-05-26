"""OpenAI chat completion generator with inline citation parsing.

OpenAIGenerator is the V1 concrete implementation of the Generator Protocol.
It formats retrieved chunks into a numbered context block, calls GPT with the
citation system prompt, and parses [chunk_N] references from the response into
the Answer.citations list.
"""

import re

import structlog
from openai import APIError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import get_settings
from src.generation.prompts import CITATION_SYSTEM_PROMPT, CITATION_USER_PROMPT_TEMPLATE
from src.utils.exceptions import GenerationError
from src.utils.models import Answer, RetrievedChunk

logger = structlog.get_logger(__name__)

_CANNOT_ANSWER = "I cannot answer this from the provided documents."


class OpenAIGenerator:
    """Generates cited answers using the OpenAI Chat Completions API.

    Formats retrieved chunks as a numbered context, calls GPT with strict
    citation instructions, and returns a structured Answer with parsed citation
    indices.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        """Initialise the generator.

        Args:
            model: Chat model name. Defaults to Settings.openai_model.
            temperature: Sampling temperature. 0.0 for deterministic, grounded answers.
            max_tokens: Maximum tokens in the generated response.
        """
        settings = get_settings()
        self._model = model or settings.openai_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = OpenAI(api_key=settings.openai_api_key)
        logger.info("OpenAIGenerator initialised", model=self._model, temperature=temperature)

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> Answer:
        """Generate a cited answer given a query and retrieved context chunks.

        Args:
            query: The user's natural-language question.
            chunks: Retrieved chunks ordered by descending relevance.

        Returns:
            Answer with generated text, citation indices, and the input chunks.

        Raises:
            GenerationError: If the API call fails after retries.
        """
        context = self._format_context(chunks)
        user_message = CITATION_USER_PROMPT_TEMPLATE.format(context=context, query=query)

        try:
            text = self._complete_with_retry(user_message)
        except (RateLimitError, APIError) as exc:
            raise GenerationError(f"OpenAI generation API failed: {exc}") from exc

        citations = self._parse_citations(text)
        logger.info(
            "Generation complete",
            answer_length=len(text),
            citation_count=len(citations),
            citations=citations,
        )

        return Answer(text=text, citations=citations, retrieved_chunks=chunks)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove constitutional amendment markers and footnote artifacts from chunk text."""
        # Transition markers ]N[ → space
        text = re.sub(r"\]\d+\[", " ", text)
        # Opening amendment markers N[ → nothing (keep inserted content)
        text = re.sub(r"\d+\[", "", text)
        # Remaining closing/opening brackets
        text = re.sub(r"[\[\]]", "", text)
        # Footnote separator lines (underscores or dashes)
        text = re.sub(r"[_\-]{10,}", "", text)
        # Footnote label lines: digit + period + amendment keyword
        text = re.sub(
            r"^\d+\.\s+(?:Subs|Ins|Rep|Omitted|Added|Renumbered)\..*$",
            "",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        # Standalone page numbers (e.g. "— 12 —" or just a bare number on its own line)
        text = re.sub(r"^\s*[—\-]?\s*\d+\s*[—\-]?\s*$", "", text, flags=re.MULTILINE)
        # Running headers that repeat (e.g. "THE CONSTITUTION OF INDIA" on every page)
        text = re.sub(r"^THE CONSTITUTION OF INDIA\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
        # Collapse excess blank lines and spaces
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"  +", " ", text)
        return text.strip()

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        """Render chunks as a numbered context block for the prompt."""
        parts = []
        for idx, rc in enumerate(chunks):
            cleaned = OpenAIGenerator._clean_text(rc.chunk.text)
            parts.append(f"[chunk_{idx}]:\n{cleaned}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_citations(text: str) -> list[int]:
        """Extract unique [chunk_N] indices from generated text, in order of appearance."""
        seen: set[int] = set()
        citations: list[int] = []
        for match in re.finditer(r"\[chunk_(\d+)\]", text):
            idx = int(match.group(1))
            if idx not in seen:
                seen.add(idx)
                citations.append(idx)
        return citations

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError)),
        reraise=True,
    )
    def _complete_with_retry(self, user_message: str) -> str:
        """Call the Chat Completions API with retry on transient errors."""
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": CITATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return (response.choices[0].message.content or "").strip()
