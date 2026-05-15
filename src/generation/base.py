"""Protocol interface for the generation layer.

The generation layer is the final step in the RAG pipeline. It receives a
user query and a list of retrieved chunks, constructs a grounded prompt, calls
an LLM, parses the response, and returns a structured Answer with citations.

Keeping this behind a Protocol enables:
- Swapping LLM providers without changing the orchestrator
- Testing the orchestrator with a stub generator that returns deterministic answers
- Running offline evaluation with a cost-free mock generator
"""

from typing import Protocol

from src.utils.models import Answer, RetrievedChunk


class Generator(Protocol):
    """Generates a cited answer from a query and retrieved context chunks.

    Implementations own the full prompt-engineering lifecycle: system prompt
    construction, chunk formatting, citation instruction injection, response
    parsing, and Answer assembly.

    V1 implementation: OpenAIGenerator — gpt-4o-mini with a structured system
        prompt that instructs the model to cite chunk IDs in [1], [2] format.
        Uses OpenAI function calling or JSON mode for structured output.
    V2 implementation: OpenAIStreamingGenerator — same as V1 but streams tokens
        to the Streamlit UI via Server-Sent Events for a ChatGPT-like UX.
    V3 implementation: AnthropicGenerator — Claude 3.5 Sonnet for tasks where
        long-context retrieval (>100K tokens) is advantageous.
    """

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> Answer:
        """Generate a cited answer given a query and retrieved context.

        Args:
            query: The original natural-language user question.
            context_chunks: Ordered list of retrieved chunks (highest relevance first).
                The implementation decides how many chunks to include in the prompt
                based on context window constraints.

        Returns:
            Answer with the generated text, citation IDs, and the input chunks
            (so the caller can surface them in the UI).

        Raises:
            GenerationError: If the LLM API call fails or returns an unparseable
                response after retries.
        """
        ...
