"""LLM-based query rewriter for conversational follow-up handling.

Uses gpt-4o-mini with a focused prompt to resolve pronouns and implicit
references in user queries against conversation history. This is critical
for retrieval quality on follow-up turns because retrievers are stateless —
they cannot resolve "that" or "it" without context.
"""

import structlog
from openai import OpenAI

from src.utils.exceptions import RetrievalError

logger = structlog.get_logger(__name__)

REWRITER_SYSTEM_PROMPT = """You are a query rewriter for a document Q&A system. Given a conversation history and a follow-up question, rewrite the question to be standalone — incorporating any context from the history that the question refers to implicitly through pronouns ("that", "it", "this") or references ("explain more", "what about").

Rules:
1. Preserve all specific identifiers (article numbers, section names, dates, etc.) from the history
2. If the question is already standalone (no pronouns or implicit references), return it unchanged
3. Do not add information that isn't grounded in the history
4. Keep the rewritten query concise — one sentence
5. Output ONLY the rewritten query, no explanation
6. Do NOT add proper noun qualifiers (such as "of the Indian Constitution", "of India", etc.) that are not present in the original question — only substitute pronouns and implicit references

Examples:
History:
User: What does Article 21 protect?
Assistant: Article 21 protects the right to life and personal liberty.
Question: tell me more about that
Rewritten: tell me more about Article 21 and the right to life and personal liberty
Question: are there exceptions?
Rewritten: are there exceptions to Article 21
Question: what about Article 19?
Rewritten: what does Article 19 cover
Question: who wrote it?
Rewritten: who wrote Article 21"""


class OpenAIQueryRewriter:
    """Rewrites queries using gpt-4o-mini for conversational coreference resolution."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_history_turns: int = 3) -> None:
        self._client = OpenAI(api_key=api_key.strip())
        self._model = model
        self._max_history_turns = max_history_turns
        logger.info(
            "OpenAIQueryRewriter initialised",
            model=model,
            max_history_turns=max_history_turns,
        )

    def rewrite(self, query: str, history: list[dict] | None = None) -> str:
        # No history → no rewriting needed
        if not history:
            return query

        # Trim history to last N turns (each turn = user message + assistant message)
        # Last N turns = last N*2 messages
        max_messages = self._max_history_turns * 2
        trimmed_history = history[-max_messages:] if len(history) > max_messages else history

        # Build the user message: history + question
        history_text = "\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in trimmed_history
        )
        user_message = f"History:\n{history_text}\n\nQuestion: {query}\nRewritten:"

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            rewritten = response.choices[0].message.content.strip()

            logger.info(
                "Query rewritten",
                original_query=query,
                rewritten_query=rewritten,
                history_turns=len(trimmed_history) // 2,
            )

            # Safety: if rewriter returns empty string, fall back to original
            if not rewritten:
                logger.warning("Query rewriter returned empty, falling back to original")
                return query

            return rewritten

        except Exception as exc:
            # Fail-safe: if rewriter fails, log and return original query
            # Better to retrieve based on original than to crash the request
            logger.error(
                "Query rewriter failed, falling back to original query",
                error=str(exc),
                query=query,
            )
            return query
