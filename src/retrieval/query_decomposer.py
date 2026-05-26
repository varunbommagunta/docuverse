"""LLM-based query decomposer for multi-aspect questions.

Complex questions that span multiple concepts (e.g. "How do Fundamental Rights
and Directive Principles differ?") fail retrieval because every chunk gets scored
against the full query — chunks covering one concept rank higher than chunks
covering the other, so the reranker drops the latter.

QueryDecomposer detects multi-aspect queries and splits them into 2-3 short,
targeted sub-queries that can each be retrieved independently. The orchestrator
then merges results from all sub-queries before generation.

Simple single-concept queries are returned unchanged (as a one-element list) so
the normal pipeline runs with no overhead.
"""

import json
import structlog
from openai import OpenAI

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are a query decomposer for a document retrieval system.

Given a question, decide whether it asks about multiple distinct concepts that should be retrieved separately.

Rules:
1. If the question covers 2-3 distinct concepts, split it into one short retrieval query per concept.
2. If the question is about a single concept, return it unchanged as a one-element list.
3. Each sub-query must be independently searchable — a short noun phrase or focused question.
4. Maximum 3 sub-queries. Do not over-split.
5. Output ONLY a valid JSON array of strings. No explanation, no markdown.

Examples:
Q: "How do Fundamental Rights and Directive Principles differ in enforceability?"
["What are Fundamental Rights and are they justiciable", "What are Directive Principles of State Policy and are they enforceable"]

Q: "Compare the powers of Lok Sabha and Rajya Sabha in passing money bills"
["Powers and functions of Lok Sabha", "Powers and functions of Rajya Sabha", "Money bill procedure Lok Sabha Rajya Sabha"]

Q: "What does Article 21 protect?"
["What does Article 21 protect"]

Q: "What are the protections under Articles 14, 19, and 21 together?"
["Article 14 equality before law", "Article 19 freedom of speech and other rights", "Article 21 right to life and personal liberty"]

Q: "What is the official language of India?"
["What is the official language of India"]"""


class QueryDecomposer:
    """Splits multi-aspect queries into focused sub-queries for parallel retrieval."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = OpenAI(api_key=api_key.strip())
        self._model = model
        logger.info("QueryDecomposer initialised", model=model)

    def decompose(self, query: str) -> list[str]:
        """Return a list of sub-queries for the given query.

        Returns a one-element list (the original query) when decomposition is
        not needed, so callers can always treat the result as a list.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Q: {query}"},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            sub_queries = json.loads(raw)

            if (
                not isinstance(sub_queries, list)
                or not sub_queries
                or not all(isinstance(q, str) and q.strip() for q in sub_queries)
            ):
                logger.warning("decomposer_invalid_output", raw=raw, fallback="original")
                return [query]

            sub_queries = [q.strip() for q in sub_queries[:3]]

            if len(sub_queries) == 1:
                logger.info("query_not_decomposed", query=query[:80])
            else:
                logger.info(
                    "query_decomposed",
                    original=query[:80],
                    sub_queries=sub_queries,
                    count=len(sub_queries),
                )

            return sub_queries

        except Exception as exc:
            logger.error("decomposer_failed", error=str(exc), fallback="original")
            return [query]
