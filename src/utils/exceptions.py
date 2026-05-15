"""Custom exception hierarchy for DocuVerse.

All domain errors inherit from DocuVerseError so callers can catch the base
class when they want to handle any application error, or catch specific
subclasses for targeted recovery logic.
"""


class DocuVerseError(Exception):
    """Base class for all DocuVerse application errors."""


class DocumentParseError(DocuVerseError):
    """Raised when a PDF or document cannot be parsed.

    Typical causes: corrupted file, password-protected PDF, unsupported format.
    """


class ChunkingError(DocuVerseError):
    """Raised when text splitting / chunking fails.

    Typical causes: empty document, invalid chunker configuration.
    """


class RetrievalError(DocuVerseError):
    """Raised when the vector store query fails or returns unexpected results.

    Typical causes: vector DB connection error, dimension mismatch, empty index.
    """


class GenerationError(DocuVerseError):
    """Raised when the LLM call fails or returns an unusable response.

    Typical causes: API rate limit, context window exceeded, content filter.
    """
