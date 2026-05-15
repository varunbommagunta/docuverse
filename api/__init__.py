"""FastAPI HTTP adapter package.

This package adapts the DocuVerse business logic (src/) to the HTTP protocol.
It must never import domain logic directly into route handlers — all business
operations go through the RAGOrchestrator.
"""
