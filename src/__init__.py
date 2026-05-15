"""DocuVerse business-logic package.

This package is framework-agnostic: it knows nothing about HTTP, Streamlit, or
any I/O adapter. All external dependencies (OpenAI, vector DBs) are injected
through Protocol interfaces defined in each sub-package's base.py.
"""
