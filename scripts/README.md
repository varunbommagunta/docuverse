# scripts/

One-off utility scripts for DocuVerse operations.

Scripts will be added here in later phases, for example:

- `ingest_sample.py` — bulk-ingest the sample PDFs in `data/sample/`
- `eval_baseline.py` — run the RAGAS evaluation harness (Phase 4)
- `reset_vector_db.py` — wipe and rebuild the Chroma index (dev utility)

All scripts should be runnable from the project root:

```bash
python scripts/<script_name>.py
```
