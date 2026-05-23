#!/usr/bin/env python
"""Download real Indian government PDFs for Phase 3a corpus expansion.

Downloads two PDFs into data/sample/:
  - constitution_of_india.pdf   (Constitution of India, cdnbbsr.s3waas.gov.in)
  - arc_ethics_governance.pdf   (ARC 4th Report: Ethics in Governance, darpg.gov.in)

Skips files that already exist. Retries once on network error. Exits non-zero
if a download fails twice — do not continue the ingestion pipeline with a
partial corpus.

Usage
-----
    python scripts/download_sample_pdfs.py
    python scripts/download_sample_pdfs.py --out-dir data/sample
"""

import argparse
import sys
import time
from pathlib import Path

import requests
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = structlog.get_logger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

_PDFS = [
    {
        "filename": "constitution_of_india.pdf",
        "url": "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2024/07/20240716890312078.pdf",
        "description": "Constitution of India (cdnbbsr.s3waas.gov.in)",
    },
    {
        "filename": "arc_ethics_governance.pdf",
        "url": "https://darpg.gov.in/sites/default/files/ethics4.pdf",
        "description": "ARC 4th Report: Ethics in Governance (darpg.gov.in)",
    },
]


def _download_one(url: str, dest: Path, description: str) -> None:
    """Download a single PDF to dest. Retries once on failure."""
    log = logger.bind(url=url, dest=str(dest), description=description)

    for attempt in range(1, 3):
        log.info("Downloading", attempt=attempt)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=60, stream=True)
            resp.raise_for_status()

            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)

            size_kb = dest.stat().st_size // 1024
            log.info("Download complete", size_kb=size_kb)
            return

        except requests.RequestException as exc:
            log.warning("Download failed", attempt=attempt, error=str(exc))
            if attempt == 1:
                time.sleep(3)
            else:
                log.error("Download failed after 2 attempts — aborting")
                if dest.exists():
                    dest.unlink()
                raise SystemExit(1) from exc


def main() -> None:
    p = argparse.ArgumentParser(description="Download Phase 3a corpus PDFs")
    p.add_argument("--out-dir", default="data/sample", help="Directory to write PDFs")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in _PDFS:
        dest = out_dir / spec["filename"]
        if dest.exists():
            size_kb = dest.stat().st_size // 1024
            logger.info("Already exists — skipping", filename=spec["filename"], size_kb=size_kb)
            continue
        _download_one(spec["url"], dest, spec["description"])

    logger.info("All PDFs ready", out_dir=str(out_dir))
    for spec in _PDFS:
        dest = out_dir / spec["filename"]
        print(f"  {dest}  ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
