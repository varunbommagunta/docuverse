.PHONY: install install-dev api ui test lint format docker-up docker-down clean

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# ── Run locally ───────────────────────────────────────────────────────────────
api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run ui/app.py --server.port 8501

# ── Tests & quality ───────────────────────────────────────────────────────────
test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up --build

docker-down:
	docker compose down

# ── Housekeeping ──────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
