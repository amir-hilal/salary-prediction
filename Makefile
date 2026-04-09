.PHONY: help data-download pipeline train api dashboard test lint docker-up docker-down clean

help:
	@echo ""
	@echo "  Salary Prediction — Developer Commands"
	@echo "  ────────────────────────────────────────────────────"
	@echo "  make data-download   Download dataset from Kaggle"
	@echo "  make pipeline        Full data → feature → train pipeline"
	@echo "  make train           Train model only"
	@echo "  make api             Start FastAPI dev server"
	@echo "  make dashboard       Start Streamlit dashboard"
	@echo "  make test            Run test suite"
	@echo "  make lint            Run ruff + mypy"
	@echo "  make docker-up       Spin up full stack (Docker Compose)"
	@echo "  make docker-down     Stop Docker Compose stack"
	@echo "  make clean           Remove generated artifacts and caches"
	@echo ""

# ─── Data ────────────────────────────────────────────────────────────────────
data-download:
	python -m src.data.ingestion

# ─── Pipeline ────────────────────────────────────────────────────────────────
pipeline: data-download train

train:
	python -m src.models.train

# ─── Services ────────────────────────────────────────────────────────────────
api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	PYTHONPATH=$(PWD) streamlit run dashboard/app.py --server.port 8501

# ─── Quality ─────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ dashboard/ config/ tests/
	mypy src/ config/

# ─── Docker ──────────────────────────────────────────────────────────────────
docker-up:
	docker compose -f deployment/docker/docker-compose.yml up --build -d

docker-down:
	docker compose -f deployment/docker/docker-compose.yml down

# ─── Housekeeping ────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
