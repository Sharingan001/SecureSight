.PHONY: dev up down logs test clean

# ── Development ────────────────────────────────────────────
dev:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# ── Docker ─────────────────────────────────────────────────
up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api worker

# ── Testing ────────────────────────────────────────────────
test:
	cd backend && python -m pytest tests/ -v --tb=short

# ── Cleanup ────────────────────────────────────────────────
clean:
	rm -rf /tmp/securesight
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
