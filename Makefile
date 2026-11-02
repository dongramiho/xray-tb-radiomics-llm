# Developer shortcuts. See README for the full setup.
.PHONY: setup backend frontend test lint compose

setup:            ## create the backend venv and install everything (incl. model package)
	cd WebBack && python3.11 -m venv .venv && .venv/bin/pip install --upgrade pip "numpy<2" versioneer \
	  && .venv/bin/pip install --no-build-isolation -e "../model[inference]" \
	  && .venv/bin/pip install -r requirements.txt
	cd WebFront && npm install --no-audit --no-fund

models:           ## download U-Net weights from the GitHub release
	scripts/download_models.sh

backend:          ## run the API with auto-reload
	cd WebBack && .venv/bin/alembic upgrade head && .venv/bin/uvicorn main:app --reload --port 8000

frontend:         ## run the Next.js dev server
	cd WebFront && npm run dev

test:             ## backend tests (fake backends) + frontend lint/types
	cd WebBack && INFERENCE_BACKEND=fake LLM_BACKEND=fake .venv/bin/pytest
	cd WebFront && npm run lint && npx tsc --noEmit

lint:             ## ruff on backend + model package
	cd WebBack && .venv/bin/ruff check . && .venv/bin/ruff check --config pyproject.toml ../model/src ../model/config.py

compose:          ## full stack with PostgreSQL via Docker Compose
	docker compose up --build
