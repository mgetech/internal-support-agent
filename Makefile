.PHONY: setup lint format test db-up reset-db seed ui

VENV := .venv
ifeq ($(OS),Windows_NT)
PYTHON := $(VENV)/Scripts/python
else
PYTHON := $(VENV)/bin/python
endif

setup:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

db-up:
	docker compose up -d db

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

# delete the database volume, start a new database and seed it. Use it after a schema change,
# because db/init.sql only runs on a new volume
reset-db:
	docker compose down -v
	docker compose up -d --wait db
	$(MAKE) seed

# generate, truncate, load — safe to re-run any time
seed:
	$(PYTHON) -m data.generate

ui:
	$(PYTHON) -m streamlit run ui/app.py
