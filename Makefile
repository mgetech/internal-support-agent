.PHONY: setup lint format test db-up reset-db seed ui eval eval-publish

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

# The database tests drop and rebuild the schema of their database, so they run against
# a test database. tests/database_guard.py makes support_agent_test on the same server
# if it is missing and prints its URL. The suite refuses to run on any other database.
test:
	DATABASE_URL="$$($(PYTHON) -m tests.database_guard)" $(PYTHON) -m pytest

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

# run every scenario against the API at http://localhost:8000 and write the table to a new
# file in eval/.runs/. Start the API first. Pass other options with ARGS, for example:
# make eval ARGS="S001"
eval:
	$(PYTHON) -m eval.run_eval $(ARGS)

# copy one run to eval/results/ so it can be committed, for example:
# make eval-publish NAME=20261105T091500Z
eval-publish:
	$(PYTHON) -m eval.publish $(NAME)
