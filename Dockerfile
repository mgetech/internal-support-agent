FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

# install the pinned dependencies from pyproject.toml first, so this layer is reused
# until the dependency list changes
COPY pyproject.toml .
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN useradd --create-home app \
    && mkdir /app/.embeddings-cache \
    && chown app:app /app /app/.embeddings-cache

# prompts.py finds prompts/ next to src/, so the layout of the repository is kept
COPY --chown=app:app src ./src
COPY --chown=app:app prompts ./prompts
COPY --chown=app:app data ./data
COPY --chown=app:app ui ./ui

USER app

EXPOSE 8000 8501
CMD ["uvicorn", "support_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
