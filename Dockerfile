# BASE
FROM python:3.11-slim-bookworm AS base
# Install curl
RUN apt-get update && apt-get install -y curl

# DEVELOPMENT
FROM base AS development

# Install poetry
ENV POETRY_HOME=/opt/poetry \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR='/tmp/poetry_cache' \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN curl -sSL https://install.python-poetry.org | python3
ENV PATH="${POETRY_HOME}/bin:${PATH}"


# Set the working directory
WORKDIR /app

# Copy the poetry.lock and pyproject.toml
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

# Install spacy and nltk
ENV NLTK_DATA=/app/nltk_data
RUN . .venv/bin/activate \
    && python -m spacy download es_core_news_lg
RUN . .venv/bin/activate \
    && python -m nltk.downloader stopwords -d ${NLTK_DATA}

# BUILDER
FROM development AS builder
WORKDIR /app
COPY . .
RUN poetry install --without dev

# RUNTIME
FROM base AS runtime

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    ENVIRONMENT=production \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WEB_CONCURRENCY=4 \
    NLTK_DATA=/app/nltk_data

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=development ${NLTK_DATA} ${NLTK_DATA}

RUN mkdir -p /app/logs && touch /app/logs/log.log

COPY src ./src
COPY config ./config
COPY models ./models
COPY gunicorn.py .env* ./

EXPOSE 8000

ENTRYPOINT [ "gunicorn", "-c", "gunicorn.py", "alba.app:create_app()" ]
