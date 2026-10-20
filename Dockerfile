# syntax=docker/dockerfile:1.7
# One image for both services (bot + portal). Three stages keep it small:
#   frontend  - builds the React portal (node only exists here)
#   deps      - resolves the locked Python dependencies with uv (no dev tools)
#   runtime   - python:slim + the prebuilt venv + source + built UI, non-root

# ---------- frontend ----------
FROM node:22-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------- python deps ----------
FROM python:3.12-slim AS deps
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock backend/.python-version ./
# Locked, production-only install into /app/backend/.venv (same path as runtime,
# so the venv's absolute paths stay valid).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
    # third-party test suites are dead weight at runtime (~65 MB)
    && find .venv/lib/python3.12/site-packages -type d \( -name tests -o -name test \) -prune -exec rm -rf {} +

# ---------- runtime ----------
FROM python:3.12-slim AS runtime
# libglib2.0-0: opencv-headless (pulled in by pipecat's webrtc extra) links against it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app/backend
COPY --from=deps --chown=app:app /app/backend/.venv ./.venv
COPY --chown=app:app backend/ ./
COPY --from=frontend --chown=app:app /app/frontend/dist /app/frontend/dist
ENV PATH="/app/backend/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER app
EXPOSE 7860 8090
# Default: the voice agent with every transport enabled. compose.yaml overrides
# this per service (Plivo mode, portal).
CMD ["python", "bot.py", "--host", "0.0.0.0", "--port", "7860"]
