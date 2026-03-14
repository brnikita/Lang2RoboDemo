FROM python:3.11-slim AS backend

WORKDIR /app

# System deps for MuJoCo (OpenGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libglew-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY backend/ backend/
COPY knowledge-base/ knowledge-base/
COPY prompts/ prompts/

RUN pip install --no-cache-dir -e .

# Build frontend
FROM node:22-slim AS frontend

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Final image
FROM backend AS final

COPY --from=frontend /app/frontend/dist /app/frontend/dist

RUN mkdir -p /app/data /app/models

EXPOSE ${BACKEND_PORT:-8000}

CMD uvicorn backend.app.main:app --host ${BACKEND_HOST:-0.0.0.0} --port ${BACKEND_PORT:-8000}
