# =============================================================================
# Izwi Audio - Multi-stage Dockerfile
# Supports both CPU and CUDA environments
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build the React UI
# -----------------------------------------------------------------------------
FROM node:20-slim AS ui-builder

WORKDIR /app/ui

# Copy package files first for better caching
COPY ui/package*.json ./

# Install dependencies
RUN npm ci --ignore-scripts

# Copy source and build
COPY ui/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Build the Rust backend
# -----------------------------------------------------------------------------
FROM rust:1.83-bookworm AS rust-builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Cargo files first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# Build release binary
RUN cargo build --release --bin izwi

# -----------------------------------------------------------------------------
# Stage 3: Python environment with uv
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS python-builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy Python project files
COPY pyproject.toml README.md ./
COPY scripts/ scripts/

# Create virtual environment and install dependencies
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN uv venv /app/.venv
RUN uv pip install --python /app/.venv/bin/python -e .

# -----------------------------------------------------------------------------
# Stage 4: Production runtime (CPU)
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS production

LABEL org.opencontainers.image.title="Izwi Audio"
LABEL org.opencontainers.image.description="Qwen3-TTS inference engine"
LABEL org.opencontainers.image.vendor="Agentem"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 izwi

# Copy Python virtual environment
COPY --from=python-builder /app/.venv /app/.venv
COPY --from=python-builder /app/scripts /app/scripts

# Copy Rust binary
COPY --from=rust-builder /app/target/release/izwi /usr/local/bin/izwi

# Copy built UI
COPY --from=ui-builder /app/ui/dist /app/ui/dist

# Copy configuration
COPY config.toml /app/config.toml

# Set up environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV IZWI_CONFIG_PATH=/app/config.toml

# Create directories for models and data
RUN mkdir -p /app/models /app/data && \
    chown -R izwi:izwi /app

# Volume for model storage
VOLUME ["/app/models"]

# Expose server port
EXPOSE 8080

USER izwi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/models || exit 1

# Start the server
CMD ["izwi"]

# -----------------------------------------------------------------------------
# Stage 5: Production runtime with CUDA support
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS production-cuda

LABEL org.opencontainers.image.title="Izwi Audio (CUDA)"
LABEL org.opencontainers.image.description="Qwen3-TTS inference engine with CUDA support"
LABEL org.opencontainers.image.vendor="Agentem"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libssl3 \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 izwi \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy Python project files and install with CUDA extras
COPY pyproject.toml README.md ./
COPY scripts/ scripts/

RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python -e ".[cuda]"

# Copy Rust binary
COPY --from=rust-builder /app/target/release/izwi /usr/local/bin/izwi

# Copy built UI
COPY --from=ui-builder /app/ui/dist /app/ui/dist

# Copy configuration
COPY config.toml /app/config.toml

# Set up environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV IZWI_CONFIG_PATH=/app/config.toml

# Create directories for models and data
RUN mkdir -p /app/models /app/data && \
    chown -R izwi:izwi /app

# Volume for model storage
VOLUME ["/app/models"]

# Expose server port
EXPOSE 8080

USER izwi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/models || exit 1

# Start the server
CMD ["izwi"]
