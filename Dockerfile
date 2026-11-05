# syntax=docker/dockerfile:1.7

# moray-surgeon — private-deployment image.
#
#   docker build -t moray-surgeon .
#   docker run --rm -p 127.0.0.1:8080:8080 \
#       -v "$PWD/workspace:/workspace" moray-surgeon
#
# The final image is based on python:3.12-slim and contains:
#   * Git                 — transactions (stash/commit/rollback) on the workspace
#   * Node.js 22 + npm    — TypeScript backend bridge (ts-morph)
#   * [dev] Python extras — ruff/mypy/pytest, so the console can verify a refactor
#   * A non-root user     — the workspace is mounted into /workspace
#
# The default command is the web console; the same entrypoint serves the CLI:
#   docker run ... moray-surgeon refactor "rename old_name" --path /workspace/repo

# --------------------------------------------------------------------------- #
# Node.js runtime stage: copied into the Python image so the final image stays
# on python:3.12-slim and no curl|bash installer or extra apt repo is needed.
# --------------------------------------------------------------------------- #
FROM node:22-bookworm-slim AS node-runtime

# --------------------------------------------------------------------------- #
# Runtime image
# --------------------------------------------------------------------------- #
FROM python:3.12-slim

# Host-matching IDs for Linux bind mounts:
#   MORAY_UID=$(id -u) MORAY_GID=$(id -g) docker compose build
ARG MORAY_UID=1000
ARG MORAY_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MORAY_HOST=0.0.0.0 \
    MORAY_PORT=8080

# ca-certificates: HTTPS model APIs and package indexes.
# git: every transaction is a git subprocess.
# libstdc++6: the Node.js binary copied below is dynamically linked.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Node 22 LTS + npm from the official image (same Debian base as this one).
COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node-runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && node --version \
    && npm --version

# Workspace repositories are bind mounts owned by the host user. The container is
# the isolation boundary; without this, git refuses to read them ("dubious
# ownership") whenever the host UID differs from the image user.
# The commit identity is a default: GIT_AUTHOR_* / GIT_COMMITTER_* environment
# variables (set in docker-compose.yml) override it per commit.
RUN git config --system --add safe.directory '*' \
    && git config --system user.name 'moray-surgeon' \
    && git config --system user.email 'moray-surgeon@localhost'

WORKDIR /app

# Copy metadata first so dependency installation is cached until pyproject.toml
# changes. The editable install keeps src/ in place, which is where the packaged
# TypeScript bridge expects its node_modules directory.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# The pip cache lives in a BuildKit cache mount, not in the image layer, so a
# source-only rebuild re-downloads nothing but still produces a clean layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -e ".[dev]"

# TypeScript bridge dependencies. Pure JavaScript (no compiler toolchain), and
# npm ci follows package-lock.json for a reproducible install.
WORKDIR /app/src/moray_surgeon/engine/backends/ts_bridge
RUN --mount=type=cache,target=/root/.npm \
    npm ci --omit=dev --no-audit --no-fund

# Non-root runtime user. /workspace is a mount point: when a bind mount hides it,
# the host directory's ownership governs (match it with MORAY_UID/MORAY_GID).
RUN groupadd --gid "${MORAY_GID}" moray \
    && useradd --uid "${MORAY_UID}" --gid "${MORAY_GID}" \
        --create-home --home-dir /home/moray --shell /bin/bash moray \
    && mkdir -p /workspace \
    && chown -R moray:moray /home/moray /workspace

# Set HOME only after useradd created the home directory: pointing it at a
# missing directory during the build makes root own /home/moray instead.
ENV HOME=/home/moray

USER moray
WORKDIR /workspace

# 8080: web console. The CLI is reachable through the same entrypoint.
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=4).status == 200 else 1)"

# Default: the web console. Use the same image for CLI commands:
#   docker run ... moray-surgeon refactor "..." --path /workspace/repo
ENTRYPOINT ["moray-surgeon"]
CMD ["web", "--host", "0.0.0.0", "--port", "8080", "--no-open"]
