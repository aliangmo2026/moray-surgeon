# Private deployment with Docker

`moray-surgeon` ships as a single image that contains the web console, the CLI,
the verification tools (ruff / mypy / pytest), Git and Node.js 22 with the
TypeScript bridge. It runs fully on an internal network; the only outbound call
is to the model endpoint you configure (a local Ollama container, or your own
OpenAI/Anthropic-compatible gateway).

```text
browser ──► 127.0.0.1:8080 ──► moray-surgeon container
                                 ├── /workspace   bind mount: your git repos
                                 ├── .moray-cache per repo (SQLite, local disk)
                                 ├── ruff / mypy / pytest
                                 └── Node 22 + ts-morph
                                         │
                         optional profile│
                                         ▼
                                 ollama:11434 (local model)
```

## 1. Prerequisites

- Docker Engine 24+ (or Docker Desktop) with Compose v2.
- 4 GB free RAM for the app; 16 GB+ if you also run a 7B local model.
- Build-time internet access for `pip` and `npm`, or a prebuilt image
  (see [Air-gapped installs](#8-air-gapped-installs)).
- Repositories to refactor must live under the mounted workspace.

## 2. Quick start (cloud or internal model gateway)

```bash
cp .env.example .env
$EDITOR .env          # set MORAY_API_TYPE / MORAY_BASE_URL / MORAY_MODEL / MORAY_API_KEY
docker compose up -d --build
docker compose ps
```

The API key stays in `.env` and is injected into the container environment; it is
never baked into the image and the browser never receives it.

Open the console:

```text
http://localhost:8080
```

Enter the project path as **`/workspace/<repo-name>`** (for example
`/workspace/payments-api`). The model, base URL and dialect inputs are prefilled
from the container environment (the page reads `GET /api/config`), so a Docker
deployment usually needs no manual changes in the browser.

Health check:

```bash
curl -fsS http://127.0.0.1:8080/api/health
# {"status":"ok","version":"0.3.0"}
```

## 3. Fully local with Ollama

Put this in `.env`:

```dotenv
MORAY_API_TYPE=ollama
MORAY_BASE_URL=http://ollama:11434
MORAY_MODEL=qwen2.5-coder:7b
```

Start the app **and** the local model:

```bash
docker compose --profile ollama up -d --build
docker compose logs -f ollama-pull    # first start downloads the model (~4.7 GB for 7B)
```

`ollama-pull` is a one-shot helper: it waits for the server, pulls `MORAY_MODEL`,
then exits. Refactors can only reach the model once the pull is finished. The
model cache lives in the `ollama-models` volume and survives restarts. For NVIDIA
GPUs, uncomment the `deploy.resources` block in `docker-compose.yml`.

## 4. CLI in the container

The image entrypoint is `moray-surgeon`, so `docker compose run` takes CLI
arguments directly:

```bash
docker compose run --rm moray-surgeon refactor \
  "rename old_name to new_name across the repo" --path /workspace/payments-api

docker compose run --rm moray-surgeon refactor "..." --path /workspace/payments-api --dry-run

docker compose exec moray-surgeon moray-surgeon cache --path /workspace/payments-api --stats
docker compose exec moray-surgeon moray-surgeon version
```

`docker compose exec` runs a command inside the running container directly, so it
needs the binary name (`moray-surgeon …`); `docker compose run` goes through the
entrypoint and does not.

## 5. Workspace and permissions (Linux)

On Docker Desktop for macOS/Windows, file ownership is virtualised and the
defaults work. On a Linux host, the container user must be able to write
`.moray-cache/` and create commits, so match the host UID/GID:

```bash
mkdir -p workspace
sudo chown -R "$(id -u):$(id -g)" workspace

MORAY_UID=$(id -u) MORAY_GID=$(id -g) docker compose build
docker compose up -d
```

If `docker compose up` created `./workspace` itself, it is owned by `root`; the
`chown` above fixes that. The image configures `safe.directory '*'` for Git
(bind-mounted repositories are owned by the host) and a default commit identity;
override it with `MORAY_GIT_NAME` / `MORAY_GIT_EMAIL` in `.env`.

Keep the workspace on a **local filesystem**. SQLite uses WAL mode for
`.moray-cache/index.sqlite3`; network filesystems (NFS/SMB) can corrupt or lock
it.

## 6. Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `MORAY_BIND_HOST` | `127.0.0.1` | Host address the console binds to |
| `MORAY_PORT` | `8080` | Host port mapped to container port 8080 |
| `MORAY_WORKSPACE` | `./workspace` | Host directory mounted at `/workspace` |
| `MORAY_API_TYPE` | `ollama` | `ollama`, `openai` or `anthropic` |
| `MORAY_MODEL` | `qwen2.5-coder:7b` | Model name/tag |
| `MORAY_BASE_URL` | `http://ollama:11434` | Model API root (compose network name) |
| `MORAY_API_KEY` | unset | Credential for openai/anthropic dialects |
| `MORAY_TIMEOUT` | `120` | Per-request model timeout (seconds) |
| `MORAY_MAX_WORKERS` | `4` | Parallel workers for independent file edits |
| `MORAY_GIT_NAME` / `MORAY_GIT_EMAIL` | `moray-surgeon` / `…@localhost` | Commit identity |
| `MORAY_UID` / `MORAY_GID` | `1000` | Build-time runtime user IDs (Linux bind mounts) |

## 7. Operations

```bash
docker compose logs -f moray-surgeon     # server log
docker compose restart moray-surgeon
docker compose down                      # stop; keeps the ollama-models volume
docker compose down -v                   # also delete the model cache
```

Upgrade:

```bash
git pull
docker compose build --pull
docker compose up -d
```

Backup: source code lives in your repositories (commit/push them); the index
cache is disposable and rebuilt on demand; the Ollama models volume can be
re-created by pulling again.

## 8. Air-gapped installs

Build on a connected machine, then transfer the image:

```bash
# connected
docker compose build
docker save moray-surgeon:local -o moray-surgeon.tar

# offline host
docker load -i moray-surgeon.tar
docker compose up -d --no-build
```

The image already contains Python, Node.js and every npm package (`ts-morph`) —
no runtime downloads. For a local model, pull it on a connected machine and move
the `ollama-models` volume (or run the same `ollama-pull` service somewhere with
access, then copy the volume). The only runtime network dependency is the model
endpoint you configure.

## 9. Security notes

- **The console has no built-in authentication.** It binds to `127.0.0.1` by
  default. Before exposing it, put it behind a TLS reverse proxy with SSO/basic
  auth and set `MORAY_BIND_HOST=0.0.0.0` (or use the proxy's Docker network).
- `.env` contains the API key: `chmod 600 .env`, never commit it, and rotate the
  key like any other credential.
- The container runs as a non-root user with `no-new-privileges`; do not mount the
  Docker socket into it.
- The engine is designed to modify and commit the mounted repositories. Mount only
  directories you are willing to let it change, and keep backups.
- Source context is sent to the configured model endpoint. For strict data
  residency, use the local Ollama profile or your own gateway.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `permission denied` writing `/workspace` | Match UID/GID: `MORAY_UID=$(id -u) MORAY_GID=$(id -g) docker compose build`, and `chown` the host directory |
| `fatal: detected dubious ownership` | The image sets `safe.directory '*'`; if you use a custom image, add the same git system config |
| `Connection refused` to the model | Inside the container `localhost` is the container: use `http://ollama:11434` and start with `--profile ollama` |
| `port is already allocated` | Change `MORAY_PORT` in `.env` |
| `database is locked` / index errors | Move the workspace off NFS/SMB; SQLite WAL needs a local disk |
| Console: "not a git repository" | Enter the path of a cloned repo inside the mount, e.g. `/workspace/my-repo` |
| Model errors in the console | Check `docker compose logs moray-surgeon`; verify `MORAY_API_KEY` and `MORAY_BASE_URL` |
| Verify Node/TypeScript support | `docker compose exec moray-surgeon node --version` (expect v22.x) |

Useful diagnostics:

```bash
docker compose exec moray-surgeon moray-surgeon version
docker compose exec moray-surgeon python -c "import libcst, fastapi, uvicorn; print('ok')"
docker compose exec moray-surgeon node --version
```
