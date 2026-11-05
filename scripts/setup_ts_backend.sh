#!/usr/bin/env bash
# Install the Node.js dependencies for moray-surgeon's TypeScript backend.
#
# Requirements: Node.js 18+ and npm on PATH. The bridge itself is checked in;
# this script only fetches ts-morph into ts_bridge/node_modules (git-ignored)
# and runs a protocol smoke test.
#
# Usage:
#   bash scripts/setup_ts_backend.sh
#
# Re-running is safe: it is an idempotent install.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_DIR="$SCRIPT_DIR/../src/moray_surgeon/engine/backends/ts_bridge"

if ! command -v node >/dev/null 2>&1; then
  echo "error: Node.js 18+ is required but 'node' was not found on PATH." >&2
  echo "       Install it from https://nodejs.org/ and re-run this script." >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "error: npm is required but was not found on PATH." >&2
  exit 1
fi

NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
if [ "$NODE_MAJOR" -lt 18 ]; then
  echo "error: Node.js 18+ is required; found $(node --version)." >&2
  exit 1
fi

if [ ! -f "$BRIDGE_DIR/bridge.mjs" ]; then
  echo "error: bridge not found at $BRIDGE_DIR/bridge.mjs" >&2
  exit 1
fi

echo "Installing ts-morph in $BRIDGE_DIR ..."
cd "$BRIDGE_DIR"
if [ -f package-lock.json ]; then
  npm ci --omit=dev --no-audit --no-fund
else
  npm install --omit=dev --no-audit --no-fund
fi

echo "Smoke-testing the JSON-RPC bridge ..."
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"ping"}' | node bridge.mjs
echo
echo "TypeScript backend ready."
