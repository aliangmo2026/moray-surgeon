# Install the Node.js dependencies for moray-surgeon's TypeScript backend.
#
# Requirements: Node.js 18+ and npm on PATH. The bridge itself is checked in;
# this script only fetches ts-morph into ts_bridge\node_modules (git-ignored)
# and runs a protocol smoke test.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_ts_backend.ps1
#
# Re-running is safe: it is an idempotent install.

$ErrorActionPreference = "Stop"

$BridgeDir = Join-Path $PSScriptRoot "..\src\moray_surgeon\engine\backends\ts_bridge"

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js 18+ is required but 'node' was not found on PATH. Install it from https://nodejs.org/."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm is required but was not found on PATH."
}

$NodeMajor = [int](node -p "process.versions.node.split('.')[0]")
if ($NodeMajor -lt 18) {
    throw "Node.js 18+ is required; found $(node --version)."
}

$BridgeScript = Join-Path $BridgeDir "bridge.mjs"
if (-not (Test-Path $BridgeScript)) {
    throw "bridge not found at $BridgeScript"
}

Write-Host "Installing ts-morph in $BridgeDir ..."
Push-Location $BridgeDir
try {
    if (Test-Path "package-lock.json") {
        npm ci --omit=dev --no-audit --no-fund
    } else {
        npm install --omit=dev --no-audit --no-fund
    }
    if ($LASTEXITCODE -ne 0) {
        throw "npm failed with exit code $LASTEXITCODE"
    }

    Write-Host "Smoke-testing the JSON-RPC bridge ..."
    '{"jsonrpc":"2.0","id":1,"method":"ping"}' | node bridge.mjs
    if ($LASTEXITCODE -ne 0) {
        throw "bridge smoke test failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "TypeScript backend ready."
