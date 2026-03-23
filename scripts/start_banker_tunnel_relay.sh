#!/usr/bin/env bash
set -euo pipefail

# Ce script tourne sur le serveur Proxmox. Il publie un relay TCP visible
# depuis les conteneurs Docker et raccorde ce relay au reverse tunnel SSH
# ouvert par le Banker Windows.

LISTEN_HOST="${1:-0.0.0.0}"
LISTEN_PORT="${2:-18101}"
TARGET_HOST="${3:-127.0.0.1}"
TARGET_PORT="${4:-18100}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="/tmp/banker_tunnel_relay.log"

pkill -f "banker_tunnel_relay.py --listen-host ${LISTEN_HOST} --listen-port ${LISTEN_PORT}" 2>/dev/null || true

nohup python3 "${ROOT_DIR}/scripts/banker_tunnel_relay.py" \
  --listen-host "${LISTEN_HOST}" \
  --listen-port "${LISTEN_PORT}" \
  --target-host "${TARGET_HOST}" \
  --target-port "${TARGET_PORT}" \
  >"${LOG_FILE}" 2>&1 &
