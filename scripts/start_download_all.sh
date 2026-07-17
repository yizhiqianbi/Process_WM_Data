#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${FASTWAM_DATA_ROOT:-$(cd -- "$PROJECT_ROOT/.." && pwd)}"
STATE_ROOT="${FASTWAM_DOWNLOAD_STATE_ROOT:-$DATA_ROOT/.fastwam_download}"
PID_FILE="$STATE_ROOT/download_all.pid"
LOG_DIR="$STATE_ROOT/operator_logs"
TMUX_SESSION="${FASTWAM_DOWNLOAD_TMUX_SESSION:-fastwam_download_all}"

mkdir -p -- "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(<"$PID_FILE")"
  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]]; then
    EXISTING_STATE="$(ps -o stat= -p "$EXISTING_PID" 2>/dev/null || true)"
    EXISTING_STATE="${EXISTING_STATE//[[:space:]]/}"
    if [[ -n "$EXISTING_STATE" && "$EXISTING_STATE" != Z* ]]; then
      printf 'Download already running: PID %s\n' "$EXISTING_PID" >&2
      exit 1
    fi
  fi
fi

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/download_all_$STAMP.log"
COMMAND_ARGS=(
  "$PYTHON_BIN"
  "$PROJECT_ROOT/scripts/download_datasets.py"
  download
  --datasets all
  --data-root "$DATA_ROOT"
  --state-root "$STATE_ROOT"
  "$@"
)

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "=$TMUX_SESSION" 2>/dev/null; then
    printf 'Download tmux session already exists: %s\n' "$TMUX_SESSION" >&2
    exit 1
  fi
  printf -v COMMAND ' %q' "${COMMAND_ARGS[@]}"
  printf -v QUOTED_LOG '%q' "$LOG_FILE"
  COMMAND="exec${COMMAND} >>${QUOTED_LOG} 2>&1"
  tmux new-session -d -s "$TMUX_SESSION" "$COMMAND"
  sleep 1
  if ! tmux has-session -t "=$TMUX_SESSION" 2>/dev/null; then
    printf 'Download exited during startup; inspect %s\n' "$LOG_FILE" >&2
    exit 1
  fi
  PID="$(tmux display-message -p -t "$TMUX_SESSION:0.0" '#{pane_pid}')"
else
  nohup "${COMMAND_ARGS[@]}" >"$LOG_FILE" 2>&1 </dev/null &
  PID=$!
fi

printf '%s\n' "$PID" >"$PID_FILE"
printf 'pid: %s\nlog: %s\nstate: %s\n' "$PID" "$LOG_FILE" "$STATE_ROOT"
if command -v tmux >/dev/null 2>&1; then
  printf 'tmux: %s\n' "$TMUX_SESSION"
fi
