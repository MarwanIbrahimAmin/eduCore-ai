#!/usr/bin/env bash

# EduCore launcher. On Windows, run from Git Bash to open visible terminals.
# On Linux/macOS, services run in the background and write logs under storage/logs.

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

API_PORT="${EDUCORE_API_PORT:-8002}"
UI_PORT="${EDUCORE_UI_PORT:-8501}"
API_URL="http://127.0.0.1:${API_PORT}/api"
IS_WINDOWS=false
case "$(uname -s 2>/dev/null || true)" in
  MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=true ;;
esac

step() { printf '\n\033[1;36m[%s/6] %s\033[0m\n' "$1" "$2"; }
fail() { printf '\n\033[1;31mEduCore startup stopped:\033[0m %s\n' "$1" >&2; exit 1; }

step 1 "Checking Python environment and configuration"
if [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON="$ROOT/.venv/Scripts/python.exe"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  fail "Virtual environment not found. Create .venv and install dependencies with: pip install -e '.[streamlit]'"
fi

"$PYTHON" -c 'import fastapi, uvicorn, streamlit, celery, openai, alembic' \
  || fail "Required packages are missing. Run: pip install -e '.[streamlit]'"
"$PYTHON" -c 'from educore.config import settings; import sys; sys.exit(0 if settings.OPENROUTER_API_KEY.strip() else 1)' \
  || fail "OPENROUTER_API_KEY is missing from .env. Add it there without sharing it in chat."
printf 'Python and OpenRouter configuration are ready.\n'

step 2 "Starting PostgreSQL, RabbitMQ, and Redis"
if command -v docker >/dev/null 2>&1; then
  docker compose up -d postgres rabbitmq redis \
    || fail "Docker services could not start. Check that Docker Desktop is running."
else
  printf 'Docker CLI not found; expecting PostgreSQL, RabbitMQ, and Redis to already be running.\n'
fi

printf 'Waiting for PostgreSQL, RabbitMQ, and Redis to accept connections...\n'
"$PYTHON" - <<'PY' || fail "A required service did not become ready. Check Docker Desktop and the service logs."
import socket
import sys
import time

services = (("PostgreSQL", 5432), ("RabbitMQ", 5672), ("Redis", 6379))
deadline = time.monotonic() + 90
while True:
    pending = []
    for name, port in services:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                pass
        except OSError:
            pending.append(name)
    if not pending:
        print("All required services are accepting connections.")
        break
    if time.monotonic() >= deadline:
        print("Timed out waiting for: " + ", ".join(pending), file=sys.stderr)
        sys.exit(1)
    time.sleep(1)
PY

step 3 "Applying database migrations"
"$PYTHON" -m alembic upgrade head \
  || fail "Database migration failed. Check the database services and DATABASE_URL in .env."

step 4 "Preparing visible service terminals"
mkdir -p "$ROOT/storage/logs"

if [[ "$IS_WINDOWS" == true ]]; then
  command -v cygpath >/dev/null 2>&1 || fail "Run this script from Git Bash so Windows paths can be resolved."
  ROOT_WIN="$(cygpath -w "$ROOT")"
  PYTHON_WIN="$(cygpath -w "$PYTHON")"

  open_terminal() {
    local title="$1"
    local ps_command="$2"
    local encoded
    encoded="$(printf '%s' "$ps_command" | iconv -f UTF-8 -t UTF-16LE | base64 | tr -d '\r\n')"
    if command -v wt.exe >/dev/null 2>&1; then
      wt.exe -w 0 new-tab --title "$title" powershell.exe -NoExit -EncodedCommand "$encoded"
    elif command -v cmd.exe >/dev/null 2>&1; then
      cmd.exe /c start "$title" powershell.exe -NoExit -EncodedCommand "$encoded"
    else
      return 1
    fi
  }

  step 5 "Opening FastAPI and Celery terminals"
  open_terminal "EduCore API" "Set-Location -LiteralPath '$ROOT_WIN'; \$Host.UI.RawUI.WindowTitle='EduCore API'; & '$PYTHON_WIN' -m uvicorn educore.main:app --host 127.0.0.1 --port $API_PORT --reload" \
    || fail "Could not open a terminal for FastAPI. Install Windows Terminal or PowerShell."
  open_terminal "EduCore Celery" "Set-Location -LiteralPath '$ROOT_WIN'; \$Host.UI.RawUI.WindowTitle='EduCore Celery'; & '$PYTHON_WIN' -m celery -A educore.celery_app worker --loglevel=info -Q educore.process --pool=solo --hostname=educore-run@%h" \
    || fail "Could not open a terminal for Celery."

  step 6 "Opening the Streamlit student/teacher interface"
  open_terminal "EduCore Streamlit" "Set-Location -LiteralPath '$ROOT_WIN'; \$Host.UI.RawUI.WindowTitle='EduCore Streamlit'; \$env:EDUCORE_API_BASE='${API_URL}'; & '$PYTHON_WIN' -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port $UI_PORT" \
    || fail "Could not open a terminal for Streamlit."
  printf '\nEduCore is starting. UI: http://127.0.0.1:%s  API docs: http://127.0.0.1:%s/docs\n' "$UI_PORT" "$API_PORT"
else
  step 5 "Starting FastAPI and Celery in the background"
  LOG_DIR="$ROOT/storage/logs"
  nohup "$PYTHON" -m uvicorn educore.main:app --host 127.0.0.1 --port "$API_PORT" \
    >"$LOG_DIR/api.log" 2>&1 &
  API_PID=$!
  if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
    CELERY_POOL="solo"
  else
    CELERY_POOL="prefork"
  fi
  nohup "$PYTHON" -m celery -A educore.celery_app worker --loglevel=info \
    -Q educore.process --pool="$CELERY_POOL" --hostname="educore-run@%h" \
    >"$LOG_DIR/celery.log" 2>&1 &
  CELERY_PID=$!

  step 6 "Starting Streamlit"
  EDUCORE_API_BASE="$API_URL" nohup "$PYTHON" -m streamlit run streamlit_app.py \
    --server.address 127.0.0.1 --server.port "$UI_PORT" \
    >"$LOG_DIR/streamlit.log" 2>&1 &
  UI_PID=$!
  printf '%s\n' "$API_PID" "$CELERY_PID" "$UI_PID" >"$LOG_DIR/pids"
  printf '\nEduCore is starting. UI: http://127.0.0.1:%s  API docs: http://127.0.0.1:%s/docs\n' "$UI_PORT" "$API_PORT"
  printf 'Logs: %s\n' "$LOG_DIR"
fi
