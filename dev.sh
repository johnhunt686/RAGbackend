#!/usr/bin/env bash
# ./dev.sh  starts the RAG API and LiteLLM; Ctrl-C stops both.
ROOT="$(cd "$(dirname "$0")" && pwd)"
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-dev}"

trap 'trap - INT TERM EXIT; kill 0' INT TERM EXIT

(cd "$ROOT/RAG" && exec .venv/bin/python -m uvicorn main:app --app-dir src --host 127.0.0.1 --port 8000) &
(cd "$ROOT/litellm" && exec .venv/bin/litellm --config config.yaml --host 127.0.0.1 --port 4000) &
wait