#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for command in curl flock ollama uv; do
    if ! command -v "$command" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$command" >&2
        exit 1
    fi
done

DOCS_DIR="$HOME/NutritionScannerbackend/research"
if [[ ! -d "$DOCS_DIR" ]]; then
    printf 'Document directory not found: %s\n' "$DOCS_DIR" >&2
    exit 1
fi

mkdir -p data
exec 9>"$SCRIPT_DIR/data/start.lock"
flock 9

OLLAMA_URL="http://127.0.0.1:11434"
OLLAMA_LOG="$SCRIPT_DIR/data/ollama.log"
EMBED_MODEL="bge-m3"

ollama_is_ready() {
    curl --fail --silent --max-time 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1
}

if ollama_is_ready; then
    printf 'Ollama is already running.\n'
else
    printf 'Starting Ollama...\n'
    nohup ollama serve >>"$OLLAMA_LOG" 2>&1 </dev/null &
    printf 'Ollama process started (PID %s).\n' "$!"

    for ((attempt = 0; attempt < 60; attempt++)); do
        if ollama_is_ready; then
            break
        fi
        sleep 1
    done

    if ! ollama_is_ready; then
        printf 'Ollama did not become ready. See %s for details.\n' "$OLLAMA_LOG" >&2
        exit 1
    fi
fi

if ! ollama list | awk -v model="$EMBED_MODEL" \
    'NR > 1 && ($1 == model || $1 == model ":latest") { found = 1 } END { exit !found }'; then
    printf 'Pulling Ollama embedding model %s...\n' "$EMBED_MODEL"
    ollama pull "$EMBED_MODEL"
fi

printf 'Ingesting documents from %s...\n' "$DOCS_DIR"
CUDA_VISIBLE_DEVICES="" uv run --project "$SCRIPT_DIR" python "$SCRIPT_DIR/src/ingest.py"
