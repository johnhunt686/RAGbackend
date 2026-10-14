#!/usr/bin/env bash
# reset.sh: wipe the vector store and ingestion manifest (leaves data/docs untouched)
set -euo pipefail
cd "$(dirname "$0")"

rm -rf data/lancedb data/manifest.json
echo "Cleared data/lancedb and data/manifest.json"