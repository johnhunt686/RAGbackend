# RAGbackend

## Start the document indexer

From the repository root, run:

```bash
./start.sh
```

The script requires `uv`, `ollama`, `curl`, and `flock`. It starts Ollama only
if its local API is not already responding, pulls the configured `bge-m3`
embedding model if needed, and ingests supported documents (`.pdf`, `.md`, and
`.txt`) from `~/NutritionScannerbackend/research`. Ollama remains running after
ingestion; the script does not clear the existing index.

To retrieve matching chunks after ingestion:

```bash
uv run python src/query.py "Your question"
```