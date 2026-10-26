from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434"
LLM_MODEL = "qwen2.5:32b"
EMBED_MODEL = "bge-m3"
TOKENIZER = "BAAI/bge-m3"        # used by the chunker to count tokens

DOCS_DIR = Path("data/docs")
LANCEDB_URI = "data/lancedb"
TABLE_NAME = "chunks"
MANIFEST = Path("data/manifest.json")   # file hashes already ingested

CHUNK_MAX_TOKENS = 8192
EMBED_BATCH_SIZE = 161
TOP_K = 10