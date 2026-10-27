from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434"
LLM_MODEL = "qwen2.5:32b"
EMBED_MODEL = "bge-m3"
TOKENIZER = "BAAI/bge-m3"        # used by the chunker to count tokens
ENRICH_FORMULAS = False   # True converts equations to LaTeX; slow on CPU, only worth it for math-heavy papers

DOCS_DIR = Path("data/docs")
LANCEDB_URI = "data/lancedb"
TABLE_NAME = "chunks"
MANIFEST = Path("data/manifest.json")   # file hashes already ingested

CHUNK_MAX_TOKENS = 8192
EMBED_BATCH_SIZE = 161
TOP_K = 10

NUM_CTX = 32768            # context sent to Ollama; the main headroom knob
SEARCH_TOP_K = 6          # distinct hits returned per search call
NEIGHBORS = 1             # chunks included before and after each hit
MAX_ROUNDS = 5            # tool-call rounds before the model must answer
MAX_TOOL_CHARS = 12000    # cap on a single tool result (~3k tokens)