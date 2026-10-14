import hashlib
import json
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

import config

META_KEYS = ["source", "pages", "headings", "chunk_index"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(config.MANIFEST.read_text()) if config.MANIFEST.exists() else {}


def save_manifest(m: dict) -> None:
    config.MANIFEST.write_text(json.dumps(m, indent=2))


def make_node(text: str, path: Path, i: int, pages: str = "", headings: str = "") -> TextNode:
    return TextNode(
        text=text,
        metadata={"source": path.name, "pages": pages, "headings": headings, "chunk_index": i},
        # keep metadata out of the embedded text; headings are already in the chunk text
        excluded_embed_metadata_keys=META_KEYS,
    )


def nodes_from_docling(path: Path, converter, chunker) -> list[TextNode]:
    doc = converter.convert(str(path)).document
    nodes = []
    for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        pages = sorted({p.page_no for item in chunk.meta.doc_items for p in item.prov})
        nodes.append(make_node(
            text=chunker.contextualize(chunk=chunk),   # heading path + body
            path=path, i=i,
            pages=", ".join(map(str, pages)),
            headings=" > ".join(chunk.meta.headings or []),
        ))
    return nodes


def nodes_from_text(path: Path) -> list[TextNode]:
    splitter = SentenceSplitter(chunk_size=config.CHUNK_MAX_TOKENS, chunk_overlap=50)
    return [make_node(t, path, i) for i, t in enumerate(splitter.split_text(path.read_text()))]


def main() -> None:
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBED_MODEL,
        base_url=config.OLLAMA_URL,
        embed_batch_size=config.EMBED_BATCH_SIZE,
    )
    store = LanceDBVectorStore(uri=config.LANCEDB_URI, table_name=config.TABLE_NAME)
    index = VectorStoreIndex.from_vector_store(store)

    converter = DocumentConverter()                    # PDFs and markdown
    chunker = HybridChunker(tokenizer=config.TOKENIZER, max_tokens=config.CHUNK_MAX_TOKENS)

    manifest = load_manifest()
    files = sorted(p for p in config.DOCS_DIR.rglob("*")
                   if p.suffix.lower() in {".pdf", ".md", ".txt"})

    for path in files:
        digest = sha256(path)
        if manifest.get(str(path)) == digest:
            print(f"skip   {path.name}")
            continue
        print(f"ingest {path.name} ...", flush=True)
        if path.suffix.lower() == ".txt":
            nodes = nodes_from_text(path)
        else:
            nodes = nodes_from_docling(path, converter, chunker)
        index.insert_nodes(nodes)                      # embeds via Ollama, writes to LanceDB
        manifest[str(path)] = digest
        save_manifest(manifest)                        # saved per file, so a crash loses little
        print(f"       {len(nodes)} chunks")


if __name__ == "__main__":
    main()