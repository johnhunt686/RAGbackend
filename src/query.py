import sys

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

import config

Settings.embed_model = OllamaEmbedding(model_name=config.EMBED_MODEL, base_url=config.OLLAMA_URL)
store = LanceDBVectorStore(uri=config.LANCEDB_URI, table_name=config.TABLE_NAME)
retriever = VectorStoreIndex.from_vector_store(store).as_retriever(similarity_top_k=config.TOP_K)

for r in retriever.retrieve(" ".join(sys.argv[1:])):
    m = r.node.metadata
    print(f"\n--- score {r.score:.3f} | {m['source']} p.{m['pages']} | {m['headings']}")
    print(r.node.get_content()[:600])