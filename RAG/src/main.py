import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.lancedb import LanceDBVectorStore

import config

SYSTEM_PROMPT = (
    "You answer questions using only the provided context excerpts. "
    "Cite the source file and page for each claim. If the context does not "
    "contain the answer, say so plainly instead of guessing. If excerpts "
    "disagree with each other, point out the disagreement."
)

Settings.embed_model = OllamaEmbedding(model_name=config.EMBED_MODEL, base_url=config.OLLAMA_URL)
llm = Ollama(
    model=config.LLM_MODEL,
    base_url=config.OLLAMA_URL,
    context_window=8192,          # sent to Ollama as num_ctx
    request_timeout=300,
    keep_alive=-1,
    temperature=0.2,
)
store = LanceDBVectorStore(uri=config.LANCEDB_URI, table_name=config.TABLE_NAME)
retriever = VectorStoreIndex.from_vector_store(store).as_retriever(similarity_top_k=config.TOP_K)

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "rag-docs"
    messages: list[Message]
    stream: bool = False


async def build_messages(req: ChatRequest):
    question = next(m.content for m in reversed(req.messages) if m.role == "user")
    nodes = await retriever.aretrieve(question)
    context = "\n\n".join(
        f"[{i}] ({n.node.metadata['source']}, p.{n.node.metadata['pages']})\n{n.node.get_content()}"
        for i, n in enumerate(nodes, 1)
    )
    history = [ChatMessage(role=m.role, content=m.content) for m in req.messages[:-1] if m.role != "system"]
    msgs = [ChatMessage(role="system", content=f"{SYSTEM_PROMPT}\n\nContext:\n{context}")]
    msgs += history + [ChatMessage(role="user", content=question)]
    sources = sorted({f"{n.node.metadata['source']} p.{n.node.metadata['pages']}" for n in nodes})
    return msgs, sources


def chunk(cid, model, delta, finish=None):
    body = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(body)}\n\n"


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    msgs, sources = await build_messages(req)
    footer = "\n\nSources: " + "; ".join(sources)
    cid = f"chatcmpl-{uuid.uuid4().hex}"

    if not req.stream:
        resp = await llm.achat(msgs)
        return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": req.model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": resp.message.content + footer}}]}

    async def gen():
        yield chunk(cid, req.model, {"role": "assistant"})
        async for part in await llm.astream_chat(msgs):
            if part.delta:
                yield chunk(cid, req.model, {"content": part.delta})
        yield chunk(cid, req.model, {"content": footer})
        yield chunk(cid, req.model, {}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "rag-docs", "object": "model", "owned_by": "local"}]}


@app.get("/health")
async def health():
    return {"ok": True}