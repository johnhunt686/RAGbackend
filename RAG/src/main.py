import json
import time
import uuid

import httpx
import lancedb
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

import config

SYSTEM_PROMPT = """You answer questions about a library of research documents using tools.

Tools:
- search(query, source?): semantic search. Returns excerpts with a chunk id, source file and pages. Make several searches with different, specific wording when a question has multiple parts or the first results are weak.
- read(chunk_id, before, after): read a chunk in full with surrounding chunks. Use it when an excerpt is cut off, a table looks incomplete, or you need the text around a result.
- list_documents(): see which documents exist.
- find_text(text, source?): exact text match. Use it for specific numbers, names, acronyms or phrases.
- read_page(source, page): read everything on one page. Use it when results point to a page but the figures you need are not in the excerpt.

Rules:
- Always search before answering. Never answer from memory.
- Try at least two differently worded searches, or one search plus find_text, before concluding something is missing.
- Base every claim on what the tools returned and cite the source file and page.
- Never ask the user to provide documents. If you cannot find the answer, say which searches you ran and what the closest results covered.
- If excerpts disagree, point out the disagreement.
- Excerpts come from automatically parsed PDFs. Tables may be flattened and chart values or formulas may be missing; if the answer probably sits in a chart or table you cannot see, say so."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Semantic search over the documents. Returns excerpts with chunk ids, source and pages.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A specific question or descriptive phrase"},
            "source": {"type": "string", "description": "Optional: restrict to one document filename"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read one chunk in full plus its neighbors. chunk_id looks like 'file.pdf#12'.",
        "parameters": {"type": "object", "properties": {
            "chunk_id": {"type": "string"},
            "before": {"type": "integer", "description": "chunks before (default 1, max 4)"},
            "after": {"type": "integer", "description": "chunks after (default 1, max 4)"}},
            "required": ["chunk_id"]}}},
    {"type": "function", "function": {
        "name": "list_documents",
        "description": "List the available documents.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "find_text",
        "description": "Exact, case-insensitive text match across the documents. Use for specific terms, numbers, names or phrases that semantic search may miss.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "source": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "read_page",
        "description": "Read all text on one page of a document, e.g. after a search result points to a page.",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string", "description": "document filename (partial match ok)"},
            "page": {"type": "integer"}}, "required": ["source", "page"]}}},
]

# ---------------------------------------------------------------- data

Settings.embed_model = OllamaEmbedding(model_name=config.EMBED_MODEL, base_url=config.OLLAMA_URL)
store = LanceDBVectorStore(uri=config.LANCEDB_URI, table_name=config.TABLE_NAME)
retriever = VectorStoreIndex.from_vector_store(store).as_retriever(similarity_top_k=config.SEARCH_TOP_K * 3)


class Corpus:
    """In-memory copy of chunk text and metadata, used for neighbor lookup and read()."""

    def __init__(self):
        self.stamp = None
        self.chunks = {}   # (source, idx) -> {"text", "pages", "headings"}
        self.docs = {}     # source -> chunk count

    def refresh(self):
        stamp = config.MANIFEST.stat().st_mtime if config.MANIFEST.exists() else 0
        if stamp == self.stamp:
            return
        table = lancedb.connect(config.LANCEDB_URI).open_table(config.TABLE_NAME)
        try:
            rows = table.search().select(["text", "metadata"]).limit(10_000_000).to_list()
        except Exception:
            rows = table.to_pandas()[["text", "metadata"]].to_dict("records")
        self.chunks, self.docs = {}, {}
        for r in rows:
            m = r["metadata"]
            key = (m["source"], int(m["chunk_index"]))
            self.chunks[key] = {"text": r["text"], "pages": m.get("pages", ""), "headings": m.get("headings", "")}
            self.docs[m["source"]] = self.docs.get(m["source"], 0) + 1
        self.stamp = stamp


corpus = Corpus()


def render(src, idxs, used, match=None, score=None):
    chunks = [corpus.chunks[(src, i)] for i in idxs]
    pages = sorted({p for c in chunks for p in c["pages"].replace(" ", "").split(",") if p},
                   key=lambda p: int(p) if p.isdigit() else 0)
    page_txt = ", ".join(pages)
    for p in pages:
        if p.isdigit():
            used.add((src, int(p)))
    anchor = corpus.chunks.get((src, match), chunks[0])
    head = f"[{src}#{match}] {src} | p.{page_txt} | {anchor['headings']} | context chunks {idxs[0]}-{idxs[-1]}"
    if score is not None:
        head += f" | score {score:.2f}"
    return head + "\n" + "\n\n".join(c["text"] for c in chunks)


def cap(text):
    if len(text) > config.MAX_TOOL_CHARS:
        return text[: config.MAX_TOOL_CHARS] + "\n[result truncated; use read() on a chunk id for more]"
    return text


# ---------------------------------------------------------------- tools

async def tool_search(args, used):
    corpus.refresh()
    query = str(args.get("query", "")).strip()
    if not query:
        return "error: query is required"
    source = args.get("source")
    seen, blocks = set(), []
    for n in await retriever.aretrieve(query):
        md = n.node.metadata
        src, idx = md["source"], int(md["chunk_index"])
        if source and source.lower() not in src.lower():
            continue
        if (src, idx) in seen:
            continue
        idxs = [i for i in range(idx - config.NEIGHBORS, idx + config.NEIGHBORS + 1)
                if (src, i) in corpus.chunks and (src, i) not in seen]
        seen.update((src, i) for i in idxs)
        blocks.append(render(src, idxs, used, match=idx, score=n.score))
        if len(blocks) >= config.SEARCH_TOP_K:
            break
    return cap("\n\n=====\n\n".join(blocks)) if blocks else "no results"


def tool_read(args, used):
    corpus.refresh()
    src, _, num = str(args.get("chunk_id", "")).rpartition("#")
    if not num.isdigit() or (src, int(num)) not in corpus.chunks:
        return "error: unknown chunk id; use an id like 'file.pdf#12' from search results"
    idx = int(num)
    before = max(0, min(int(args.get("before", 1)), 4))
    after = max(0, min(int(args.get("after", 1)), 4))
    idxs = [i for i in range(idx - before, idx + after + 1) if (src, i) in corpus.chunks]
    return cap(render(src, idxs, used, match=idx))


def tool_list():
    corpus.refresh()
    return "\n".join(f"{s} ({n} chunks)" for s, n in sorted(corpus.docs.items())) or "no documents ingested"


async def run_tool(name, args, used):
    try:
        if isinstance(args, str):
            args = json.loads(args or "{}")
        if name == "search":
            return await tool_search(args, used)
        if name == "read":
            return tool_read(args, used)
        if name == "list_documents":
            return tool_list()
        if name == "find_text":
            return tool_find(args, used)
        if name == "read_page":
            return tool_read_page(args, used)
        return f"error: unknown tool {name}"
    except Exception as e:
        return f"error: {e}"


def tool_find(args, used):
    corpus.refresh()
    needle = str(args.get("text", "")).strip().lower()
    if not needle:
        return "error: text is required"
    source, hits = args.get("source"), []
    for (src, idx), c in sorted(corpus.chunks.items()):
        if source and source.lower() not in src.lower():
            continue
        pos = c["text"].lower().find(needle)
        if pos < 0:
            continue
        snip = c["text"][max(0, pos - 150): pos + len(needle) + 150].replace("\n", " ")
        hits.append(f"[{src}#{idx}] p.{c['pages']} | {c['headings']}\n...{snip}...")
        if len(hits) >= 8:
            break
    return cap("\n\n".join(hits)) if hits else "no exact matches (try different wording, or use search)"


def tool_read_page(args, used):
    corpus.refresh()
    name, page = str(args.get("source", "")), str(args.get("page", "")).strip()
    names = [s for s in corpus.docs if name.lower() in s.lower()]
    if len(names) != 1:
        return f"error: source must match exactly one document; available: {', '.join(sorted(corpus.docs))}"
    s = names[0]
    idxs = sorted(i for (d, i), c in corpus.chunks.items()
                  if d == s and page in c["pages"].replace(" ", "").split(","))
    if not idxs:
        return f"no chunks found for page {page} of {s}"
    return cap(render(s, idxs, used, match=idxs[0]))


# ---------------------------------------------------------------- agent loop

async def ollama_stream(client, messages, with_tools):
    body = {"model": config.LLM_MODEL, "messages": messages, "stream": True, "keep_alive": -1,
            "options": {"num_ctx": config.NUM_CTX, "temperature": 0.2}}
    if with_tools:
        body["tools"] = TOOLS
    async with client.stream("POST", f"{config.OLLAMA_URL}/api/chat", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line:
                d = json.loads(line)
                if d.get("done"):
                    print(f"[ollama] prompt_tokens={d.get('prompt_eval_count')} out={d.get('eval_count')}", flush=True)
                m = d.get("message", {})
                yield m.get("content", ""), m.get("tool_calls") or []


async def agent_events(history, used):
    """Yields ('token', text) for answer text and ('progress', text) for tool activity."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    async with httpx.AsyncClient(timeout=None) as client:
        for round_no in range(config.MAX_ROUNDS + 1):
            content, calls = "", []
            async for text, tcs in ollama_stream(client, messages, with_tools=round_no < config.MAX_ROUNDS):
                if text:
                    content += text
                    yield "token", text
                calls += tcs
            if not calls:
                return
            messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            for c in calls:
                name, args = c["function"]["name"], c["function"]["arguments"]
                yield "progress", f"\n*{name}: {json.dumps(args)[:100]}*\n\n"
                print(f"[tool] {name} {json.dumps(args)[:200]}", flush=True)
                messages.append({"role": "tool", "tool_name": name, "content": await run_tool(name, args, used)})


# ---------------------------------------------------------------- API

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "rag-docs"
    messages: list[Message]
    stream: bool = False


def sse(cid, model, delta, finish=None):
    body = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(body)}\n\n"


def ranges(nums):
    out, start, prev = [], None, None
    for n in sorted(set(nums)):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append((start, prev)); start = prev = n
    if start is not None:
        out.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


def footer(used):
    by = {}
    for src, p in used:
        by.setdefault(src, []).append(p)
    if not by:
        return ""
    return "\n\nSources: " + "; ".join(f"{s} p.{ranges(ps)}" for s, ps in sorted(by.items()))

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.messages if m.role != "system"]
    used, cid = set(), f"chatcmpl-{uuid.uuid4().hex}"

    if not req.stream:
        text = "".join([t async for kind, t in agent_events(history, used) if kind == "token"])
        return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": req.model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text + footer(used)}}]}

    async def gen():
        yield sse(cid, req.model, {"role": "assistant"})
        async for _, text in agent_events(history, used):
            yield sse(cid, req.model, {"content": text})
        yield sse(cid, req.model, {"content": footer(used)})
        yield sse(cid, req.model, {}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "rag-docs", "object": "model", "owned_by": "local"}]}


@app.get("/health")
async def health():
    return {"ok": True}