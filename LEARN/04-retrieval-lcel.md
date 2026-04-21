# Lesson 4 — Retrieval & LCEL Chains

> ⏱ ~40 minutes · 📁 reference file: `src/rag_chain.py`

This is where all the ingredients from Lessons 1–3 come together. We retrieve relevant chunks, wrap them into a prompt, and send the whole thing to the LLM. The flow feels magical but is entirely mechanical — once you see the moving parts, it's obvious.

## 1. The retrieval step

From `src/rag_chain.py`:

```64:68:src/rag_chain.py
def retrieve(question: str, *, top_k: int | None = None) -> list[Document]:
    """Fetch the most relevant chunks for the question."""
    store = get_vector_store()
    k = top_k or settings.top_k
    return store.similarity_search(question, k=k)
```

What this does under the hood:

1. Chroma calls `OpenAIEmbeddings.embed_query(question)` → gets a 1536-dim vector.
2. Walks the HNSW index to find the `k` closest stored vectors.
3. Returns a `list[Document]` (each with `page_content` + `metadata`).

We default to `k=4`. Why?

- **k=1:** fragile. If the best chunk is slightly off, the answer has nothing to work from.
- **k=4:** sweet spot. Covers the likely answer plus related context.
- **k=10+:** context becomes noisy, cost goes up, LLM gets distracted.

Once you add a **reranker** (Cohere Rerank, Voyage Rerank), the pattern becomes: retrieve 20 → rerank to 4. Retrieval is cheap; rerankers are small cross-encoder models that deeply compare the query to each chunk and surface the real winners.

## 2. Prompts — where most of your output quality is decided

Beginners think the LLM does the work. Pros know **the prompt does 60% of the work**.

Our system prompt:

```15:23:src/rag_chain.py
SYSTEM_PROMPT = """You are DocuMind, a precise and helpful assistant that answers \
questions strictly using the provided context from the user's documents.

Rules:
- Ground every answer in the provided context.
- If the answer is not in the context, say so clearly. Do not invent facts.
- Cite sources inline using [filename, page N] where available.
- Be concise, structured, and use markdown for lists and code.
"""
```

Every line here is deliberate:

- **"strictly using the provided context"** — the anti-hallucination clause.
- **"If the answer is not in the context, say so clearly"** — explicit permission to say "I don't know." Without this, LLMs often fabricate.
- **"Cite sources inline"** — makes answers auditable. Users trust cited answers far more.
- **"Concise, structured, markdown"** — style control. Prevents 500-word essays when a list of 3 bullets is better.

Our user prompt:

```25:33:src/rag_chain.py
USER_PROMPT = """Context from the user's documents:
---
{context}
---

Question: {question}

Answer using only the context above. Include citations like [filename, page N].
"""
```

The `---` delimiters and the word "Context" help the LLM cleanly separate retrieved material from the question. This matters — without delimiters, LLMs sometimes confuse retrieved text with instructions (a mild form of prompt injection vulnerability).

## 3. How retrieved chunks get into the prompt

The helper that glues it:

```43:52:src/rag_chain.py
def _format_context(docs: list[Document]) -> str:
    """Render retrieved chunks with source labels the LLM can cite."""
    blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        filename = meta.get("filename") or meta.get("source") or f"source_{idx}"
        page = meta.get("page")
        header = f"[{filename}" + (f", page {page + 1}" if page is not None else "") + "]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)
```

Result looks like this when passed to the LLM:

```
[refund-policy.pdf, page 3]
Enterprise customers are eligible for a full refund within 60 days of purchase...

[refund-policy.pdf, page 4]
For seat-based licenses, refunds are prorated based on unused seat-months...

[terms-of-service.pdf, page 12]
All refund requests must be submitted via the billing portal...
```

Each chunk has its source label **right above the text**. When the LLM writes the answer, it can quote the label for citations. This is the cheapest, most reliable citation trick in RAG — no post-processing needed.

## 4. Enter LCEL — the pipe operator

LangChain Expression Language (LCEL) is a DSL for composing AI pipelines. You write them using the `|` (pipe) operator, just like Unix shell pipes.

```78:81:src/rag_chain.py
    chain = prompt | llm
    response = chain.invoke(
        {"context": _format_context(docs), "question": question}
    )
```

What's happening:

- `prompt` is a `ChatPromptTemplate` (a Runnable).
- `llm` is a `ChatOpenAI` (also a Runnable).
- `prompt | llm` creates a new Runnable that: takes a dict input → formats the prompt → sends to the LLM → returns the response.

Every LCEL Runnable supports these methods:

| Method | What it does |
|---|---|
| `.invoke(input)` | Run synchronously. Returns one output. |
| `.stream(input)` | Run and yield tokens as they arrive. |
| `.batch(inputs)` | Run on a list in parallel. Useful for evaluation. |
| `.ainvoke(input)` | Async version. |

Because each piece conforms to the Runnable interface, you can chain as many as you want:

```python
retriever | prompt | llm | output_parser
```

That one-liner is a valid RAG pipeline. LCEL is less magic and more "every component has the same shape."

### Why LCEL matters

1. **Streaming for free.** The same `.stream()` method works on every Runnable. Swap `.invoke` → `.stream` and you get token-by-token output.
2. **Observability for free.** Wire up LangSmith and every `.invoke`/`.stream` auto-logs inputs, outputs, and latencies.
3. **Parallelization.** `RunnableParallel` lets you run retriever + web search simultaneously, then merge results.
4. **Swappable components.** Need Claude instead of GPT? Replace `ChatOpenAI` with `ChatAnthropic`. Nothing else changes.

## 5. The `ChatPromptTemplate` pattern

```74:76:src/rag_chain.py
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
```

A `ChatPromptTemplate` is a list of role/template pairs. Roles are:

- **`system`** — sets the model's persona, rules, constraints. Highest authority.
- **`human`** (aka `user`) — the end-user's input. Gets formatted with `{variable}` substitutions.
- **`ai`** (aka `assistant`) — a previous AI response. Used for multi-turn chat.

When you call `.invoke({"context": ..., "question": ...})`, the template fills in `{context}` and `{question}` in the USER_PROMPT string, then assembles the chat messages and sends them to the API.

### Quick mental model

```
[("system", "You are DocuMind..."),
 ("human", "Context: {context}\n\nQuestion: {question}")]

    + {"context": "<retrieved chunks>", "question": "refund policy?"}
    ↓
    ↓ .invoke()
    ↓
[{"role": "system", "content": "You are DocuMind..."},
 {"role": "user",   "content": "Context: ...\n\nQuestion: refund policy?"}]

    → OpenAI Chat Completions API
    → response
```

## 6. Putting it all together: the full `answer()` flow

```71:82:src/rag_chain.py
def answer(question: str, *, model: str | None = None, top_k: int | None = None) -> RagAnswer:
    """Run a one-shot RAG query and return the answer with its source chunks."""
    docs = retrieve(question, top_k=top_k)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    llm = _build_llm(model=model)
    chain = prompt | llm
    response = chain.invoke(
        {"context": _format_context(docs), "question": question}
    )
    return RagAnswer(question=question, answer=response.content, sources=docs)
```

Trace it:

1. **`retrieve(question)`** — top-K chunks from Chroma.
2. **Build `prompt`** — template that will combine system + user messages.
3. **Build `llm`** — a configured ChatOpenAI instance (temperature 0.1 for deterministic citations).
4. **`chain = prompt | llm`** — compose.
5. **`chain.invoke({...})`** — format template with context & question, send to OpenAI, get response.
6. **Return `RagAnswer`** — a small dataclass packaging the answer with its sources so the UI can display citations.

Notice how `answer()` is **18 lines of code** and implements a complete production-grade RAG query. LCEL's composition + LangChain's abstractions do the heavy lifting.

## 7. Why temperature = 0.1 for RAG

From `_build_llm`:

```55:61:src/rag_chain.py
def _build_llm(model: str | None = None, *, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
        streaming=streaming,
    )
```

**Temperature** controls randomness in LLM output. 0 = deterministic, 1 = creative, up to 2.

- **Creative tasks** (marketing copy, brainstorming) → 0.7–1.0.
- **Factual tasks** (RAG, summaries, code) → 0.0–0.2.

We use 0.1 — near-deterministic but with just enough variation to avoid the occasional weird repetition artifact you sometimes get at temp=0.

## 8. Remember this

- **Retrieval:** embed the question → search vector store → top-K chunks.
- **Prompts:** system prompt sets rules; user prompt carries context + question with clear delimiters.
- **LCEL pipe `|`:** composes Runnables. Same interface gives you `.invoke`, `.stream`, `.batch`, async.
- **Source labels inside the prompt** = cheap, reliable citations.
- **Low temperature for RAG** so answers are consistent and grounded.

## 9. Interview Q&A

**Q1: "Walk me through what happens when a user asks a question."**
> (1) The question is embedded into a 1536-dim vector. (2) Chroma finds the top-4 closest chunks. (3) We build a prompt with a system message setting the rules and a user message containing the retrieved chunks + the question, with clear delimiters. (4) The prompt is sent to `gpt-4o-mini` at temperature 0.1. (5) We return the answer along with the source chunks so the UI can show citations.

**Q2: "What's LCEL and why use it?"**
> LangChain Expression Language — the pipe syntax that composes AI components into pipelines. Every component (prompts, LLMs, retrievers, parsers) implements the Runnable interface, which means each one supports `.invoke`, `.stream`, `.batch`, and async automatically. You get streaming and observability for free, and components are trivially swappable.

**Q3: "How do you prevent hallucinations in RAG?"**
> Four layers. (1) A strict system prompt that says "answer only from context; otherwise say 'I don't know'." (2) Low temperature. (3) Explicit citation requirement, so the LLM's answer is anchored to specific labeled chunks. (4) Post-hoc evaluation with a framework like RAGAS that measures answer faithfulness against retrieved chunks. DocuMind has the first three; production projects also add the fourth.

**Q4: "How would you improve answer quality on a hard domain?"**
> Three levers in order of impact: (1) **better retrieval** — add a reranker so the top-4 are actually the right 4 (biggest win usually). (2) **query rewriting** — use a cheap LLM to expand the user's question into 3 search-optimized variants, retrieve for each, merge. (3) **few-shot examples in the prompt** — show 2–3 ideal Q&A pairs for the domain so the model adopts the answer style.

**Q5: "How do citations work?"**
> We prepend each retrieved chunk with a header like `[filename.pdf, page 3]` before sending to the LLM. The system prompt asks the LLM to include those labels inline. Because the labels appear in the retrieved context, the model copies them naturally — no post-processing or JSON mode needed. In production we'd also validate that every `[label]` in the answer maps to a real source, and reject answers with fabricated citations.

## 10. Exercise

1. **Inspect a prompt.** Add a one-line print before `chain.invoke(...)` in `answer()`:
   ```python
   print(prompt.format(context=_format_context(docs), question=question))
   ```
   Run a query. Read the full prompt the LLM sees. This demystifies everything.

2. **Try different top-K values.** Change `TOP_K=1` and `TOP_K=10` in `.env`. Ask the same question. Look at the retrieved chunks and the answers. Notice how low-K misses context and high-K adds noise.

3. **Add a query rewriter.** Before `retrieve()`, add a quick LLM call that rewrites the user's question into 3 variants. Retrieve for each, merge results, dedupe by chunk ID, keep top-4. You just implemented a technique used by ChatGPT's browsing mode.

4. **Swap to Anthropic.** Replace `from langchain_openai import ChatOpenAI` with `ChatAnthropic` from `langchain_anthropic`. No other code change needed. Proves LCEL's swappability to yourself.

---

[← Lesson 3](./03-embeddings-vectors.md) &nbsp;·&nbsp; [Next: Lesson 5 — Streaming & Streamlit UI →](./05-streaming-ui.md)
