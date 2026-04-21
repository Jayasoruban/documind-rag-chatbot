# Lesson 6 — Production & What's Next

> ⏱ ~30 minutes · 📁 reference files: `src/config.py`, `Dockerfile`, `.streamlit/config.toml`

You now understand the whole RAG pipeline. This final lesson is about what separates an MVP from something a client is willing to pay real money for — the production concerns — and a clear roadmap for what to learn next.

## 1. Config via environment variables (12-factor)

From `src/config.py`:

```16:41:src/config.py
@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    chroma_persist_dir: Path
    collection_name: str

    @classmethod
    def load(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        return cls(
            openai_api_key=api_key,
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("TOP_K", "4")),
            chroma_persist_dir=(
                PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "./chroma_db").lstrip("./")
            ),
            collection_name=os.getenv("COLLECTION_NAME", "documind"),
        )
```

Principles baked in:

- **Every tunable knob is an env var.** Nothing is hardcoded. You can redeploy to a new client with a new `.env` file — no code changes.
- **Sensible defaults.** Missing env vars don't crash; they use defaults so dev setup is zero-friction.
- **Immutable settings** (`frozen=True`). Settings get loaded once at startup. No mutable globals.
- **Typed conversion.** `int(os.getenv("CHUNK_SIZE", "1000"))` — env vars are always strings, we convert once at the boundary.

This is called **12-factor config** (from the 12-factor app manifesto). Every serious backend follows this. In client-facing work, this design is what lets you deploy the same codebase to staging, prod, and on-prem environments with just env var changes.

## 2. Docker — why and how

Our Dockerfile gives you the production deploy story:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir uv && uv pip install --system -e .
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Two reasons containers matter for freelance work:

1. **"Works on my machine" is career-ending.** A client's ops team will want something they can deploy without understanding your Python setup. Docker = `docker build && docker run` and it works.
2. **Cloud deploy targets (Render, Railway, Fly.io, AWS ECS) all speak Docker.** A Dockerfile in your repo makes you deployable to any platform.

**What each line does:**
- `FROM python:3.12-slim` — small base image, production-friendly.
- `WORKDIR /app` — all commands run here.
- `COPY pyproject.toml .` first → `RUN pip install` → `COPY . .` — this ordering makes Docker cache the slow dependency layer. Code changes don't re-install dependencies.
- `EXPOSE 8501` — Streamlit's default port.
- `CMD [...]` — the command to run when the container starts.

## 3. The deployment options (ranked by ease)

For a portfolio project like DocuMind:

| Option | When to use | Cost | Notes |
|---|---|---|---|
| **Streamlit Community Cloud** | Portfolio demos, public. | Free | One-click from GitHub. The right answer for DocuMind. |
| **Render** | Private demos, client walkthroughs. | $7/mo Hobby | Auto-deploys from git; has persistent disks. |
| **Railway** | Same use case as Render, newer DX. | $5/mo | Good for demos, slightly less mature. |
| **Fly.io** | Multi-region or edge. | Free tier + pay-as-you-go | More DevOps knowledge needed. |
| **AWS / GCP / Azure** | Enterprise clients. | Depends | Use ECS/Cloud Run for containerized deploy. Worth learning once. |

**For DocuMind specifically:** Streamlit Cloud. 5 minutes. Free. URL looks like `documind-jayasoruban.streamlit.app`.

## 4. Error handling & observability — what we have and what's missing

Current state:
- The Streamlit UI wraps `stream_answer` in a try/except (Lesson 5).
- Config has `has_api_key` guard to fail clearly if the API key is missing.

What a production-grade version would add:

1. **Structured logging.** Use `structlog` or `logging` with JSON output. Log every query, retrieved chunk IDs, token usage, latency.
2. **LangSmith tracing.** Set two env vars (`LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY=...`) and every LCEL call auto-logs to a UI showing inputs, outputs, and latencies for every node. This is game-changing when debugging why an answer was wrong.
3. **Cost tracking.** OpenAI responses include token counts. Sum `prompt_tokens` and `completion_tokens` per query to build a cost dashboard. Clients love this.
4. **Rate limiting.** Wrap OpenAI calls in `tenacity` retries with exponential backoff for 429s.
5. **Eval pipeline.** Maintain a golden set of 30–50 questions with expected answers. Every time you change chunking/retrieval/prompts, run the eval to confirm nothing regressed. Tools: RAGAS, DeepEval, LangSmith evaluators.

## 5. Extensions that turn DocuMind into a product

If a client hires you to extend DocuMind, here's the ranked backlog by business value:

1. **Multi-user + authentication.** Each user sees only their own documents. Auth0, Supabase Auth, or Clerk. Add `user_id` to chunk metadata and filter retrievals.
2. **Reranker.** Retrieve 20 chunks with Chroma, rerank with Cohere Rerank API, keep top 4. Retrieval quality jumps noticeably.
3. **Hybrid search.** BM25 + vector, fused with RRF. Catches both exact keywords (SKUs, product names) and semantic matches.
4. **Query rewriting.** Use a cheap LLM to rewrite the user's question into 3 search variants. Retrieve for each. Merge. Better recall on ambiguous questions.
5. **Persistent chat history.** Per-user conversation threads saved to SQLite or Postgres.
6. **Document updates & versioning.** Hash chunks; only re-embed changed chunks when a document is re-uploaded.
7. **Streaming sources.** Show the retrieved chunks *before* the answer starts streaming — Perplexity-style.
8. **Evaluation dashboard.** Internal admin page showing answer quality metrics over time.
9. **Team-level permissions.** Shared workspaces, document ACLs.
10. **Slack/Teams integration.** Ship DocuMind as a chat bot in the user's existing tools, not just a standalone web app.

Each of these is a legit client ask. When you see one on Upwork, you know exactly what to scope.

## 6. What to learn next — the roadmap beyond RAG

You've now built a single-step RAG system: retrieve once, answer once. Modern AI systems go well beyond this. Here's the ordered learning path for Projects 2–6.

### Stage 1 — Agents (Project 2–3)
**Why:** single-step RAG can't handle multi-step reasoning ("research X, compare with Y, summarize trade-offs"). Agents loop — plan, act, observe, replan.

**Learn:**
- **LangGraph** — LangChain's state-machine-based agent framework. Industry standard. Use for any multi-step workflow.
- **CrewAI** — multi-agent collaboration (a "researcher" agent + "writer" agent + "editor" agent). Great for content generation pipelines.
- **Tool use / function calling** — how LLMs call external APIs, run Python, query databases. The foundation of everything agentic.

Project 2 (**ResearchCrew**): a multi-agent research assistant that does web search + retrieval + synthesis.

### Stage 2 — MCP (Project 4)
**Why:** Anthropic's Model Context Protocol is becoming the standard way LLMs integrate with external systems — Cursor, Claude Desktop, and more already support it. Knowing MCP is a differentiator right now because most engineers haven't caught up yet.

**Learn:** how to build MCP servers that expose tools/resources to any MCP-compatible client.

Project 4: a custom MCP server that exposes DocuMind as a tool to Claude Desktop.

### Stage 3 — Evaluation & Production (Project 5)
**Why:** clients pay 3–5x more for AI systems with measurable quality guarantees. "My system is accurate" means nothing; "my system scores 87% faithfulness on this eval set" sells.

**Learn:**
- **RAGAS** / **DeepEval** — eval frameworks.
- **LangSmith** — tracing + eval + prompt management.
- **A/B testing prompts** in production.
- **Guardrails** — `guardrails-ai` or `NeMo Guardrails` for input/output validation.

### Stage 4 — Fullstack AI products (Project 6)
**Why:** Streamlit is great for MVPs; customer-facing products need a real frontend. Clients who pay the most want end-to-end delivery.

**Learn:**
- **FastAPI** as an async API backend.
- **Next.js** with the App Router for the frontend.
- **Server-Sent Events** or **Vercel AI SDK** for streaming.
- **Stripe** for payments.
- **Supabase** or **Neon** for auth + Postgres.

Project 6: a productized AI SaaS with Stripe, auth, billing, the works.

## 7. Remember this

- **Env vars > hardcoded config.** Always. Redeployability is a force multiplier.
- **A Dockerfile in your repo** multiplies the number of clients who can deploy you.
- **Observability (LangSmith) is free debugging superpowers.** Turn it on from day one on real client work.
- **Evaluation sets** are the single biggest quality lever in production RAG.
- **The roadmap:** RAG → Agents → MCP → Eval/Prod → Fullstack. In that order.

## 8. Interview Q&A

**Q1: "How would you productionize this for a Fortune 500 client?"**
> Five things. (1) Swap Streamlit for Next.js + FastAPI for a real frontend and a proper API. (2) Multi-tenant isolation via `tenant_id` metadata filtering, backed by SSO. (3) Swap ChromaDB for Pinecone or Azure AI Search for scale and SLAs. (4) Swap OpenAI for Azure OpenAI or a private VPC-deployed LLM if data regulations require. (5) LangSmith tracing + RAGAS evals + structured logging + cost tracking in a dashboard.

**Q2: "How do you measure if a RAG system is 'good'?"**
> Three metrics: (a) **Retrieval recall** — of the chunks relevant to the question, how many are in the top-K retrieved? (b) **Faithfulness** — does the answer only claim things supported by retrieved context? (RAGAS measures this with an LLM-as-judge.) (c) **Answer relevance** — does the answer actually address the question? You need a golden set of 30–100 questions with expected answers/context; you measure these every time you change anything.

**Q3: "How do you handle data privacy for regulated clients?"**
> Three-layer approach. (1) Data residency: self-host everything — swap OpenAI for a local LLM via Ollama or vLLM, swap OpenAI embeddings for BGE or nomic-embed. Run ChromaDB or Qdrant inside the client's VPC. (2) PII redaction: run Presidio or a custom regex pipeline on documents before embedding. (3) Audit logging: every query, retrieved chunk, and response gets logged with user, timestamp, and source.

**Q4: "What's the total cost to run DocuMind for 1,000 users, 100 queries each per month?"**
> ~100K queries/month. Per query: ~$0.0001 embedding (question) + ~$0.0003 LLM call ≈ $0.0004. Total ≈ $40/month in API costs. One-time embedding cost for documents: depends on size, roughly $5–50 per user for a typical docset. Infra on Render: $7/mo per instance. Very fat margins at this scale.

**Q5: "What's the biggest mistake engineers new to LLMs make?"**
> Shipping without evaluation. They prompt-engineer until things "look good" in manual testing, ship, and then can't tell whether Monday's change made things better or worse — because there's no measurement. The fix: build a golden eval set *before* building the product. 30 questions, expected outputs, run them on every change. This one habit separates amateurs from professionals.

## 9. Final exercise — your graduation project

Pick one and do it over the next 2 days:

1. **Deploy DocuMind live.** Streamlit Cloud. Get a URL you can share on LinkedIn. Record a 60-second Loom demo.
2. **Add LangSmith tracing.** Set the two env vars, view traces, screenshot the dashboard, post about what you learned.
3. **Build a 10-question eval set.** Pick a single document (e.g., a research paper). Write 10 realistic questions with expected keywords. Write a `evaluate.py` script that runs all 10 and checks whether retrieved chunks contain the keywords. This is your first taste of AI eval.

Finish one of these → you're ready for Project 2.

## 10. Where to go from here

1. **Make sure DocuMind runs end-to-end on your machine** with a real OpenAI key.
2. **Deploy it** (Streamlit Cloud).
3. **Update your GitHub profile README** with the live URL and a GIF demo.
4. **Post on LinkedIn** — "I built and deployed my first production RAG system." Link the repo, link the live demo, link this LEARN folder. Show clients you don't just ship — you understand.
5. **Start Project 2 (ResearchCrew)** with the same mental framework: build → document → learn → deploy.

You've got 6 months of runway and now a real foundation. Build the next one deeper, not faster.

---

[← Lesson 5](./05-streaming-ui.md) &nbsp;·&nbsp; [Back to index](./README.md)
