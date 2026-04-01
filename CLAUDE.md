# WGSN Agentic Search System — Project Rules

## What We Are Building

A multi-modal agentic search system over a mixed-media archive (text documents + images + metadata).
This is a **search problem first**, not a conversational AI problem.
The deliverable is an architecture document + working code demonstrating the design.

## Design Principles

- **Search-first**: every design decision must serve retrieval quality before UX
- **Grounded outputs only**: the synthesis agent must cite retrieved evidence; no fabricated claims
- **Deterministic orchestration**: sequential agent invocation by default; parallelism only when explicitly justified
- **Separate, then fuse**: text and image embeddings live in separate vector spaces; fusion happens at ranking time via late fusion, not at indexing time
- **Observable by default**: every agent emits structured traces (latency, token count, retrieval scores)

## Tech Stack (locked unless justified otherwise)

| Component | Choice | Reason |
|---|---|---|
| LLM | `gpt-4o-mini` (OpenAI) | Cost-efficient, structured output, vision capable |
| Text Embeddings | `text-embedding-3-small` (OpenAI) | 1536-dim, strong quality/cost ratio |
| Image Embeddings | CLIP `ViT-L/14` via `sentence-transformers` (local) | OpenAI has no CLIP API — runs in dedicated `embedder` service |
| Vector DB | PostgreSQL 16 + pgvector (HNSW) | Single DB for relational + vector; sufficient at this scale |
| Hybrid text search | PostgreSQL `tsvector` + pgvector RRF | Native, no extra service |
| API | FastAPI + WebSockets | Async, streaming-native |
| Frontend | React (nginx container) | WebSocket client for streaming |
| Containers | Docker Compose | Service isolation, shared `wsgn-net` network |
| Observability | Prometheus + Loki + Grafana (PLG stack) | Full open-source triad; custom metrics cover all LLM eval needs |

## Agent Contracts (never break these)

Each agent must:
- Accept a typed `Input` dataclass and return a typed `Output` dataclass
- Log its own latency, input token count (if LLM-backed), and result count
- Raise a typed exception on failure — no silent fallbacks that swallow errors
- Never call another agent directly — the orchestrator owns invocation order

## Code Rules

- Python 3.11+, strict type hints throughout
- Pydantic v2 for all data models (Input/Output contracts)
- No global state; agents are stateless classes instantiated by the orchestrator
- Unit tests mock the vector DB and LLM; integration tests hit real services
- No `print()` for logging — use `logging` with structured JSON formatter
- Environment variables via `python-dotenv`; never hardcode keys

## Code Style — Strictly Enforced

- **No comments in code.** Code must be self-explanatory through naming. If a comment feels necessary, rename the variable or extract a function instead.
- **No docstrings** unless the function is part of a public API surface.
- **No inline type annotations as comments** (e.g. `# type: ignore` is allowed only with a specific error code and a reason on the same line).
- **No helper utilities for one-off operations.** Three similar lines of code is better than a premature abstraction.
- **No backwards-compatibility shims** — we are not maintaining a public API; just change the code.
- **No unused imports, variables, or parameters** — remove them entirely.
- Function and variable names must be unambiguous without context (e.g. `normalised_query` not `nq`, `text_chunks` not `chunks`).
- Functions do one thing. If a function needs a comment to explain what a section does, split it.
- Max function length: 40 lines. If it exceeds this, decompose it.

## Isolation Rules

- Each agent is a self-contained module — it imports only from `src/models/` and the standard library / approved third-party libs.
- Agents never import from other agents. Cross-agent data flow goes through the orchestrator only.
- No shared mutable state between agents — pass data explicitly via typed contracts.
- Tests are fully isolated: each test sets up its own fixtures, no shared test state across test files.
- Every external call (LLM, vector DB, embedding model) is behind an interface that can be swapped for a mock in tests.
- `.env` is never committed; `.env.example` documents all required keys.

## What Claude Must Never Do Unprompted

- Do not refactor code that was not part of the task.
- Do not add error handling for scenarios that cannot happen given the typed contracts.
- Do not add features or configurability beyond what the current task requires.
- Do not create new files when editing an existing one suffices.
- Do not add logging statements beyond what the agent contract specifies.
- Do not suggest or implement backwards-compatibility for internal interfaces.

## Anti-Patterns — Never Do These

- Do not embed text and images into the same vector space without a cross-modal bridge
- Do not generate an answer before retrieval completes
- Do not allow the synthesis agent to introduce facts not present in the retrieved context
- Do not skip provenance tracking — every retrieved chunk must carry `doc_id`, `source`, `score`
- Do not add retry loops without exponential backoff + jitter

## File Layout

```
wsgn/
  README.md              # Architecture document with diagram
  CLAUDE.md              # This file
  docker-compose.yml
  .env.example
  api/                   # FastAPI service
    Dockerfile
    src/
      agents/
        query_intent.py
        text_retriever.py
        image_retriever.py
        ranker.py
        synthesizer.py
      models/            # Pydantic contracts (shared across agents)
      orchestrator.py
      observability.py
      main.py            # FastAPI app + WebSocket endpoint
  embedder/              # CLIP + text embedding service
    Dockerfile
    src/
      main.py
  react/                 # Frontend
    Dockerfile
    src/
  infra/
    prometheus/
      prometheus.yml
    loki/
      loki-config.yml
    grafana/
      dashboards/
  tests/
    unit/
    integration/
    eval/
      cases.jsonl
      baseline_metrics.json
  .claude/agents/        # Claude Code subagent definitions
```

## Evaluation — Always Measure These

1. `retrieval_recall_at_k` — ground truth hit in top-K results (text + image separately)
2. `mrr` — mean reciprocal rank across result sets
3. `hallucination_rate` — LLM-as-judge check: claims vs retrieved evidence
4. `e2e_latency_p95` — full pipeline latency at 95th percentile
5. `citation_coverage` — % of synthesis output sentences with a source citation

Regressions = any metric degrades >5% vs baseline on the eval dataset.
