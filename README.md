# Internal HR/IT Support Agent

[![CI](https://github.com/mgetech/internal-support-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/mgetech/internal-support-agent/actions/workflows/ci.yml)

An internal employee support agent for HR and IT — the AskHR / internal-helpdesk class
of enterprise agent, built from scratch. It is designed to resolve requests like "how
many vacation days do I have left?", "can you book me off next week?" and "is the VPN
down, or do I need a ticket?" by querying the requesting employee's own records and
searching company policy with **hybrid retrieval** — then terminating in exactly one
of four outcomes: **resolve**, **propose an action** into a human approval queue,
**escalate** to a human, or **refuse with a citation**. Every request also produces a
**Decision Record**: a persisted, structured account of why the agent did what it did.
All data is synthetic.

**Status: early.** Project scaffolding and the database schema are in place; the agent
does not handle requests yet. The table below marks what is wired and what is planned.

## Stack

| Layer | Choice | Status |
|---|---|---|
| Language | Python 3.12 | In place |
| DB | PostgreSQL 16 + **pgvector** | In place |
| Tests | pytest, incl. DB-backed tests against the compose/CI Postgres | In place |
| CI | GitHub Actions — ruff + pytest always; eval gate when model secrets are configured | Partial — lint and tests |
| Packaging | Docker + docker-compose (db, api, ui) | Partial — db only |
| Orchestration | **LangGraph** | Planned |
| Model inference | **Azure OpenAI via AI Foundry** — a capable deployment for the agent loop, a small one for classification and claim extraction, `text-embedding-3-small` (1536-dim) | Planned |
| API | FastAPI | Planned |
| Retrieval | Metadata pre-filtering → hybrid pgvector cosine + Postgres BM25, reciprocal-rank fusion (k=60), top-5 | Planned |
| Chunking | Pluggable `Chunker` interface; structural (heading-aware) default, chosen by ablation against fixed-size, recursive and semantic | Planned |
| MCP | Official Python MCP SDK (FastMCP) exposing the tool belt; stdio + streamable HTTP | Planned |
| Eval tooling | Own deterministic harness (gates, trajectory, retrieval) + **RAGAS** for answer quality | Planned |
| Observability | **Langfuse** (cloud keys via `.env`; self-hostable for residency) | Planned |
| UI | Streamlit — chat, approval queue, decision records, cost | Planned |

## Roadmap

Rough build order; the table above says what exists today.

- **Walking skeleton** — deterministic synthetic HR/IT data and a heading-aware chunker;
  session-bound identity and a full audit trail; the typed tool belt, with its
  access-control tests written before the tools they constrain; the agent graph,
  citation verifier and Decision Records; role-checked human approval behind a REST
  API; a Streamlit review surface.
- **Evaluation gate** — scenarios with authored expected outcomes, run n=3 and wired
  into CI, so a change that makes the agent behave worse fails the build. Prompts are
  versioned config: a wording change without a version bump is a test failure.
- **Depth** — the same tool belt over MCP, with identity bound at session
  initialization rather than passed as an argument; adversarial scenarios (prompt
  injection, colleague-data probing, approval bypass); retrieval evaluation and the
  chunking ablation, with tables published.
- **Governance** — DSAR export, retention purge, and a database-level assertion that no
  record was ever written without passing through human approval.
- **Observability and feedback** — per-node traces with cost and latency budgets, a
  usage dashboard, and a path that turns a thumbs-down into a new eval scenario.