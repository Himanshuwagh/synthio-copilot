# Pharma Sales Co-pilot

**[Live Demo: Try it here](https://synthio-copilot.onrender.com)**  (*Cold start: might take few seconds to start first time*)

This project is a fast, conversational intelligence co-pilot designed for pharmaceutical sales representatives. It allows field reps to ask natural language questions and instantly get accurate insights derived from complex, relational healthcare datasets.

## What makes this architecture different

Most Text-to-SQL copilots stop at "generate SQL → run → show answer."  
This one has five extra layers designed for real-world reliability:

| Layer | What it does | Why it matters |
|---|---|---|
| **Query rewriter** | Resolves vague follow-ups ("what about last month?") into standalone questions before touching SQL | Prevents bad SQL from misinterpreted context |
| **Intent classifier** | Detects if the new question just wants elaboration on existing data vs. a fresh DB query | Skips unnecessary SQL round-trips |
| **Ambiguity detector** | Regex-gates an LLM check on questions like "show me top HCPs" — surfaces metric alternatives (NRx? TRx? visits?) to the user | Avoids silently answering the wrong question |
| **Self-correcting SQL loop** | On failure → feeds error back to LLM for a rewrite. On empty result → runs diagnostics + rewrites with relaxed filters | Handles hallucinated joins & jargon (e.g. "rheum docs") transparently |
| **LRU answer cache** | SHA-256 keys last 4 turns + schema; bounded `OrderedDict` per session | Multi-user safe; avoids redundant LLM calls on repeat questions |

**The result:** a rep can ask "show me my top rheumatologists," get a clarifying prompt ("by NRx or visits?"), follow up with "why is Dr. Smith so high?" — and the system handles all of it without re-querying the DB or confusing one rep's session with another's.

## High-Level Backend Architecture

The backend is deliberately designed to be robust, perfectly suited to the task, and extremely easy to read—avoiding the trap of over-engineering.

- **FastAPI (The Web Framework):** Provides lightning-fast REST endpoints (`/chat`, `/history`) and serves the lightweight, local-first web interface seamlessly.
- **DuckDB (The Analytical Engine):** Chosen instead of typical relational databases due to its exceptional speed in performing complex SQL aggregations locally and entirely in memory. It dynamically parses our structured CSV data on startup.
- **Python-Native Agentic Loop (Core Reasoning):** Instead of adopting heavy frameworks like LangChain or LangGraph, the agent uses a clean, transparent Python loop to reason, run queries, and synthesize data. For a straightforward sequence (classify → loop SQL → synthesize), native Python provides a perfectly readable state machine with zero abstraction overhead.
- **LLM Engine (Text-to-SQL & NLU):** Handles interpreting human intent and robust Text-to-SQL logic. By strictly prompting the LLM and providing it runtime schema context, it natively traverses complex schemas.
- **Supabase (Context Management):** Tracks per-rep chat history remotely. This powers context-aware, turn-by-turn conversations—an essential feature for reps who ask sequential follow-up questions.

**Why no Vector DB?** 
Vector pipelines answer *"what document is most similar to this text?"* Our users ask for exact math ("What is the SUM of NRx...?"). You cannot do arithmetic with fuzzy semantic search! A purely structured, self-correcting Text-to-SQL pipeline is vastly more accurate for this type of data.

## Challenges Faced (And How They Were Solved)

Building AI for structured Pharma data presents several unique hurdles.

1. **SQL Hallucinations & Joins:** Joining fact tables and dimension tables can lead to cartesian products (the "fan-out" problem), causing wildly inflated prescription counts if the LLM hallucinates a join key. 
   * **The Solution:** We built a self-correcting execution loop. If the DuckDB execution fails or throws a syntax error, the agent feeds the raw error back to the LLM and asks it to debug and rewrite the query—fixing it invisibly before the user even sees it.
2. **Jargon and Entity Resolution:** Field reps use heavy slang (like "rheum docs" instead of "Rheumatologist") and misspell drug names (e.g. "Gaziva" instead of "Gazyva").
   * **The Solution:** The agent leverages careful prompt engineering and domain-centric system prompts so the LLM understands standard pharmaceutical vernacular and maps it to the precise schema constraints.
3. **Ambiguous Business Logic:** A rep asking "Show me my top docs" is painfully ambiguous. Do they mean top by NRx? Total volume? 
   * **The Solution:** Handled through conversational design where the agent explicitly specifies which assumptions it made to derive its answer (e.g., assuming NRx volume).

## Path to Production (Scaling with More Data)

To move this from a powerful assessment prototype to a true enterprise-grade reporting system querying millions of rows across thousands of territories, we would transition to:

1. **A True Semantic Layer (dbt / Cube.js):**
   Rather than letting the LLM write raw SQL directly against raw fact tables, it would query pre-defined semantic metrics (e.g., `SELECT * from rep_hcp_monthly_summary`). This hardcodes business logic safely and completely eliminates the risk of bad math.
2. **Schema RAG (Retrieval-Augmented Generation):**
   When dealing with an enterprise warehouse of 500+ tables, feeding the whole schema to the LLM's context window wastes tokens and degrades accuracy. We would use a Vector Search to retrieve *only* the relevant table definitions (DDLs) specific to the user's question before generating the query.
3. **Row-Level Security (RLS) & Compliance:**
   Production reps must never see another territory's prescriptions. Security filters would be aggressively clamped at the API or Database layer by automatically appending a hardcoded `WHERE territory_id = X` to every executed LLM query, keeping data strictly walled off.
