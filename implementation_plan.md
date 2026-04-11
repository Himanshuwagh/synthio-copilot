# Implementation Plan — Pharma Sales Co-pilot (Terminal)

## What We're Building

A terminal REPL where a pharma sales rep logs in by entering their rep ID,
then asks questions in plain English. The system uses their territory + conversation
history to give accurate, context-aware, data-backed answers.

---

## Project Structure

```
synthio-assignment/
│
├── data/                         ← all 9 original CSVs (already exist, don't touch)
│   ├── rep_dim.csv
│   ├── territory_dim.csv
│   ├── account_dim.csv
│   ├── hcp_dim.csv
│   ├── date_dim.csv
│   ├── fact_rx.csv
│   ├── fact_rep_activity.csv
│   ├── fact_ln_metrics.csv
│   └── fact_payor_mix.csv
│
├── conversations/                ← auto-created on first run, one CSV per rep
│   ├── rep_1.csv
│   ├── rep_2.csv
│   └── ...
│
├── main.py                       ← entry point, terminal REPL
├── db.py                         ← DuckDB setup + view creation
├── agent.py                      ← planner + executor + synthesizer
├── context.py                    ← read/write rep conversation history
├── prompts.py                    ← all LLM prompt strings (no logic here)
└── requirements.txt
```

---

## File-by-File Breakdown

---

### `requirements.txt`

```
duckdb
ollama
python-dotenv
```

That's it. No LangChain. No vector stores. No OpenAI API. Raw Python + local model.

> **Ollama setup:**
> ```bash
> # Install Ollama (mac)
> brew install ollama
>
> # Pull the model once
> ollama pull llama3.1:8b
>
> # Start Ollama server (keep running in background)
> ollama serve
> ```
> In `agent.py`, `llm_call()` uses the `ollama` library directly:
> ```python
> import ollama
> response = ollama.chat(
>     model='llama3.1:8b',
>     messages=[
>         {'role': 'system', 'content': system},
>         {'role': 'user',   'content': user}
>     ]
> )
> return response['message']['content']
> ```
> No API key. No base_url tricks. No cost per call.

---

### `db.py` — Database Setup

**Responsibility:** Load all 9 CSVs into DuckDB in-memory at startup.
Create 2 pre-materialized views. Also export a `get_schema()` function
that dumps the real column names at runtime for injection into prompts.

#### What it exports:
```python
def get_connection() -> duckdb.DuckDBPyConnection
    # loads all CSVs and creates views
    # returns the connection object

def get_schema(conn) -> str
    # queries DuckDB's information_schema at runtime
    # returns a formatted string of all tables + views with their actual columns
    # injected into SYSTEM_PROMPT instead of hardcoded column names
    #
    # Why: if a column name is wrong in the hardcoded prompt, the LLM
    # generates silently broken SQL with no error until runtime.
    # This is cheap insurance — runs once at startup.
    #
    # Output format:
    # TABLE rep_hcp_monthly: rep_id, rep_name, hcp_id, hcp_name, specialty, ...
    # TABLE hcp_rx_monthly: hcp_id, hcp_name, specialty, tier, month, trx_cnt, ...
    # TABLE fact_payor_mix: account_id, date_id, payor_type, pct_of_volume
    # ... (all tables and views)
```

#### The 2 views to create:

**View 1: `rep_hcp_monthly`**
Purpose: Flattens rep activity with HCP details — avoids fan-out by 
aggregating to a monthly summary BEFORE any joining.

```sql
CREATE VIEW rep_hcp_monthly AS
SELECT
    a.rep_id,
    r.first_name || ' ' || r.last_name   AS rep_name,
    a.hcp_id,
    h.full_name                           AS hcp_name,
    h.specialty,
    h.tier,
    h.territory_id,
    t.name                                AS territory_name,
    a.account_id,
    ac.name                               AS account_name,
    ac.account_type,
    d.year,
    d.quarter,
    strftime(d.calendar_date, '%Y-%m')    AS month,
    MAX(d.calendar_date)                  AS last_visit_date,
    COUNT(*)                              AS total_visits,
    COUNT(CASE WHEN a.status = 'completed'    THEN 1 END) AS completed_visits,
    COUNT(CASE WHEN a.status = 'cancelled'    THEN 1 END) AS cancelled_visits,
    COUNT(CASE WHEN a.activity_type = 'lunch_meeting' THEN 1 END) AS lunch_meetings,
    COUNT(CASE WHEN a.activity_type = 'call'          THEN 1 END) AS calls,
    AVG(a.duration_min)                   AS avg_duration_min
FROM fact_rep_activity a
JOIN rep_dim        r  ON a.rep_id     = r.rep_id
JOIN hcp_dim        h  ON a.hcp_id     = h.hcp_id
JOIN account_dim    ac ON a.account_id = ac.account_id
JOIN territory_dim  t  ON h.territory_id = t.territory_id
JOIN date_dim       d  ON a.date_id    = d.date_id
GROUP BY
    a.rep_id, rep_name, a.hcp_id, hcp_name, h.specialty, h.tier,
    h.territory_id, territory_name, a.account_id, account_name,
    ac.account_type, d.year, d.quarter, month
```

**View 2: `hcp_rx_monthly`**
Purpose: Combines HCP prescriptions with their territory/rep info.
Same monthly grain as fact_rx — safe to query without fan-out.

```sql
CREATE VIEW hcp_rx_monthly AS
SELECT
    rx.hcp_id,
    h.full_name      AS hcp_name,
    h.specialty,
    h.tier,
    h.territory_id,
    t.name           AS territory_name,
    d.year,
    d.quarter,
    strftime(d.calendar_date, '%Y-%m') AS month,
    rx.brand_code,
    rx.trx_cnt,
    rx.nrx_cnt
FROM fact_rx rx
JOIN hcp_dim       h ON rx.hcp_id  = h.hcp_id
JOIN territory_dim t ON h.territory_id = t.territory_id
JOIN date_dim      d ON rx.date_id = d.date_id
```

> ⚠️ **Why 2 views and not 1?** `fact_rx` is monthly grain.
> `fact_rep_activity` is daily grain. Joining them directly causes the
> fan-out problem (duplicate rows). We aggregate each to monthly separately,
> then the LLM can cross-reference them safely.

---

### `context.py` — Conversation History

**Responsibility:** Read and write per-rep conversation history stored as a CSV.

#### Conversation CSV schema (one file per rep):

```
conversations/rep_1.csv

timestamp,role,message
2026-04-11 10:00:01,user,"who are my top doctors by NRx?"
2026-04-11 10:00:04,assistant,"Your top doctors by NRx are Dr. Blake Garcia (42 NRx), ..."
2026-04-11 10:01:15,user,"how many times have I visited Dr. Blake?"
2026-04-11 10:01:17,assistant,"You have visited Dr. Blake Garcia 3 times, all completed, ..."
```

#### What it exports:

```python
def load_history(rep_id: int, last_n: int = 10) -> list[dict]
    # reads conversations/rep_{rep_id}.csv
    # returns last_n rows as [{role, message}, ...]
    # returns [] if file doesn't exist yet

def save_turn(rep_id: int, user_msg: str, assistant_msg: str) -> None
    # appends 2 rows (user + assistant) to conversations/rep_{rep_id}.csv
    # creates file + header if it doesn't exist
```

- Store last 10 messages loaded as context (= last 5 Q&A pairs)
- No truncation logic needed for demo — CSV grows naturally
- No database needed — just standard `csv` module from Python stdlib

---

### `prompts.py` — All Prompt Strings

**Responsibility:** Houses all prompt templates as plain strings with
`{placeholders}`. Zero logic here. Keeps agent.py clean.

#### The 4 prompts:

**1. `SYSTEM_PROMPT`** — injected into every LLM call

Note: `{schema}` is now populated at runtime by `db.get_schema(conn)`,
not hardcoded. This guarantees column names in the prompt always match
the actual database.

```
You are a pharma sales analytics co-pilot for {rep_name}.

REP CONTEXT (hardcoded, cannot be changed):
- Rep ID: {rep_id}
- Territory: {territory_name} (ID: {territory_id})
- All queries MUST filter by territory_id = {territory_id}

DATABASE SCHEMA (use ONLY these exact column names):
{schema}

VIEW PREFERENCE ORDER:
1. rep_hcp_monthly  — for anything about rep visits, activity, last contact
2. hcp_rx_monthly   — for anything about prescriptions (TRx, NRx)
3. Raw tables       — only when views cannot answer (e.g. payor_mix, ln_metrics)

RULES:
- Always scope queries to territory_id = {territory_id}
- Use DuckDB SQL dialect (strftime, date_diff, epoch functions)
- Write clean, correct SQL. If unsure, write a simpler query.
- Never use column names not present in the schema above.
```

**2. `PLANNER_PROMPT`** — fast classification call

```
Question: "{question}"

Is this answerable with ONE SQL query, or does it need multiple queries
and reasoning across results?

Respond with EXACTLY one of:
  SIMPLE
  COMPLEX: [sub-question 1] | [sub-question 2] | [sub-question 3]

Examples:
  "What is Dr. Blake's TRx this month?"
    → SIMPLE

  "Show me visit count vs NRx trend for Dr. Blake last 6 months"
    → SIMPLE

  "Which Tier A doctors haven't been visited in 30 days?"
    → SIMPLE

  "Who are my top doctors by NRx?"
    → SIMPLE

  "Which doctors should I visit next week?"
    → COMPLEX: Get Tier A/B HCPs in my territory | Get last visit date per HCP |
      Get NRx trend last 3 months | Rank by priority

  "Prepare me for my visit with Dr. Chen tomorrow"
    → COMPLEX: Get last 3 visits with Dr. Chen and their outcomes |
      Get NRx trend for Dr. Chen last 6 months |
      Get payor mix for Dr. Chen's account |
      Summarize into a visit prep brief

  "Is my visit activity actually driving prescriptions?"
    → COMPLEX: Get monthly completed visits per HCP last 6 months |
      Get monthly NRx per HCP same period |
      Correlate visit frequency with NRx changes

Only output SIMPLE or COMPLEX followed by sub-questions. Nothing else.
```

**3. `SQL_PROMPT`** — generates one SQL query

```
Generate a single DuckDB SQL query to answer:
"{sub_question}"

Context from previous steps:
{prior_results}

Territory filter that MUST appear in WHERE clause: territory_id = {territory_id}

Output ONLY the raw SQL query. No markdown. No explanation. No semicolon needed.
```

**4. `SYNTHESIZER_PROMPT`** — final answer from all results

```
Original question: "{original_question}"

Data gathered across {n_steps} queries:
{results_summary}

Conversation history for context:
{history}

Write a clear, direct answer using specific numbers from the data.
If ranking doctors or accounts, briefly explain why each is ranked that way.
Keep response under 200 words. Be conversational, not robotic.
```

---

### `agent.py` — The Brain

**Responsibility:** Orchestrates the 2-path flow (simple vs. complex).
Calls the LLM, runs SQL, collects results, synthesizes the answer.

#### Core flow:

```python
def run(question: str, rep: dict, history: list[dict]) -> str:
    
    # Step 1: Planner — classify the question
    plan = llm_call(
        system=SYSTEM_PROMPT.format(**rep),
        user=PLANNER_PROMPT.format(question=question)
    )
    
    # Step 2: Route based on plan
    if plan.startswith("SIMPLE"):
        sub_questions = [question]   # treat the original as the one query
    else:
        # parse out the sub-questions after "COMPLEX: "
        sub_questions = parse_sub_questions(plan)
    
    # Step 3: Execute SQL loop — with 1 retry on failure
    results = {}
    for i, sub_q in enumerate(sub_questions):
        sql = llm_call(
            system=SYSTEM_PROMPT.format(**rep, schema=schema),
            user=SQL_PROMPT.format(
                sub_question=sub_q,
                prior_results=format_results(results),
                territory_id=rep['territory_id']
            )
        )
        sql = clean_sql(sql)           # strip markdown fences if any
        try:
            df = conn.execute(sql).df()
            results[f"step_{i+1}"] = df.to_string(index=False)
        except Exception as first_error:
            # --- RETRY ONCE: feed the error back to the LLM ---
            # LLMs fail SQL ~15-20% of the time on first try.
            # One retry with error context drops that to ~2-3%.
            sql_retry = llm_call(
                system=SYSTEM_PROMPT.format(**rep, schema=schema),
                user=SQL_RETRY_PROMPT.format(
                    sub_question=sub_q,
                    failed_sql=sql,
                    error=str(first_error),
                    territory_id=rep['territory_id']
                )
            )
            sql_retry = clean_sql(sql_retry)
            try:
                df = conn.execute(sql_retry).df()
                results[f"step_{i+1}"] = df.to_string(index=False)
            except Exception as second_error:
                # Both attempts failed — record it and move on
                results[f"step_{i+1}"] = f"[Could not retrieve data: {second_error}]"
    
    # Step 4: Synthesize
    answer = llm_call(
        system=SYSTEM_PROMPT.format(**rep),
        user=SYNTHESIZER_PROMPT.format(
            original_question=question,
            results_summary=format_results(results),
            history=format_history(history),
            n_steps=len(sub_questions)
        )
    )
    
    return answer
```

#### Helper functions in agent.py:

```python
def llm_call(system: str, user: str) -> str
    # calls ollama.chat(model='llama3.1:8b', messages=[...])
    # uses native ollama Python library — no API key, runs fully local
    # returns response['message']['content'] as string

def parse_sub_questions(plan: str) -> list[str]
    # "COMPLEX: Q1 | Q2 | Q3" → ["Q1", "Q2", "Q3"]

def clean_sql(raw: str) -> str
    # strips ```sql ... ``` fences if LLM wraps output in markdown
    # also strips leading/trailing whitespace and trailing semicolons

def format_results(results: dict) -> str
    # formats step_1, step_2... into readable string for next LLM call

def format_history(history: list[dict]) -> str
    # formats [{role, message}...] into "User: ...\nAssistant: ...\n"
```

#### 5th prompt added: `SQL_RETRY_PROMPT`

Added to `prompts.py` — only used when first SQL attempt fails:

```
The following SQL query failed:

{failed_sql}

Error message:
{error}

The original question was: "{sub_question}"

Fix the SQL and return a corrected query.
Territory filter MUST be included: territory_id = {territory_id}
Output ONLY the corrected raw SQL. No explanation. No markdown.
```

---

### `main.py` — Terminal REPL

**Responsibility:** Entry point. Handles user input, coordinates all modules.

#### Terminal flow:

```
$ python main.py

╔══════════════════════════════════════════╗
║   Pharma Sales Co-pilot — SynthioLabs   ║
╚══════════════════════════════════════════╝

Available reps:
  1. Morgan Chen      (Territory 1)
  2. Jamie Thomas     (Territory 1)
  3. Casey Gonzalez   (Territory 1)
  4. River White      (Territory 2)
  5. Taylor Wilson    (Territory 2)
  6. Sage Brown       (Territory 2)
  7. River Miller     (Territory 3)
  8. Reese Miller     (Territory 3)
  9. Taylor Kim       (Territory 3)

Enter your Rep ID (1-9): 1

Welcome back, Morgan Chen! (Territory 1)
Loaded 6 previous messages from your history.

Type your question (or 'quit' to exit):
─────────────────────────────────────────

You: who are my top doctors by NRx this quarter?

[Thinking...]

Co-pilot: Your top 5 doctors by NRx in Q3 2025 are:
  1. Dr. Sage White (Tier A, Rheumatology) — 47 NRx
  2. Dr. Drew Wilson (Tier A, Internal Med) — 43 NRx
  ...

─────────────────────────────────────────
You: when did I last visit Dr. Sage White?

[Thinking...]

Co-pilot: Your last completed visit with Dr. Sage White was on 2025-07-17 —
a lunch meeting at Bay Hospital lasting 64 minutes.

─────────────────────────────────────────
You: quit

Goodbye, Morgan!
```

#### Logic in main.py:

```python
def main():
    conn = get_connection()          # db.py: load CSVs + create views
    rep = select_rep(conn)           # show rep list, read input, validate
    history = load_history(rep['rep_id'])   # context.py
    
    print(f"Welcome back, {rep['rep_name']}!")
    print(f"Loaded {len(history)} previous messages from your history.\n")
    
    while True:
        question = input("You: ").strip()
        if question.lower() in ('quit', 'exit', 'q'):
            break
        if not question:
            continue
        
        print("\n[Thinking...]\n")
        answer = run(question, rep, history)   # agent.py
        print(f"Co-pilot: {answer}\n")
        print("─" * 41)
        
        save_turn(rep['rep_id'], question, answer)   # context.py
        history.append({"role": "user",      "message": question})
        history.append({"role": "assistant", "message": answer})
        # keep only last 10 in memory
        history = history[-10:]
```

---

## Setup Steps for Running

```bash
# 1. Install and start Ollama (one time)
brew install ollama
ollama pull llama3.1:8b
ollama serve          # leave this running in a separate terminal tab

# 2. Navigate to project
cd /Users/himanshuwagh/Documents/synthio-assignment

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies (no OpenAI key needed)
pip install duckdb ollama python-dotenv

# 5. Run
python main.py
```

---

## What Gets Built in Each Session

```
First run:
  conversations/          ← created automatically
  
After first question by rep 1:
  conversations/rep_1.csv ← created automatically with header
```

---

## Summary of What Each File Touches

| File | Touches LLM? | Touches DB? | Touches CSV? |
|---|---|---|---|
| `main.py` | ✗ | ✗ | ✗ |
| `db.py` | ✗ | ✅ | ✗ |
| `agent.py` | ✅ | ✅ | ✗ |
| `context.py` | ✗ | ✗ | ✅ |
| `prompts.py` | ✗ | ✗ | ✗ |

Clean separation. Each file does exactly one thing.

---

## Conscious Tradeoffs (For README)

| Decision | What was traded off | Why |
|---|---|---|
| DuckDB in-memory | Data lost on restart | Acceptable for demo |
| CSV for history | Not concurrent-safe | Fine for single-user terminal |
| 10-message window | Long history lost | Keeps prompt size small |
| 1 retry on bad SQL (not 3) | Very rare 2nd failures shown as error | LLMs fix most errors in 1 retry; 3 retries triple latency for marginal gain |
| Runtime schema injection | Tiny extra startup cost | Eliminates a class of silent bugs; worth it |
| Views not persisted | Recreated each startup (~50ms) | Zero setup required, negligible cost |
| `ollama` lib directly, not OpenAI client | Slightly less portable LLM switching | No fake API key, no base_url override, cleaner code, zero cost |
