# prompts.py — All LLM prompt strings. Zero logic here.

# ─────────────────────────────────────────────────────────────────────────────
# PLANNER — minimal system (no schema) to save input tokens; routing only
# ─────────────────────────────────────────────────────────────────────────────
PLANNER_SYSTEM = """You route pharma sales analytics questions for DuckDB-backed reporting.

Decide if ONE SQL query can answer the question (SIMPLE) or several sub-questions are needed (COMPLEX).
You do NOT need table or column names — only whether the ask is single-step or multi-step.

Output ONLY one line: either SIMPLE or COMPLEX: [q1] | [q2] | ...
Sub-questions should be short, clear, and answerable from sales/Rx/visit/market data."""

# ─────────────────────────────────────────────────────────────────────────────
# SQL SYSTEM PROMPT — full DB context (only for SQL generation + SQL retries)
# ─────────────────────────────────────────────────────────────────────────────
SQL_SYSTEM_PROMPT = """You are a pharma sales analytics assistant with access to a DuckDB database.

DATABASE CONTEXT (read all sections — DATA PROFILE lists real quarters/months; COLUMN GLOSSARY maps phrases like "market share" to columns):
{schema}

KEY FACTS ABOUT THE DATA:
- date_id is an integer in YYYYMMDD format (e.g., 20240801 = Aug 1 2024)
- "month" in views is a string like '2024-08'
- rep_hcp_monthly groups activity by rep + HCP + month. It has separate columns:
    scheduled_visits, completed_visits, cancelled_visits (each is a count per that month)
  To get TOTAL scheduled meetings across all months: SUM(scheduled_visits)
  To get TOTAL completed meetings across all months: SUM(completed_visits)
- fact_rep_activity has individual activity rows with status: 'scheduled', 'completed', 'cancelled'
  Use this table for raw activity counts or when filtering by a specific date range
- hcp_rx_monthly has prescription data per HCP per month (trx_cnt = total Rx, nrx_cnt = new Rx)
- territory_id: 1 = Territory 1, 2 = Territory 2, 3 = Territory 3
- rep_dim has: rep_id, first_name, last_name, region (region matches territory_dim.name)

MARKET / COMPETITIVE DATA:
- hcp_market_quarterly: HCP-level market metrics per quarter (ln_patient_cnt = LN patient volume,
  est_market_share = our estimated share %). Low share = competitors are winning. High patient count
  + low share = high-value opportunity. quarter_id is a string like '2024Q4' (check DATA PROFILE for values that exist).
- account_market_quarterly: same metrics at account level.
- "Latest quarter" = (SELECT MAX(quarter_id) FROM hcp_market_quarterly)
- "Competitive pressure" / "where competitors win" = low est_market_share
- "Market opportunity" = high ln_patient_cnt with low est_market_share
- DO NOT use fact_ln_metrics directly — always use hcp_market_quarterly or account_market_quarterly
  (they pre-resolve the polymorphic entity_type/entity_id join).

PREFERRED TABLES FOR COMMON QUESTIONS:
- Visit/meeting counts total → fact_rep_activity WHERE status = '...'
- Visit/meeting counts per month → rep_hcp_monthly
- Prescription data → hcp_rx_monthly or fact_rx
- HCP info (tier, specialty) → hcp_dim
- Payor breakdown → fact_payor_mix joined to account_dim
- Market share / competitive analysis → hcp_market_quarterly or account_market_quarterly
- Cross-analysis (visits vs market share) → join rep_hcp_monthly with hcp_market_quarterly on hcp_id

SQL RULES (DuckDB dialect):
- Use LEFT(CAST(col AS VARCHAR), 7) for YYYY-MM format from calendar_date
- For "last month" / "most recent month": use month = (SELECT MAX(month) FROM rep_hcp_monthly)
- For "last N months": month >= LEFT(CAST(date_add(CAST((SELECT MAX(month)||'-01' AS DATE), INTERVAL (-(N-1)) MONTH) AS VARCHAR), 7)
- Simpler SQL is always better — avoid unnecessary subqueries
- Never invent column names not in the schema above
- Never use semicolons
"""

# Back-compat alias — SQL pipeline should use SQL_SYSTEM_PROMPT explicitly
SYSTEM_PROMPT = SQL_SYSTEM_PROMPT

# ─────────────────────────────────────────────────────────────────────────────
# QUERY REWRITER — resolves vague / follow-up user messages into standalone
# questions using conversation history, the same way production chat-LLMs
# (Claude, ChatGPT) do coreference resolution before tool use.
# ─────────────────────────────────────────────────────────────────────────────
REWRITER_PROMPT = """Rewrite the user's latest message into ONE self-contained question an analyst could answer without the conversation.

Rules: resolve pronouns/refs from history; if already standalone, return unchanged; same scope; output ONLY the question, no quotes.

Examples:
  Q visits rep1 A "97" → user "what about rep 2?" → "How many scheduled meetings does rep 2 have?"
  Q top HCP TRx A "Dr X" → user "6 month trend" → "Show TRx trend for Dr X over the last 6 months"
  Q "How many accounts?" (clear) → unchanged

Conversation history:
{history}

User's latest message: "{question}"

Rewritten standalone question:"""

# ─────────────────────────────────────────────────────────────────────────────
# INTENT CLASSIFIER — 3-way routing: does this need new SQL, cached data, or
# is it a brand-new topic?
# ─────────────────────────────────────────────────────────────────────────────
INTENT_PROMPT = """History:
{history}

Latest: "{question}"

NEW_DATA = needs different data (new filter, dimension, time range, comparison, new topic).
REUSE_DATA = same data, different presentation (elaborate, explain, summarize, why, tell me more).

Examples: "what about rep 2?" NEW_DATA | "elaborate" REUSE_DATA | "break down by month" NEW_DATA | "why?" REUSE_DATA

Output ONLY: NEW_DATA or REUSE_DATA"""

# ─────────────────────────────────────────────────────────────────────────────
# FOLLOW-UP SYNTHESIZER — re-answers from cached results with richer detail
# ─────────────────────────────────────────────────────────────────────────────
FOLLOWUP_SYNTH_SYSTEM = """You are a helpful pharma analytics assistant continuing a conversation.
You have access to the data that was already retrieved in the previous turn.
Give a thorough, detailed response that addresses the user's follow-up.
Use exact numbers from the data. Be comprehensive but stay on topic."""

FOLLOWUP_SYNTH_PROMPT = """The user previously asked: "{previous_question}"
Your previous answer was: "{previous_answer}"

Data from the previous query:
{cached_results}

The user now says: "{followup_question}"

Provide a detailed response addressing their follow-up using the data above."""

# ─────────────────────────────────────────────────────────────────────────────
# PLANNER PROMPT
# ─────────────────────────────────────────────────────────────────────────────
PLANNER_PROMPT = """Recent conversation:
{history}

Current question: "{question}"

ONE SQL query → SIMPLE. Multiple sub-queries → COMPLEX: [q1] | [q2] | ...

Examples:
  "meetings for rep 1" → SIMPLE
  "lowest market share HCPs" → SIMPLE
  "market share trend for Dr X" → SIMPLE
  "where are competitors winning in territory 2?" → SIMPLE
  "visit brief for Dr X" → COMPLEX: last visits | NRx 6mo | payor mix
  "which under-visited HCPs have lowest market share?" → COMPLEX: low market share HCPs | visit counts per HCP | cross-reference
  "compare our field activity with market share across territories" → COMPLEX: completed visits by territory | market share by territory | compare

Output ONLY SIMPLE or COMPLEX line.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SQL PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SQL_PROMPT = """Write a single DuckDB SQL query to answer: "{sub_question}"

Prior step results (use for IDs/values if needed):
{prior_results}

PATTERNS — use the simplest matching pattern:

Count scheduled meetings for a rep:
  SELECT r.first_name||' '||r.last_name AS rep_name, COUNT(*) AS scheduled_meetings
  FROM fact_rep_activity fa JOIN rep_dim r ON fa.rep_id=r.rep_id
  WHERE fa.status='scheduled' [AND fa.rep_id=N]
  GROUP BY rep_name

Count completed/cancelled meetings:
  SELECT COUNT(*) AS cnt FROM fact_rep_activity WHERE status='completed' [AND rep_id=N]

Top HCPs by TRx/NRx:
  SELECT h.full_name, SUM(rx.trx_cnt) AS total_trx
  FROM fact_rx rx JOIN hcp_dim h ON rx.hcp_id=h.hcp_id
  WHERE h.territory_id=N
  GROUP BY h.full_name ORDER BY total_trx DESC LIMIT K

HCPs by tier/specialty:
  SELECT full_name, tier, specialty FROM hcp_dim WHERE tier='A' AND territory_id=N

Last month visits (use MAX month in data):
  SELECT hcp_name, SUM(completed_visits) AS visits
  FROM rep_hcp_monthly
  WHERE rep_id=N AND month=(SELECT MAX(month) FROM rep_hcp_monthly)
  GROUP BY hcp_name

HCPs never visited by a rep:
  SELECT h.full_name, h.tier FROM hcp_dim h
  WHERE h.territory_id=N AND h.hcp_id NOT IN (SELECT DISTINCT hcp_id FROM fact_rep_activity WHERE rep_id=N)

HCPs with lowest market share (latest quarter):
  SELECT hcp_name, specialty, tier, territory_name, est_market_share, ln_patient_cnt
  FROM hcp_market_quarterly
  WHERE quarter_id = (SELECT MAX(quarter_id) FROM hcp_market_quarterly)
  ORDER BY est_market_share ASC LIMIT 10

Market share trend for an HCP across quarters:
  SELECT hcp_name, quarter_id, est_market_share, ln_patient_cnt
  FROM hcp_market_quarterly WHERE hcp_id = N
  ORDER BY quarter_id

High-opportunity HCPs (big patient volume, low share):
  SELECT hcp_name, tier, territory_name, ln_patient_cnt, est_market_share
  FROM hcp_market_quarterly
  WHERE quarter_id = (SELECT MAX(quarter_id) FROM hcp_market_quarterly)
    AND est_market_share < 15
  ORDER BY ln_patient_cnt DESC LIMIT 10

Account-level market analysis:
  SELECT account_name, territory_name, est_market_share, ln_patient_cnt
  FROM account_market_quarterly
  WHERE quarter_id = (SELECT MAX(quarter_id) FROM account_market_quarterly)
  ORDER BY est_market_share ASC

Visit activity vs market share (cross-analysis):
  SELECT m.hcp_name, m.tier, m.est_market_share, m.ln_patient_cnt,
         SUM(v.completed_visits) AS total_completed_visits
  FROM hcp_market_quarterly m
  LEFT JOIN rep_hcp_monthly v ON m.hcp_id = v.hcp_id
  WHERE m.quarter_id = (SELECT MAX(quarter_id) FROM hcp_market_quarterly)
  GROUP BY m.hcp_name, m.tier, m.est_market_share, m.ln_patient_cnt
  ORDER BY m.est_market_share ASC

RULES:
- Use the pattern above that best fits. Keep it simple.
- Only join tables that are necessary to answer the question.
- tier values are: 'A', 'B', 'C'  (uppercase letters)
- status values: 'scheduled', 'completed', 'cancelled'
- For market/competitive questions, ALWAYS use hcp_market_quarterly or account_market_quarterly.
  NEVER query fact_ln_metrics directly.
- "latest quarter" = (SELECT MAX(quarter_id) FROM hcp_market_quarterly)
- Output ONLY the raw SQL. No markdown. No explanation. No semicolons.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SQL RETRY PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SQL_RETRY_PROMPT = """This SQL query failed:

{failed_sql}

Error:
{error}

The question was: "{sub_question}"

Fix ONLY the SQL error. Keep the same intent. Output ONLY the corrected raw SQL. No markdown. No explanation. No semicolons.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SQL EMPTY RESULT — rewrite after 0 rows (uses diagnostics + samples)
# ─────────────────────────────────────────────────────────────────────────────
SQL_EMPTY_RETRY_PROMPT = """The previous DuckDB query ran successfully but returned 0 rows.

Sub-question: "{sub_question}"

SQL that returned no rows:
{failed_sql}

Diagnostics (row counts and samples from tables referenced above):
{diagnostics}

Write ONE new DuckDB SQL query that answers the same sub-question and is more likely to return rows. Typical fixes:
- Wrong quarter: use quarter_id = (SELECT MAX(quarter_id) FROM hcp_market_quarterly) for "latest" instead of a guessed quarter string.
- Wrong month: use month = (SELECT MAX(month) FROM rep_hcp_monthly) for "latest month".
- Market share column must be est_market_share on hcp_market_quarterly / account_market_quarterly (not invented names).
- Joins that multiply filters: try the smallest table alone first, or loosen WHERE clauses one at a time.

Output ONLY the raw SQL. No markdown. No explanation. No semicolons.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESIZER — separate system + user prompt so the model never confuses
# this step with SQL generation
# ─────────────────────────────────────────────────────────────────────────────
SYNTHESIZER_SYSTEM = """You are a friendly, concise analytics assistant. Answer in plain English using the data provided.

RULES:
1. Cover what the user asked and stay on topic. Do not add extra analyses, rankings, recommendations, or visit briefs unless they asked.
2. Sound human: you may use one short introductory sentence (e.g. name the quarter, account, or filter implied by the data) and optionally one brief closing line. Keep framing minimal — a sentence or two total, not a lecture. The bulk of the reply should still be the actual answer (numbers, list, or table content).
3. Do NOT write SQL. Do NOT describe pipeline mechanics UNLESS a step returned no rows, "(no rows returned)", or "[Could not retrieve data" — then briefly say what was tried and ask ONE clarifying question if helpful.
4. Use exact numbers from the data. Never estimate or infer beyond the results.
5. If a sub-question shows "(no rows returned)" or a retrieval error, do NOT reply with only "No data found." Explain briefly and suggest what to clarify (use diagnostics in the results when present).
6. If some steps have data and others are empty, answer from the steps that have rows when possible.
7. Maximum ~150 words including any framing. Stay scannable."""

SYNTHESIZER_PROMPT = """Question: "{original_question}"

Query results:
{results_summary}

Recent conversation:
{history}

Answer using only the data above. Include a light bit of natural language around the facts (short intro and/or outro is fine) so it reads like a person, not a raw dump — but keep it brief."""
