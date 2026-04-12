# agent.py — Orchestrates the full question-answering pipeline.
#
# Flow:
#   0a. Rewriter → resolve vague / follow-up messages into standalone questions
#   0b. Intent   → REUSE_DATA (elaborate from cache) vs NEW_DATA (run SQL)
#   0c. Ambiguity → detect metric ambiguity; surface alternatives to the user
#   1.  Planner  → classify question as SIMPLE or COMPLEX
#   2.  SQL loop → generate + execute SQL per sub-question (1 retry on failure)
#   3.  Synthesizer → produce a plain-English answer from all gathered data
#
# Cost controls: skip rewriter when history is empty; heuristic reuse-intent;
# short acknowledgments without LLM; bounded LRU answer cache; lower max_tokens
# on short structured outputs.
#
# Return value: agent.run() always returns a dict (see _make_result helper).
# The "answer" key is always a plain string; the rest provides product-instinct
# metadata consumed by the UI (interpretation, alternatives, sql_steps, etc.).

import hashlib
import json
import re
from collections import OrderedDict
from typing import Optional, Tuple

from prompts import (
    PLANNER_SYSTEM,
    SQL_SYSTEM_PROMPT,
    REWRITER_PROMPT,
    INTENT_PROMPT,
    FOLLOWUP_SYNTH_SYSTEM,
    FOLLOWUP_SYNTH_PROMPT,
    PLANNER_PROMPT,
    SQL_PROMPT,
    SQL_RETRY_PROMPT,
    SQL_EMPTY_RETRY_PROMPT,
    SYNTHESIZER_SYSTEM,
    SYNTHESIZER_PROMPT,
    AMBIGUITY_PROMPT,
)
from llm_client import chat as _llm_chat
from db import gather_empty_sql_diagnostics

_conn = None
# Full SQL context (profile + glossary + DESCRIBE). Used only for SQL steps — not for planner.
_sql_context = ""

# Last-turn cache per browser/session id so concurrent users never share context.
_LAST_TURN_MAX = 512
_last_turn_by_session: "OrderedDict[str, dict]" = OrderedDict()

ANSWER_CACHE_MAX = 128
_answer_cache: "OrderedDict[str, Tuple[str, str]]" = OrderedDict()

_ACK_RE = re.compile(
    r"^\s*(thanks?|thank\s+you|thx|ok+ay|got\s+it|cool|nice|great)\s*[!.,]?\s*$",
    re.I,
)

# Heuristic patterns that indicate a potentially ambiguous question.
# Only these trigger the LLM ambiguity check — keeps it cheap.
_AMBIGUOUS_PATTERNS = [
    re.compile(r'\btop\s*\d*\s*(hcps?|reps?|accounts?|territories?)\b', re.I),
    re.compile(r'\bbest\s+(performing|hcps?|reps?|accounts?)\b', re.I),
    re.compile(r'\bwho\s+should\s+i\s+(focus|prioritize|visit)\b', re.I),
    re.compile(r'\b(under|over)(performing|visited)\b', re.I),
    re.compile(r'\bperformance\s+(breakdown|by|of|across)\b', re.I),
    re.compile(r'\bwhich\s+(hcps?|reps?|accounts?)\s+(are|have)\s+(the\s+)?(best|worst|top|highest|lowest)\b', re.I),
]


def _session_scope(session_id: Optional[str]) -> str:
    s = (session_id or "").strip()
    return s if s else "__cli__"


def _last_turn_get(scope: str) -> dict:
    d = _last_turn_by_session.get(scope)
    return dict(d) if d else {}


def _last_turn_set(scope: str, data: dict) -> None:
    _last_turn_by_session[scope] = data
    _last_turn_by_session.move_to_end(scope)
    while len(_last_turn_by_session) > _LAST_TURN_MAX:
        _last_turn_by_session.popitem(last=False)


def _answer_cache_key(resolved: str, history: list, session_scope: str) -> str:
    tail = [{"role": m.get("role", ""), "message": m.get("message", "")} for m in history[-4:]]
    payload = json.dumps(
        {"sid": session_scope, "q": resolved.strip().lower(), "h": tail},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256((payload + "\0" + _sql_context).encode("utf-8")).hexdigest()


def _answer_cache_get(key: str) -> Optional[Tuple[str, str]]:
    if key not in _answer_cache:
        return None
    _answer_cache.move_to_end(key)
    return _answer_cache[key]


def _answer_cache_set(key: str, answer: str, results_text: str) -> None:
    if key in _answer_cache:
        del _answer_cache[key]
    _answer_cache[key] = (answer, results_text)
    while len(_answer_cache) > ANSWER_CACHE_MAX:
        _answer_cache.popitem(last=False)


def _looks_like_reuse_intent(question: str) -> bool:
    """Obvious elaboration-style follow-ups — skip LLM intent when prior turn exists."""
    t = question.strip().lower()
    if len(t) > 120:
        return False
    if re.match(
        r"^\s*(please\s+)?(elaborate|explain(\s+more|\s+further)?|tell\s+me\s+more|"
        r"go\s+on|summarize|shorter|briefly|in\s+brief|more\s+detail|in\s+more\s+detail|"
        r"what\s+does\s+that\s+mean|can\s+you\s+expand|clarify)\s*[!.,]?\s*$",
        t,
    ):
        return True
    if re.match(r"^\s*why\s*[!?.,]?\s*$", t):
        return True
    return False


def init(conn, schema: str) -> None:
    global _conn, _sql_context
    _conn = conn
    _sql_context = schema


# ─────────────────────────────────────────────────────────────────────────────
def llm_call(system: str, user: str, max_tokens: Optional[int] = None) -> str:
    return _llm_chat(system, user, max_tokens=max_tokens)


def clean_sql(raw: str) -> str:
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    # Strip surrounding quotes the model sometimes adds
    if (raw.startswith('"') and raw.endswith('"')) or \
       (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()
    return raw.rstrip(";").strip()


def parse_sub_questions(plan: str) -> list:
    plan = plan.strip()
    if "COMPLEX" in plan.upper():
        parts = plan.split(":", 1)
        if len(parts) > 1:
            questions = [q.strip() for q in parts[1].split("|") if q.strip()]
            if questions:
                return questions
    return [plan]


def format_results(results: dict) -> str:
    if not results:
        return "No prior results."
    lines = []
    for step, data in results.items():
        lines.append(f"--- {step} ---\n{data}")
    return "\n\n".join(lines)


def format_history(history: list) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history:
        role = "You" if msg["role"] == "user" else "Co-pilot"
        lines.append(f"{role}: {msg['message']}")
    return "\n".join(lines)


def _sql_step_result(system: str, sub_q: str, prior_results: str) -> dict:
    """
    Generate SQL, execute, retry on SQL errors, then on empty results run diagnostics + one LLM rewrite.

    Returns a dict:
      {
        "result":     str,          # data text passed to the synthesizer
        "sql":        str,          # the final SQL (succeeded or last attempted)
        "diagnostic": str | None,   # diagnostic info if query returned no rows
      }
    """
    if _conn is None:
        return {"result": "[Agent error: database is not initialized]", "sql": "", "diagnostic": None}

    sql = llm_call(
        system=system,
        user=SQL_PROMPT.format(
            sub_question=sub_q,
            prior_results=prior_results,
        ),
        max_tokens=2048,
    )
    sql = clean_sql(sql)

    df = None
    try:
        df = _conn.execute(sql).df()
    except Exception as first_error:
        sql_retry = llm_call(
            system=system,
            user=SQL_RETRY_PROMPT.format(
                sub_question=sub_q,
                failed_sql=sql,
                error=str(first_error),
            ),
            max_tokens=2048,
        )
        sql_retry = clean_sql(sql_retry)
        try:
            df = _conn.execute(sql_retry).df()
            sql = sql_retry
        except Exception as second_error:
            human_msg = (
                "I couldn't compute this result — the query failed after two attempts. "
                "Try rephrasing or asking about a specific time period."
            )
            return {
                "result": f"[Could not retrieve data: {second_error}]",
                "sql": sql_retry,
                "diagnostic": None,
                "error_message": human_msg,
            }

    if df is not None and not df.empty:
        return {"result": df.to_string(index=False), "sql": sql, "diagnostic": None}

    # Empty result path: gather diagnostics and attempt one rewrite
    diag = gather_empty_sql_diagnostics(_conn, sql)
    sql_empty = llm_call(
        system=system,
        user=SQL_EMPTY_RETRY_PROMPT.format(
            sub_question=sub_q,
            failed_sql=sql,
            diagnostics=diag,
        ),
        max_tokens=2048,
    )
    sql_empty = clean_sql(sql_empty)
    try:
        df2 = _conn.execute(sql_empty).df()
    except Exception as empty_fix_err:
        return {
            "result": "(no rows returned)\n"
                      f"--- SQL attempted ---\n{sql}\n"
                      f"--- Diagnostics ---\n{diag}\n"
                      f"--- Empty-result rewrite failed ---\n{empty_fix_err}",
            "sql": sql,
            "diagnostic": diag,
        }

    if not df2.empty:
        return {"result": df2.to_string(index=False), "sql": sql_empty, "diagnostic": None}

    diag2 = gather_empty_sql_diagnostics(_conn, sql_empty)
    combined_diag = f"{diag}\n\n--- After relaxed query ---\n{diag2}"
    return {
        "result": "(no rows returned)\n"
                  f"--- SQL attempted ---\n{sql}\n"
                  f"--- Empty-result SQL ---\n{sql_empty}\n"
                  f"--- Diagnostics ---\n{combined_diag}",
        "sql": sql_empty,
        "diagnostic": combined_diag,
    }


# ─────────────────────────────────────────────────────────────────────────────
def _rewrite_question(question: str, history: list) -> str:
    """Resolve vague / anaphoric follow-ups into standalone questions."""
    if not history:
        return question.strip()
    hist_text = format_history(history[-6:])
    rewritten = llm_call(
        system="You are a query rewriter. Output only the rewritten question.",
        user=REWRITER_PROMPT.format(question=question, history=hist_text),
        max_tokens=384,
    )
    return rewritten.strip().strip('"') if rewritten and not rewritten.startswith("[LLM error") else question


def _classify_intent(question: str, history: list, last_turn: dict) -> str:
    """Return 'REUSE_DATA' or 'NEW_DATA'."""
    if not last_turn:
        return "NEW_DATA"
    hist_text = format_history(history[-6:])
    intent = llm_call(
        system="You classify user intent. Output ONLY: NEW_DATA or REUSE_DATA",
        user=INTENT_PROMPT.format(question=question, history=hist_text),
        max_tokens=24,
    )
    return "REUSE_DATA" if "REUSE_DATA" in (intent or "").upper() else "NEW_DATA"


def _handle_followup(
    question: str, rewritten: str, history: list, last_turn: dict
) -> str:
    """Re-synthesize from cached last-turn data with richer detail."""
    return llm_call(
        system=FOLLOWUP_SYNTH_SYSTEM,
        user=FOLLOWUP_SYNTH_PROMPT.format(
            previous_question=last_turn.get("question", ""),
            previous_answer=last_turn.get("answer", ""),
            cached_results=last_turn.get("results_text", "No cached data."),
            followup_question=rewritten,
        ),
        max_tokens=2048,
    )


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response (handles markdown fences)."""
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # Find first {...} block (handles nested braces one level deep)
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _check_ambiguity(question: str) -> dict:
    """
    Detect whether a question has multiple valid metric interpretations.

    Returns:
      {
        "is_ambiguous": bool,
        "interpretation": str,   # default interpretation we will answer
        "alternatives": list,    # [{label, query}, ...]
      }

    Only runs an LLM call when the question matches known ambiguous patterns —
    this keeps cost/latency impact minimal for unambiguous queries.
    """
    is_candidate = any(p.search(question) for p in _AMBIGUOUS_PATTERNS)
    if not is_candidate:
        return {"is_ambiguous": False, "interpretation": "", "alternatives": []}

    raw = llm_call(
        system="You analyze analytics questions for metric ambiguity. Return ONLY valid JSON.",
        user=AMBIGUITY_PROMPT.format(question=question),
        max_tokens=384,
    )

    if not raw or raw.startswith("[LLM error"):
        return {"is_ambiguous": False, "interpretation": "", "alternatives": []}

    data = _extract_json(raw)
    return {
        "is_ambiguous": bool(data.get("is_ambiguous", False)),
        "interpretation": str(data.get("interpretation", "")),
        "alternatives": list(data.get("alternatives", [])),
    }


def _make_result(
    answer: str,
    interpretation: str = "",
    alternatives: Optional[list] = None,
    sql_steps: Optional[list] = None,
    empty_diagnostic: Optional[str] = None,
    warning: Optional[str] = None,
) -> dict:
    """Construct the standardised return dict for agent.run()."""
    return {
        "answer": answer,
        "interpretation": interpretation,
        "alternatives": alternatives or [],
        "sql_steps": sql_steps or [],
        "empty_diagnostic": empty_diagnostic,
        "warning": warning,
    }


# ─────────────────────────────────────────────────────────────────────────────
def run(question: str, history: list, session_id: Optional[str] = None) -> dict:
    """
    Execute the full pipeline and return a structured result dict:
      - answer:          plain-English answer (always a string)
      - interpretation:  how the system understood the question
      - alternatives:    list of [{label, query}] for ambiguous questions
      - sql_steps:       list of [{step, question, sql}] for transparency
      - empty_diagnostic: diagnostic text when no data was found
      - warning:         context warning when the answer may be incomplete
    """
    if _conn is None:
        return _make_result("[Agent error: database is not initialized]")

    scope = _session_scope(session_id)
    last_turn = _last_turn_get(scope)

    qstrip = question.strip()
    if history and _ACK_RE.match(qstrip):
        return _make_result("You're welcome! Ask if you need anything else about the sales data.")

    sql_system = SQL_SYSTEM_PROMPT.format(schema=_sql_context)

    # ── Step 0a: Rewrite for context resolution ─────────────────────────────
    resolved = _rewrite_question(question, history)

    # ── Step 0b: Classify intent — can we reuse cached data? ─────────────────
    if last_turn and _looks_like_reuse_intent(question):
        intent = "REUSE_DATA"
    else:
        intent = _classify_intent(question, history, last_turn)

    # ── Fast path: pure elaboration/explanation on same data ─────────────────
    if intent == "REUSE_DATA" and last_turn:
        answer = _handle_followup(question, resolved, history, last_turn)
        _last_turn_set(
            scope,
            {
                "question": question,
                "resolved": resolved,
                "answer": answer,
                "results_text": last_turn.get("results_text", ""),
            },
        )
        return _make_result(answer, interpretation=resolved)

    # ── Step 0c: Ambiguity detection (only for new data queries) ─────────────
    ambiguity = _check_ambiguity(resolved)
    interpretation = ambiguity["interpretation"] if ambiguity["is_ambiguous"] else ""
    alternatives = ambiguity["alternatives"] if ambiguity["is_ambiguous"] else []

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_key = _answer_cache_key(resolved, history, scope)
    cached = _answer_cache_get(cache_key)
    if cached is not None:
        answer, results_text = cached
        _last_turn_set(
            scope,
            {
                "question": question,
                "resolved": resolved,
                "answer": answer,
                "results_text": results_text,
            },
        )
        # Return cached answer; alternatives still shown (computed above)
        return _make_result(
            answer,
            interpretation=interpretation or resolved,
            alternatives=alternatives,
        )

    # ── Step 1: Planner ───────────────────────────────────────────────────────
    plan = llm_call(
        system=PLANNER_SYSTEM,
        user=PLANNER_PROMPT.format(
            question=resolved,
            history=format_history(history[-4:]),
        ),
        max_tokens=512,
    )

    if plan.startswith("[LLM error"):
        try:
            from config import get_llm_settings
            prov = get_llm_settings().provider
        except Exception:
            prov = "ollama"
        detail = plan
        if detail.startswith("[LLM error: ") and detail.endswith("]"):
            detail = detail[len("[LLM error: "):-1].strip()
        if prov == "ollama":
            msg = (
                "Sorry, I couldn't reach the local LLM. Is Ollama running? "
                "(`ollama serve`) Also check `model` in config.ini.\n\n"
                f"Details: {detail}"
            )
        else:
            msg = (
                "Sorry, the cloud LLM request failed. Check API keys in `.env`, "
                "`provider` and `model` in `config.ini`, and restart the server after changes.\n\n"
                f"Details: {detail}"
            )
        return _make_result(msg, interpretation=resolved, alternatives=alternatives)

    # ── Step 2: Route ─────────────────────────────────────────────────────────
    if "COMPLEX" in plan.upper():
        sub_questions = parse_sub_questions(plan)
    else:
        sub_questions = [resolved]

    # ── Step 3: SQL Execution Loop ────────────────────────────────────────────
    results: dict[str, str] = {}
    sql_steps: list[dict] = []
    empty_diagnostic: Optional[str] = None

    for i, sub_q in enumerate(sub_questions):
        step_key = f"step_{i + 1}"
        step_data = _sql_step_result(sql_system, sub_q, format_results(results))

        results[step_key] = step_data["result"]
        if step_data.get("sql"):
            sql_steps.append({
                "step": step_key,
                "question": sub_q,
                "sql": step_data["sql"],
            })
        if step_data.get("diagnostic") and not empty_diagnostic:
            empty_diagnostic = step_data["diagnostic"]

    # ── Step 4: Synthesize ────────────────────────────────────────────────────
    answer = llm_call(
        system=SYNTHESIZER_SYSTEM,
        user=SYNTHESIZER_PROMPT.format(
            original_question=resolved,
            results_summary=format_results(results),
            history=format_history(history),
        ),
        max_tokens=640,
    )

    # Cache this turn for potential follow-ups
    results_text = format_results(results)
    _last_turn_set(
        scope,
        {
            "question": question,
            "resolved": resolved,
            "answer": answer,
            "results_text": results_text,
        },
    )
    if not answer.startswith("[LLM error"):
        _answer_cache_set(cache_key, answer, results_text)

    # Expose the empty diagnostic only when the final answer actually says "no data"
    expose_diag = empty_diagnostic if (
        empty_diagnostic
        and answer
        and ("no data" in answer.lower() or "no rows" in answer.lower() or "not found" in answer.lower())
    ) else None

    return _make_result(
        answer,
        interpretation=interpretation or resolved,
        alternatives=alternatives,
        sql_steps=sql_steps,
        empty_diagnostic=expose_diag,
    )
