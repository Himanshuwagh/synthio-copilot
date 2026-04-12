# agent.py — Orchestrates the full question-answering pipeline.
#
# Flow:
#   0a. Rewriter → resolve vague / follow-up messages into standalone questions
#   0b. Intent   → REUSE_DATA (elaborate from cache) vs NEW_DATA (run SQL)
#   1.  Planner  → classify question as SIMPLE or COMPLEX
#   2.  SQL loop → generate + execute SQL per sub-question (1 retry on failure)
#   3.  Synthesizer → produce a plain-English answer from all gathered data
#
# Cost controls: skip rewriter when history is empty; heuristic reuse-intent;
# short acknowledgments without LLM; bounded LRU answer cache; lower max_tokens
# on short structured outputs.

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


def _sql_step_result(system: str, sub_q: str, prior_results: str) -> str:
    """
    Generate SQL, execute, retry on SQL errors, then on empty results run diagnostics + one LLM rewrite.
    """
    if _conn is None:
        return "[Agent error: database is not initialized]"

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
            return f"[Could not retrieve data: {second_error}]"

    if df is not None and not df.empty:
        return df.to_string(index=False)

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
        return (
            "(no rows returned)\n"
            f"--- SQL attempted ---\n{sql}\n"
            f"--- Diagnostics ---\n{diag}\n"
            f"--- Empty-result rewrite failed ---\n{empty_fix_err}"
        )

    if not df2.empty:
        return df2.to_string(index=False)

    diag2 = gather_empty_sql_diagnostics(_conn, sql_empty)
    return (
        "(no rows returned)\n"
        f"--- SQL attempted ---\n{sql}\n"
        f"--- Empty-result SQL ---\n{sql_empty}\n"
        f"--- Diagnostics (original) ---\n{diag}\n"
        f"--- Diagnostics (after empty retry) ---\n{diag2}"
    )


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


def run(question: str, history: list, session_id: Optional[str] = None) -> str:
    if _conn is None:
        return "[Agent error: database is not initialized]"

    scope = _session_scope(session_id)
    last_turn = _last_turn_get(scope)

    qstrip = question.strip()
    if history and _ACK_RE.match(qstrip):
        return "You're welcome! Ask if you need anything else about the sales data."

    sql_system = SQL_SYSTEM_PROMPT.format(schema=_sql_context)

    # ── Step 0a: Rewrite for context resolution (always runs) ──────────
    resolved = _rewrite_question(question, history)

    # ── Step 0b: Classify intent — can we reuse cached data? ─────────────
    if last_turn and _looks_like_reuse_intent(question):
        intent = "REUSE_DATA"
    else:
        intent = _classify_intent(question, history, last_turn)

    # ── Fast path: pure elaboration/explanation on same data ─────────────
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
        return answer

    # ── Everything else (including scope-changing follow-ups like
    #    "what about rep 2?") goes through the full SQL pipeline
    #    using the rewritten standalone question. ─────────────────────────

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
        return answer

    # ── Step 1: Planner ──────────────────────────────────────────────────────
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
            detail = detail[len("[LLM error: ") : -1].strip()
        if prov == "ollama":
            return (
                "Sorry, I couldn't reach the local LLM. Is Ollama running? "
                "(`ollama serve`) Also check `model` in config.ini.\n\n"
                f"Details: {detail}"
            )
        return (
            "Sorry, the cloud LLM request failed. Check API keys in `.env`, "
            "`provider` and `model` in `config.ini`, and restart the server after changes.\n\n"
            f"Details: {detail}"
        )

    # ── Step 2: Route ────────────────────────────────────────────────────────
    if "COMPLEX" in plan.upper():
        sub_questions = parse_sub_questions(plan)
    else:
        sub_questions = [resolved]

    # ── Step 3: SQL Execution Loop ───────────────────────────────────────────
    results = {}
    for i, sub_q in enumerate(sub_questions):
        step_key = f"step_{i + 1}"
        results[step_key] = _sql_step_result(
            sql_system, sub_q, format_results(results)
        )

    # ── Step 4: Synthesize ───────────────────────────────────────────────────
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

    return answer
