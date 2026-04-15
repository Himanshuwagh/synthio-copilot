# app.py — FastAPI web server wrapping the pharma sales agent.
#
# Startup:  loads all CSVs into DuckDB and initialises the agent.
# Routes:
#   GET  /            → serve static/index.html
#   POST /chat        → run agent pipeline, return JSON {answer}
#   POST /chat/stream → same pipeline, Server-Sent Events (status + token + done)

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import agent
from config import get_supabase_settings, reset_llm_settings_cache
from context import load_history, save_turn
from db import build_sql_system_context, get_connection

_static = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading data and initialising agent…", flush=True)
    reset_llm_settings_cache()
    conn = get_connection()
    schema = build_sql_system_context(conn)
    agent.init(conn, schema)
    print("✓ Ready", flush=True)
    yield


app = FastAPI(title="Pharma Sales Co-pilot", lifespan=lifespan)


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = _static / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/favicon.ico")
async def favicon() -> Response:
    # Keep logs clean if no favicon asset is present.
    return Response(status_code=204)


@app.get("/history")
async def history_endpoint(
    session_id: str = Query(..., min_length=1, max_length=256),
) -> JSONResponse:
    if not get_supabase_settings():
        return JSONResponse({"messages": []})
    try:
        messages = load_history(last_n=20, session_id=session_id.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse({"messages": messages})


@app.post("/chat")
async def chat_endpoint(req: ChatRequest) -> JSONResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    supabase_on = get_supabase_settings() is not None
    sid = (req.session_id or "").strip()
    if supabase_on:
        if not sid:
            raise HTTPException(
                status_code=400,
                detail="session_id is required when Supabase is configured",
            )
        # History always from Supabase for this session — never trust client body.
        history = load_history(last_n=10, session_id=sid)
    else:
        history = req.history

    # Run synchronously on the event-loop thread. The in-memory DuckDB
    # connection is created during startup on this same thread; DuckDB
    # does not allow using that connection from worker threads (hangs).
    try:
        result: dict = agent.run(
            req.message,
            history,
            session_id=sid or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer_text: str = result.get("answer", "")
    save_turn(req.message, answer_text, session_id=sid if supabase_on else None)

    # Return the full structured result so the UI can render interpretation,
    # alternative chips, collapsible SQL, and empty-result diagnostics.
    return JSONResponse(result)


@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest) -> StreamingResponse:
    """
    Server-Sent Events endpoint.  Emits newline-delimited `data: <json>` lines:
      {"type": "status", "message": "..."}   — pipeline progress
      {"type": "token",  "text":    "..."}   — one synthesizer token
      {"type": "done",   ...result keys...}  — final metadata (always last)
      {"type": "error",  "message": "..."}   — fatal error

    The frontend reads this with fetch() + ReadableStream so it can show live
    status updates and stream the answer token-by-token.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    supabase_on = get_supabase_settings() is not None
    sid = (req.session_id or "").strip()
    if supabase_on:
        if not sid:
            raise HTTPException(
                status_code=400,
                detail="session_id is required when Supabase is configured",
            )
        history = load_history(last_n=10, session_id=sid)
    else:
        history = req.history

    async def event_generator():
        final_answer = ""
        try:
            async for chunk in agent.run_stream(
                req.message, history, session_id=sid or None
            ):
                if chunk.get("type") == "done":
                    final_answer = chunk.get("answer", "")
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            # Save the turn to Supabase (or no-op when Supabase is off)
            save_turn(req.message, final_answer, session_id=sid if supabase_on else None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # tell nginx/proxies not to buffer SSE
            "Connection": "keep-alive",
        },
    )
