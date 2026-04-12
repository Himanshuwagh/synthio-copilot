# app.py — FastAPI web server wrapping the pharma sales agent.
#
# Startup:  loads all CSVs into DuckDB and initialises the agent.
# Routes:
#   GET  /      → serve static/index.html
#   POST /chat  → run agent pipeline, return JSON {answer}

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
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
        answer: str = agent.run(
            req.message,
            history,
            session_id=sid or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    save_turn(req.message, answer, session_id=sid if supabase_on else None)
    return JSONResponse({"answer": answer})
