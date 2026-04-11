# context.py — Conversation history: Supabase when configured, else local CSV.

from __future__ import annotations

import csv
import os
from datetime import datetime

from config import get_supabase_settings

CONV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations")
HISTORY_FILE = os.path.join(CONV_DIR, "history.csv")

_supabase_client = None


def _default_session_id() -> str:
    return os.environ.get("CHAT_SESSION_ID", "cli").strip() or "cli"


def _get_supabase():
    global _supabase_client
    settings = get_supabase_settings()
    if not settings:
        return None
    if _supabase_client is None:
        from supabase import create_client

        _supabase_client = create_client(settings.url, settings.service_role_key)
    return _supabase_client


def load_history(last_n: int = 10, session_id: str | None = None) -> list:
    """Load the last `last_n` messages as [{role, message}, ...] in conversation order."""
    sid = (session_id or _default_session_id()).strip() or _default_session_id()
    client = _get_supabase()
    if client:
        # Sort by seq DESC (guaranteed insertion order), then reverse so oldest is first.
        # Falls back to created_at if seq column hasn't been added yet.
        try:
            resp = (
                client.table("chat_messages")
                .select("role,message,seq")
                .eq("session_id", sid)
                .order("seq", desc=True)
                .limit(last_n)
                .execute()
            )
        except Exception:
            # seq column not yet migrated — fall back to created_at
            resp = (
                client.table("chat_messages")
                .select("role,message,created_at")
                .eq("session_id", sid)
                .order("created_at", desc=True)
                .limit(last_n)
                .execute()
            )
        rows = list(reversed(resp.data or []))
        return [{"role": r["role"], "message": r["message"]} for r in rows]

    if not os.path.exists(HISTORY_FILE):
        return []
    out = []
    with open(HISTORY_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append({"role": row["role"], "message": row["message"]})
    return out[-last_n:]


def save_turn(
    user_msg: str, assistant_msg: str, session_id: str | None = None
) -> None:
    """Persist one Q&A turn (user row then assistant row)."""
    sid = (session_id or _default_session_id()).strip() or _default_session_id()
    client = _get_supabase()
    if client:
        # Insert user first, assistant second — seq column assigns consecutive
        # integers automatically so reload order is always guaranteed.
        client.table("chat_messages").insert(
            [
                {"session_id": sid, "role": "user", "message": user_msg},
                {"session_id": sid, "role": "assistant", "message": assistant_msg},
            ]
        ).execute()
        return

    os.makedirs(CONV_DIR, exist_ok=True)
    file_exists = os.path.exists(HISTORY_FILE)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "role", "message"])
        writer.writerow([now, "user", user_msg])
        writer.writerow([now, "assistant", assistant_msg])
