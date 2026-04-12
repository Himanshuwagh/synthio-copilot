# main.py — Terminal REPL entry point.
# General-purpose analytics chat — no rep login required.

import sys
import agent
from db import build_sql_system_context, get_connection
from context import load_history, save_turn

BANNER = """
╔══════════════════════════════════════════════╗
║    Pharma Sales Co-pilot  —  SynthioLabs     ║
║    Powered by local LLM (Ollama)             ║
╚══════════════════════════════════════════════╝
Ask about any rep, HCP, territory, Rx, visits, or accounts.
Type 'quit' to exit.
"""

DIVIDER = "─" * 48


def main():
    print(BANNER)

    # ── 1. Connect to DB and build schema ────────────────────────────────────
    print("Loading data...", end=" ", flush=True)
    try:
        conn = get_connection()
        schema = build_sql_system_context(conn)
        print("✓ Ready\n")
    except Exception as e:
        print(f"\n✗ Failed to load data: {e}")
        sys.exit(1)

    # ── 2. Initialise agent + validate LLM config ────────────────────────────
    try:
        from config import get_llm_settings

        llm = get_llm_settings()
        print(f"LLM: {llm.provider} / {llm.model}")
    except RuntimeError as e:
        print(f"\n✗ {e}")
        sys.exit(1)

    agent.init(conn, schema)

    # ── 3. Load conversation history ─────────────────────────────────────────
    history = load_history()

    print(DIVIDER)

    # ── 4. REPL loop ─────────────────────────────────────────────────────────
    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q", "bye"):
            break

        print("\n[Thinking...]\n")

        answer = agent.run(question, history)

        print(f"Co-pilot: {answer}")
        print(f"\n{DIVIDER}")

        save_turn(question, answer)

        history.append({"role": "user",      "message": question})
        history.append({"role": "assistant", "message": answer})
        history = history[-10:]

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
