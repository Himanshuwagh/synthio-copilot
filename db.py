# db.py — DuckDB setup, view creation, and runtime schema injection.
# All CSVs are loaded into an in-memory DuckDB connection at startup.
# Two pre-materialized views prevent the fan-out problem when joining
# multi-grain fact tables (daily activity vs monthly prescriptions).

import os
import duckdb

# CSVs live in the data folder next to this script
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# All 9 source CSV tables
TABLES = [
    "rep_dim",
    "territory_dim",
    "account_dim",
    "hcp_dim",
    "date_dim",
    "fact_rx",
    "fact_rep_activity",
    "fact_ln_metrics",
    "fact_payor_mix",
]


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Load all CSVs into an in-memory DuckDB connection and create views.
    Returns the connection for use throughout the app.
    """
    conn = duckdb.connect()  # in-memory, no file on disk

    # Load each CSV as a table
    for table in TABLES:
        path = os.path.join(DATA_DIR, f"{table}.csv")
        conn.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto('{path}')"
        )

    # ── View 1: rep_hcp_monthly ──────────────────────────────────────────────
    # Aggregates daily rep activity to monthly grain BEFORE joining with HCP/date.
    # This prevents the fan-out problem that occurs when joining daily activity
    # with monthly prescription data (fact_rx). Each row = one rep x one HCP
    # x one month x one account combination.
    conn.execute("""
        CREATE VIEW rep_hcp_monthly AS
        SELECT
            a.rep_id,
            r.first_name || ' ' || r.last_name          AS rep_name,
            a.hcp_id,
            h.full_name                                  AS hcp_name,
            h.specialty,
            h.tier,
            h.territory_id,
            t.name                                       AS territory_name,
            a.account_id,
            ac.name                                      AS account_name,
            ac.account_type,
            d.year,
            d.quarter,
            LEFT(CAST(d.calendar_date AS VARCHAR), 7)    AS month,
            MAX(CAST(d.calendar_date AS VARCHAR))         AS last_visit_date,
            COUNT(*)                                     AS total_visits,
            COUNT(CASE WHEN a.status = 'completed'       THEN 1 END) AS completed_visits,
            COUNT(CASE WHEN a.status = 'cancelled'       THEN 1 END) AS cancelled_visits,
            COUNT(CASE WHEN a.status = 'scheduled'       THEN 1 END) AS scheduled_visits,
            COUNT(CASE WHEN a.activity_type = 'lunch_meeting' THEN 1 END) AS lunch_meetings,
            COUNT(CASE WHEN a.activity_type = 'call'     THEN 1 END) AS calls,
            ROUND(AVG(a.duration_min), 1)                AS avg_duration_min
        FROM fact_rep_activity a
        JOIN rep_dim        r  ON a.rep_id     = r.rep_id
        JOIN hcp_dim        h  ON a.hcp_id     = h.hcp_id
        JOIN account_dim    ac ON a.account_id = ac.account_id
        JOIN territory_dim  t  ON h.territory_id = t.territory_id
        JOIN date_dim       d  ON a.date_id    = d.date_id
        GROUP BY
            a.rep_id, rep_name, a.hcp_id, hcp_name,
            h.specialty, h.tier, h.territory_id, territory_name,
            a.account_id, account_name, ac.account_type,
            d.year, d.quarter, month
    """)

    # ── View 2: hcp_rx_monthly ───────────────────────────────────────────────
    # Joins prescription data with HCP and territory details.
    # Same monthly grain as fact_rx — safe to query without fan-out.
    # Do NOT join with rep_hcp_monthly here; keep grains separate.
    conn.execute("""
        CREATE VIEW hcp_rx_monthly AS
        SELECT
            rx.hcp_id,
            h.full_name                                  AS hcp_name,
            h.specialty,
            h.tier,
            h.territory_id,
            t.name                                       AS territory_name,
            d.year,
            d.quarter,
            LEFT(CAST(d.calendar_date AS VARCHAR), 7)    AS month,
            rx.brand_code,
            rx.trx_cnt,
            rx.nrx_cnt
        FROM fact_rx rx
        JOIN hcp_dim       h ON rx.hcp_id      = h.hcp_id
        JOIN territory_dim t ON h.territory_id  = t.territory_id
        JOIN date_dim      d ON rx.date_id      = d.date_id
    """)

    # ── View 3: hcp_market_quarterly ─────────────────────────────────────────
    # Pre-resolves the polymorphic join in fact_ln_metrics (entity_type='H').
    # Gives the LLM a clean table with HCP names, territories, and market
    # metrics — no need to understand entity_type/entity_id mapping.
    conn.execute("""
        CREATE VIEW hcp_market_quarterly AS
        SELECT
            h.hcp_id,
            h.full_name                     AS hcp_name,
            h.specialty,
            h.tier,
            h.territory_id,
            t.name                          AS territory_name,
            m.quarter_id,
            m.ln_patient_cnt,
            m.est_market_share
        FROM fact_ln_metrics m
        JOIN hcp_dim        h ON m.entity_id = h.hcp_id
        JOIN territory_dim  t ON h.territory_id = t.territory_id
        WHERE m.entity_type = 'H'
    """)

    # ── View 4: account_market_quarterly ──────────────────────────────────────
    # Same pattern for account-level LN metrics (entity_type='A').
    conn.execute("""
        CREATE VIEW account_market_quarterly AS
        SELECT
            a.account_id,
            a.name                          AS account_name,
            a.account_type,
            a.territory_id,
            t.name                          AS territory_name,
            m.quarter_id,
            m.ln_patient_cnt,
            m.est_market_share
        FROM fact_ln_metrics m
        JOIN account_dim    a ON m.entity_id = a.account_id
        JOIN territory_dim  t ON a.territory_id = t.territory_id
        WHERE m.entity_type = 'A'
    """)

    return conn


def build_sql_system_context(conn: duckdb.DuckDBPyConnection) -> str:
    """
    Return the runtime schema string injected into the SQL system prompt.
    Called once at startup; the result is passed to agent.init().
    """
    return get_schema(conn)


def gather_empty_sql_diagnostics(conn: duckdb.DuckDBPyConnection, sql: str) -> str:
    """
    Generate a human-readable diagnostic string for a query that returned no rows.
    Checks row counts in every referenced table/view and surfaces the active WHERE
    conditions so the LLM (and the user) can reason about why the result was empty.
    """
    import re as _re

    lines: list[str] = []

    # Extract table / view names that appear after FROM or JOIN
    table_pattern = _re.compile(r'\b(?:FROM|JOIN)\s+(\w+)', _re.IGNORECASE)
    tables = {m.group(1).lower() for m in table_pattern.finditer(sql)}

    if tables:
        lines.append("Row counts in referenced tables/views:")
        for table in sorted(tables):
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                lines.append(f"  - {table}: {count:,} rows")
            except Exception:
                lines.append(f"  - {table}: (unavailable)")

    # Surface the WHERE clause so the LLM can spot over-restrictive filters
    where_match = _re.search(
        r'\bWHERE\b(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|$)',
        sql, _re.IGNORECASE | _re.DOTALL,
    )
    if where_match:
        condition = " ".join(where_match.group(1).split())
        lines.append(f"Active filters: {condition[:300]}")

    return "\n".join(lines) if lines else "No diagnostic information available."


def get_schema(conn: duckdb.DuckDBPyConnection) -> str:
    """
    Query DuckDB at runtime to get real column names for all tables and views.
    Injected into SYSTEM_PROMPT so the LLM always has the correct schema.

    Why runtime instead of hardcoded: hardcoded column names silently generate
    wrong SQL if there's any mismatch. This is cheap insurance — runs once at startup.
    """
    # Get all table and view names
    all_names = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]

    # Views first (preferred for querying), then raw tables alphabetically
    view_priority = [
        "rep_hcp_monthly",
        "hcp_rx_monthly",
        "hcp_market_quarterly",
        "account_market_quarterly",
    ]
    lines = []

    for view in view_priority:
        if view in all_names:
            cols = conn.execute(f"DESCRIBE {view}").fetchall()
            col_names = [c[0] for c in cols]
            lines.append(f"VIEW {view}: {', '.join(col_names)}")

    for name in sorted(all_names):
        if name not in view_priority:
            cols = conn.execute(f"DESCRIBE {name}").fetchall()
            col_names = [c[0] for c in cols]
            lines.append(f"TABLE {name}: {', '.join(col_names)}")

    return "\n".join(lines)
