# db.py — DuckDB setup, view creation, and runtime schema injection.
# All CSVs are loaded into an in-memory DuckDB connection at startup.
# Two pre-materialized views prevent the fan-out problem when joining
# multi-grain fact tables (daily activity vs monthly prescriptions).

import os
import re
import duckdb

# Semantic hints for SQL generation (paired with runtime DESCRIBE output).
# Keys use "table_or_view.column" — helps map natural language to real names.
COLUMN_DESCRIPTIONS = {
    # --- rep_dim.csv ---
    "rep_dim.rep_id": "Primary key; joins to fact_rep_activity.rep_id",
    "rep_dim.first_name": "Rep first name",
    "rep_dim.last_name": "Rep last name",
    "rep_dim.region": "Region label (matches territory_dim.name / rep territory)",
    # --- territory_dim.csv ---
    "territory_dim.territory_id": "Primary key; joins hcp_dim / account_dim / facts",
    "territory_dim.name": "Territory display name (e.g. Territory 1)",
    "territory_dim.geo_type": "Geography roll-up type (e.g. State Cluster, Metro Area)",
    "territory_dim.parent_territory_id": "Parent territory if nested; often empty",
    # --- account_dim.csv ---
    "account_dim.account_id": "Primary key; joins fact_rep_activity, fact_payor_mix",
    "account_dim.name": "Account / site name",
    "account_dim.account_type": "e.g. Hospital, Clinic",
    "account_dim.address": "Location string",
    "account_dim.territory_id": "FK to territory_dim",
    # --- hcp_dim.csv ---
    "hcp_dim.hcp_id": "Primary key; joins fact_rx, fact_rep_activity, market views",
    "hcp_dim.full_name": "HCP display name",
    "hcp_dim.specialty": "Medical specialty",
    "hcp_dim.tier": "Segment tier: A, B, or C",
    "hcp_dim.territory_id": "FK to territory_dim",
    # --- date_dim.csv ---
    "date_dim.date_id": "Primary key YYYYMMDD integer; joins all facts on activity/Rx day",
    "date_dim.calendar_date": "Date value",
    "date_dim.year": "Calendar year",
    "date_dim.quarter": "Calendar quarter label",
    "date_dim.week_num": "Week of year",
    "date_dim.day_of_week": "Weekday label or code (per data)",
    # --- fact_rx.csv ---
    "fact_rx.hcp_id": "FK to hcp_dim",
    "fact_rx.date_id": "FK to date_dim",
    "fact_rx.brand_code": "Product / brand identifier",
    "fact_rx.trx_cnt": "Total prescriptions (TRx) for that HCP-day-brand",
    "fact_rx.nrx_cnt": "New prescriptions (NRx)",
    # --- fact_rep_activity.csv ---
    "fact_rep_activity.activity_id": "Surrogate key for one activity row",
    "fact_rep_activity.rep_id": "FK to rep_dim",
    "fact_rep_activity.hcp_id": "FK to hcp_dim",
    "fact_rep_activity.account_id": "FK to account_dim",
    "fact_rep_activity.date_id": "FK to date_dim",
    "fact_rep_activity.activity_type": "e.g. call, lunch_meeting",
    "fact_rep_activity.status": "scheduled | completed | cancelled",
    "fact_rep_activity.time_of_day": "Time bucket or label for the touch",
    "fact_rep_activity.duration_min": "Activity length in minutes",
    # --- fact_ln_metrics.csv (prefer hcp_market_quarterly / account_market_quarterly) ---
    "fact_ln_metrics.entity_type": "H = HCP-level row; A = account-level row",
    "fact_ln_metrics.entity_id": "hcp_id when entity_type=H; account_id when entity_type=A",
    "fact_ln_metrics.quarter_id": "Quarter bucket e.g. 2024Q4",
    "fact_ln_metrics.ln_patient_cnt": "LN patient volume for entity in quarter",
    "fact_ln_metrics.est_market_share": "Estimated market share % (synonym: 'market share')",
    # --- fact_payor_mix.csv ---
    "fact_payor_mix.account_id": "FK to account_dim",
    "fact_payor_mix.date_id": "FK to date_dim (mix as of this day)",
    "fact_payor_mix.payor_type": "Payor channel / type label",
    "fact_payor_mix.pct_of_volume": "Share of volume for that payor at account-date",
    # --- VIEW rep_hcp_monthly (monthly rep×HCP×account activity) ---
    "rep_hcp_monthly.rep_id": "Sales rep id",
    "rep_hcp_monthly.rep_name": "Rep full name (from rep_dim)",
    "rep_hcp_monthly.hcp_id": "HCP id",
    "rep_hcp_monthly.hcp_name": "HCP name",
    "rep_hcp_monthly.specialty": "HCP specialty",
    "rep_hcp_monthly.tier": "HCP tier A/B/C",
    "rep_hcp_monthly.territory_id": "Territory id",
    "rep_hcp_monthly.territory_name": "Territory name",
    "rep_hcp_monthly.account_id": "Account id for this activity grain",
    "rep_hcp_monthly.account_name": "Account name",
    "rep_hcp_monthly.account_type": "Hospital / Clinic / etc.",
    "rep_hcp_monthly.year": "Calendar year of activity month",
    "rep_hcp_monthly.quarter": "Calendar quarter of activity",
    "rep_hcp_monthly.month": "Activity month string 'YYYY-MM'",
    "rep_hcp_monthly.last_visit_date": "Latest calendar date in that month for this grain",
    "rep_hcp_monthly.total_visits": "All activity rows counted in the month",
    "rep_hcp_monthly.completed_visits": "Rows with status completed",
    "rep_hcp_monthly.cancelled_visits": "Rows with status cancelled",
    "rep_hcp_monthly.scheduled_visits": "Rows with status scheduled",
    "rep_hcp_monthly.lunch_meetings": "Rows with activity_type lunch_meeting",
    "rep_hcp_monthly.calls": "Rows with activity_type call",
    "rep_hcp_monthly.avg_duration_min": "Mean duration_min for rows in the bucket",
    # --- VIEW hcp_rx_monthly ---
    "hcp_rx_monthly.hcp_id": "HCP id",
    "hcp_rx_monthly.hcp_name": "HCP name",
    "hcp_rx_monthly.specialty": "Specialty",
    "hcp_rx_monthly.tier": "Tier",
    "hcp_rx_monthly.territory_id": "Territory id",
    "hcp_rx_monthly.territory_name": "Territory name",
    "hcp_rx_monthly.year": "Year of Rx month",
    "hcp_rx_monthly.quarter": "Quarter of Rx month",
    "hcp_rx_monthly.month": "Rx month 'YYYY-MM'",
    "hcp_rx_monthly.brand_code": "Brand / product code",
    "hcp_rx_monthly.trx_cnt": "Total Rx count for month grain",
    "hcp_rx_monthly.nrx_cnt": "New Rx count for month grain",
    # --- VIEW hcp_market_quarterly (HCP market share — preferred for share questions) ---
    "hcp_market_quarterly.hcp_id": "HCP id; join to hcp_dim / rep_hcp_monthly",
    "hcp_market_quarterly.hcp_name": "HCP display name",
    "hcp_market_quarterly.specialty": "Medical specialty",
    "hcp_market_quarterly.tier": "HCP tier A/B/C",
    "hcp_market_quarterly.territory_id": "Territory id",
    "hcp_market_quarterly.territory_name": "Territory label",
    "hcp_market_quarterly.quarter_id": "Quarter bucket e.g. 2024Q4",
    "hcp_market_quarterly.ln_patient_cnt": "LN patient volume in the quarter",
    "hcp_market_quarterly.est_market_share": "Estimated our brand share % ('market share')",
    # --- VIEW account_market_quarterly ---
    "account_market_quarterly.account_id": "Account id",
    "account_market_quarterly.account_name": "Account name",
    "account_market_quarterly.account_type": "Account type",
    "account_market_quarterly.territory_id": "Territory id",
    "account_market_quarterly.territory_name": "Territory name",
    "account_market_quarterly.quarter_id": "Quarter bucket e.g. 2024Q4",
    "account_market_quarterly.ln_patient_cnt": "LN patient volume at account in quarter",
    "account_market_quarterly.est_market_share": "Estimated market share % at account level",
}


_SQL_KW_AFTER_FROM = frozenset(
    {
        "SELECT",
        "LATERAL",
        "UNNEST",
    }
)

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


# Tables worth keeping per-column prose for SQL prompts (views + facts + date grain).
# Dimension tables (hcp_dim, rep_dim, …) rely on the authoritative DESCRIBE list only — saves tokens.
SQL_DICTIONARY_TABLES = frozenset(
    {
        "rep_hcp_monthly",
        "hcp_rx_monthly",
        "hcp_market_quarterly",
        "account_market_quarterly",
        "fact_rep_activity",
        "fact_rx",
        "fact_ln_metrics",
        "fact_payor_mix",
        "date_dim",
    }
)


def format_column_dictionary(only_tables=None) -> str:
    """Human-readable data dictionary. If only_tables is set, restrict to those table prefixes."""
    if only_tables is None:
        items = sorted(COLUMN_DESCRIPTIONS.items())
    else:
        items = sorted(
            (k, v)
            for k, v in COLUMN_DESCRIPTIONS.items()
            if k.split(".", 1)[0] in only_tables
        )
    lines = [f"- {key}: {desc}" for key, desc in items]
    return "\n".join(lines)


def get_data_profile(conn: duckdb.DuckDBPyConnection) -> str:
    """Compact facts about what values actually exist (reduces impossible filters)."""
    parts: list[str] = []

    try:
        rows = conn.execute(
            "SELECT DISTINCT quarter_id FROM hcp_market_quarterly ORDER BY 1"
        ).fetchall()
        qs = ", ".join(str(r[0]) for r in rows)
        parts.append(f"- quarter_id values in hcp_market_quarterly: {qs}")
    except Exception as e:
        parts.append(f"- quarters list: unavailable ({e})")

    try:
        lo, hi = conn.execute(
            "SELECT MIN(month), MAX(month) FROM rep_hcp_monthly"
        ).fetchone()
        parts.append(f"- month range in rep_hcp_monthly: {lo} .. {hi}")
    except Exception as e:
        parts.append(f"- month range: unavailable ({e})")

    try:
        tids = conn.execute(
            "SELECT DISTINCT territory_id FROM hcp_dim ORDER BY 1"
        ).fetchall()
        parts.append(
            "- territory_id values (hcp_dim): "
            + ", ".join(str(r[0]) for r in tids)
        )
    except Exception as e:
        parts.append(f"- territory list: unavailable ({e})")

    try:
        for label, table in (
            ("hcp_market_quarterly", "hcp_market_quarterly"),
            ("rep_hcp_monthly", "rep_hcp_monthly"),
            ("fact_rep_activity", "fact_rep_activity"),
        ):
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            parts.append(f"- row count {label}: {n}")
    except Exception as e:
        parts.append(f"- row counts: unavailable ({e})")

    return "\n".join(parts)


def build_sql_system_context(conn: duckdb.DuckDBPyConnection) -> str:
    """
    Context for SQL-generation calls only (not the planner). Tuned for token cost:
    - Full DESCRIBE output (authoritative column names — do not drop)
    - Data profile (small, high value)
    - Column glossary restricted to views + facts + date_dim; small dims omitted
      (their columns are still listed in the DESCRIBE section)
    """
    return (
        "=== DATA PROFILE (what exists in this database) ===\n"
        f"{get_data_profile(conn)}\n\n"
        "=== COLUMN GLOSSARY (semantic hints; not exhaustive — all names below in TABLES) ===\n"
        f"{format_column_dictionary(SQL_DICTIONARY_TABLES)}\n\n"
        "Note: account_dim, hcp_dim, rep_dim, territory_dim — use exact columns from "
        "TABLES AND COLUMNS; join keys match *_id fields.\n\n"
        "=== TABLES AND COLUMNS (exact names — authoritative) ===\n"
        f"{get_schema(conn)}"
    )


def build_llm_schema_context(conn: duckdb.DuckDBPyConnection) -> str:
    """Alias for `build_sql_system_context` (same string)."""
    return build_sql_system_context(conn)


def extract_sql_tables(sql: str) -> list[str]:
    """Best-effort table/view names from FROM/JOIN (DuckDB-style SQL)."""
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(
        r"\b(?:FROM|JOIN)\s+(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)",
        sql,
        re.IGNORECASE,
    ):
        name = m.group(1)
        if name.upper() in _SQL_KW_AFTER_FROM:
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _is_safe_sql_identifier(name: str) -> bool:
    return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name))


def gather_empty_sql_diagnostics(conn: duckdb.DuckDBPyConnection, sql: str) -> str:
    """
    When a query returns 0 rows, summarize whether tables are empty and show samples.
    """
    tables = extract_sql_tables(sql)
    if not tables:
        return "Could not infer table names from the SQL to run diagnostics."

    lines: list[str] = []
    known = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}

    for t in tables[:6]:
        if not _is_safe_sql_identifier(t):
            lines.append(f"- {t}: skipped (invalid identifier)")
            continue
        if t not in known:
            lines.append(
                f"- {t}: not found in SHOW TABLES — may be a subquery alias or typo"
            )
            continue
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            lines.append(f"- {t}: {n} total rows")
            if n:
                sample = conn.execute(f"SELECT * FROM {t} LIMIT 3").df()
                lines.append(sample.to_string(index=False))
        except Exception as e:
            lines.append(f"- {t}: diagnostic query failed ({e})")

    return "\n".join(lines)
