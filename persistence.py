"""Persistence for handicapping sessions and cumulative horse database.

Supports SQLite (default, local dev) and PostgreSQL (production).
Set DATABASE_URL env var to use Postgres; otherwise falls back to SQLite.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


# ---------------------------------------------------------------------------
# Postgres compatibility layer
# ---------------------------------------------------------------------------

def _translate_sql(sql: str) -> str:
    """Translate SQLite SQL dialect to Postgres."""
    # Parameter placeholders: ? -> %s
    sql = sql.replace("?", "%s")
    # AUTOINCREMENT -> Postgres SERIAL
    sql = re.sub(
        r"INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT",
        "SERIAL PRIMARY KEY",
        sql,
        flags=re.IGNORECASE,
    )
    # INSERT OR IGNORE -> INSERT ... ON CONFLICT DO NOTHING
    if re.search(r"INSERT\s+OR\s+IGNORE\s+INTO", sql, re.IGNORECASE):
        sql = re.sub(r"INSERT\s+OR\s+IGNORE\s+INTO", "INSERT INTO", sql, flags=re.IGNORECASE)
        sql = sql.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
    return sql


class _PgCursorResult:
    """Wraps a psycopg2 cursor to provide sqlite3-compatible attributes."""

    def __init__(self, cursor, lastrowid: Optional[int] = None):
        self._cursor = cursor
        self._lastrowid = lastrowid

    @property
    def lastrowid(self) -> Optional[int]:
        return self._lastrowid

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def __iter__(self):
        return iter(self._cursor)


class _PgConnectionWrapper:
    """Wraps a psycopg2 connection so Persistence can use the same API as sqlite3."""

    def __init__(self, pg_conn, cursor_factory):
        self._conn = pg_conn
        self._cursor_factory = cursor_factory
        self._last_rowcount = 0

    def execute(self, sql, params=None):
        sql = _translate_sql(sql)
        cur = self._conn.cursor(cursor_factory=self._cursor_factory)

        # Handle SQLite's SELECT changes() — return previous rowcount
        if "select lastval()" in sql.lower() or "select changes()" in sql.lower():
            # Return the stored rowcount from previous UPDATE
            class _FakeRow:
                def __init__(self, val):
                    self._val = val
                def __getitem__(self, idx):
                    return self._val
            class _FakeResult:
                lastrowid = None
                rowcount = 1
                def __init__(self, val):
                    self._val = val
                def fetchone(self):
                    return _FakeRow(self._val)
                def fetchall(self):
                    return [_FakeRow(self._val)]
            return _FakeResult(self._last_rowcount)

        cur.execute(sql, params or ())
        self._last_rowcount = cur.rowcount

        # For INSERT, retrieve the auto-generated serial value via lastval()
        _lastrowid = None
        is_insert = sql.strip().upper().startswith("INSERT")
        if is_insert and cur.rowcount and cur.rowcount > 0:
            try:
                lv_cur = self._conn.cursor()
                lv_cur.execute("SELECT lastval()")
                _lastrowid = lv_cur.fetchone()[0]
                lv_cur.close()
            except Exception:
                _lastrowid = None

        return _PgCursorResult(cur, _lastrowid)

    def executescript(self, sql):
        """Execute multiple SQL statements separated by semicolons."""
        cur = self._conn.cursor(cursor_factory=self._cursor_factory)
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(_translate_sql(stmt))
        return cur

    def commit(self):
        self._conn.commit()

    @property
    def total_changes(self):
        return self._last_rowcount


SESSION_COLUMNS = [
    "session_id",
    "track_name",
    "race_date",
    "pdf_name",
    "created_at",
    "parser_used",
    "horses_count",
    "total_races",
    "parser_meta",
]

# Reverse lookup: full track name -> short code
_TRACK_FULL_TO_CODE = {
    'Churchill Downs': 'CD', 'Santa Anita': 'SA', 'Gulfstream Park': 'GP',
    'Aqueduct': 'AQ', 'Belmont': 'BEL', 'Saratoga': 'SAR',
    'Keeneland': 'KEE', 'Del Mar': 'DM', 'Oaklawn Park': 'OP',
    'Tampa Bay Downs': 'TAM', 'Fair Grounds': 'FG',
    'Laurel Park': 'LRL', 'Parx Racing': 'PRX', 'Monmouth Park': 'MTH',
    'Pimlico': 'PIM', 'Woodbine': 'WO', 'Ellis Park': 'ELP',
    'Canterbury Park': 'CBY', 'Golden Gate Fields': 'GG',
    'Los Alamitos': 'LA', 'Turfway Park': 'TP',
}


class Persistence:
    def __init__(self, db_path: Path = None):
        database_url = os.environ.get("DATABASE_URL")
        if database_url:
            import psycopg2
            import psycopg2.extras
            # Railway gives postgres:// but psycopg2 requires postgresql://
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            try:
                pg_conn = psycopg2.connect(database_url, connect_timeout=5)
            except Exception as exc:
                import sys
                print(f"[persistence] FATAL: Cannot connect to Postgres "
                      f"(timeout 5s): {exc}", file=sys.stderr, flush=True)
                raise
            pg_conn.autocommit = False
            self.conn = _PgConnectionWrapper(pg_conn, psycopg2.extras.DictCursor)
            self.db_backend = "postgres"
            self.db_path = None
        else:
            self.db_path = Path(db_path or "horses.db")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.db_backend = "sqlite"
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                track_name TEXT,
                race_date TEXT,
                pdf_name TEXT,
                created_at TEXT,
                parser_used TEXT,
                horses_count INTEGER,
                total_races INTEGER,
                parser_meta TEXT
            );

            CREATE TABLE IF NOT EXISTS race_inputs (
                session_id TEXT,
                race_no TEXT,
                data_json TEXT,
                updated_at TEXT,
                PRIMARY KEY (session_id, race_no)
            );

            CREATE TABLE IF NOT EXISTS race_outputs (
                session_id TEXT,
                race_no TEXT,
                data_json TEXT,
                updated_at TEXT,
                PRIMARY KEY (session_id, race_no)
            );

            CREATE TABLE IF NOT EXISTS race_results (
                session_id TEXT,
                race_no TEXT,
                data_json TEXT,
                updated_at TEXT,
                PRIMARY KEY (session_id, race_no)
            );

            CREATE TABLE IF NOT EXISTS uploads (
                upload_id    TEXT PRIMARY KEY,
                source_type  TEXT NOT NULL,
                pdf_filename TEXT NOT NULL,
                pdf_hash     TEXT,
                track        TEXT,
                race_date    TEXT,
                session_id   TEXT,
                uploaded_at  TEXT NOT NULL,
                horses_count INTEGER DEFAULT 0,
                status       TEXT DEFAULT 'completed'
            );

            CREATE TABLE IF NOT EXISTS sheets_horses (
                horse_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id       TEXT NOT NULL REFERENCES uploads(upload_id),
                horse_name      TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                race_number     INTEGER,
                post            TEXT,
                sex             TEXT,
                age             INTEGER DEFAULT 0,
                top_fig         REAL,
                track           TEXT,
                race_date       TEXT,
                sire            TEXT,
                dam             TEXT,
                trainer         TEXT,
                UNIQUE(normalized_name, track, race_date, race_number)
            );
            CREATE INDEX IF NOT EXISTS idx_sh_norm_name ON sheets_horses(normalized_name);
            CREATE INDEX IF NOT EXISTS idx_sh_track_date ON sheets_horses(track, race_date);

            CREATE TABLE IF NOT EXISTS sheets_lines (
                line_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                horse_id      INTEGER NOT NULL REFERENCES sheets_horses(horse_id),
                line_index    INTEGER NOT NULL,
                parsed_figure REAL,
                raw_text      TEXT,
                surface       TEXT,
                track         TEXT,
                flags         TEXT,
                data_text     TEXT,
                race_date     TEXT,
                race_type     TEXT,
                UNIQUE(horse_id, line_index)
            );
            CREATE INDEX IF NOT EXISTS idx_sl_horse ON sheets_lines(horse_id);
            CREATE INDEX IF NOT EXISTS idx_sl_surface ON sheets_lines(surface);

            CREATE TABLE IF NOT EXISTS brisnet_horses (
                brisnet_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id       TEXT NOT NULL REFERENCES uploads(upload_id),
                horse_name      TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                post            INTEGER,
                race_number     INTEGER,
                runstyle        TEXT,
                runstyle_rating INTEGER,
                prime_power     REAL,
                sire            TEXT,
                dam             TEXT,
                breeder         TEXT,
                weight          INTEGER,
                life_starts     INTEGER DEFAULT 0,
                life_record     TEXT,
                life_earnings   TEXT,
                life_speed      INTEGER,
                surface_records TEXT,
                track           TEXT,
                race_date       TEXT,
                restrictions    TEXT,
                age             INTEGER DEFAULT 0,
                UNIQUE(normalized_name, track, race_date, race_number)
            );
            CREATE INDEX IF NOT EXISTS idx_bh_norm_name ON brisnet_horses(normalized_name);
            CREATE INDEX IF NOT EXISTS idx_bh_track_date ON brisnet_horses(track, race_date);

            CREATE TABLE IF NOT EXISTS brisnet_lines (
                bline_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                brisnet_id INTEGER NOT NULL REFERENCES brisnet_horses(brisnet_id),
                line_index INTEGER NOT NULL,
                race_date  TEXT,
                track      TEXT,
                surface    TEXT,
                speed      INTEGER,
                race_type  TEXT,
                raw_text   TEXT,
                UNIQUE(brisnet_id, line_index)
            );
            CREATE INDEX IF NOT EXISTS idx_bl_brisnet ON brisnet_lines(brisnet_id);

            CREATE TABLE IF NOT EXISTS brisnet_workouts (
                workout_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                brisnet_id   INTEGER NOT NULL REFERENCES brisnet_horses(brisnet_id),
                workout_date TEXT,
                track        TEXT,
                distance     TEXT,
                surface      TEXT,
                time         TEXT,
                rank_letter  TEXT,
                rank         TEXT
            );

            CREATE TABLE IF NOT EXISTS reconciliation (
                recon_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                horse_id     INTEGER NOT NULL REFERENCES sheets_horses(horse_id),
                brisnet_id   INTEGER NOT NULL REFERENCES brisnet_horses(brisnet_id),
                match_method TEXT NOT NULL,
                confidence   TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                UNIQUE(horse_id, brisnet_id)
            );
            CREATE INDEX IF NOT EXISTS idx_recon_horse ON reconciliation(horse_id);
            CREATE INDEX IF NOT EXISTS idx_recon_brisnet ON reconciliation(brisnet_id);

            CREATE TABLE IF NOT EXISTS horse_aliases (
                canonical_name TEXT NOT NULL,
                alias_name     TEXT NOT NULL,
                PRIMARY KEY (canonical_name, alias_name)
            );
            CREATE INDEX IF NOT EXISTS idx_alias_alias ON horse_aliases(alias_name);

            CREATE TABLE IF NOT EXISTS result_races (
                result_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                track        TEXT NOT NULL,
                race_date    TEXT NOT NULL,
                race_number  INTEGER NOT NULL,
                surface      TEXT,
                distance     TEXT,
                winner_post  INTEGER,
                winner_name  TEXT,
                imported_at  TEXT NOT NULL,
                UNIQUE(track, race_date, race_number)
            );
            CREATE INDEX IF NOT EXISTS idx_rr_track_date ON result_races(track, race_date);

            CREATE TABLE IF NOT EXISTS result_entries (
                entry_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                track           TEXT NOT NULL,
                race_date       TEXT NOT NULL,
                race_number     INTEGER NOT NULL,
                post            INTEGER NOT NULL,
                horse_name      TEXT,
                normalized_name TEXT,
                finish_pos      INTEGER,
                beaten_lengths  REAL,
                odds            REAL,
                win_payoff      REAL,
                place_payoff    REAL,
                show_payoff     REAL,
                sheets_horse_id  INTEGER,
                brisnet_horse_id INTEGER,
                projection_type  TEXT,
                pick_rank        INTEGER,
                session_id       TEXT,
                UNIQUE(track, race_date, race_number, post)
            );
            CREATE INDEX IF NOT EXISTS idx_re_track_date ON result_entries(track, race_date);
            CREATE INDEX IF NOT EXISTS idx_re_name ON result_entries(normalized_name);
            CREATE INDEX IF NOT EXISTS idx_re_key ON result_entries(track, race_date, race_number, post);

            CREATE TABLE IF NOT EXISTS result_predictions (
                prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                track           TEXT NOT NULL,
                race_date       TEXT NOT NULL,
                race_number     INTEGER NOT NULL,
                horse_name      TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                pick_rank       INTEGER NOT NULL,
                projection_type TEXT NOT NULL,
                bias_score      REAL,
                raw_score       REAL,
                confidence      REAL,
                projected_low   REAL,
                projected_high  REAL,
                tags            TEXT,
                new_top_setup   INTEGER DEFAULT 0,
                bounce_risk     INTEGER DEFAULT 0,
                tossed          INTEGER DEFAULT 0,
                saved_at        TEXT NOT NULL,
                UNIQUE(session_id, track, race_date, race_number, normalized_name)
            );
            CREATE INDEX IF NOT EXISTS idx_rp_session ON result_predictions(session_id);
            CREATE INDEX IF NOT EXISTS idx_rp_key ON result_predictions(track, race_date, race_number, normalized_name);

            CREATE TABLE IF NOT EXISTS bet_plans (
                plan_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id     TEXT NOT NULL,
                track          TEXT NOT NULL,
                race_date      TEXT NOT NULL,
                settings_json  TEXT NOT NULL,
                plan_json      TEXT NOT NULL,
                total_risk     REAL NOT NULL DEFAULT 0,
                paper_mode     INTEGER NOT NULL DEFAULT 1,
                engine_version TEXT NOT NULL DEFAULT '',
                created_at     TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_bp_session ON bet_plans(session_id);
            CREATE INDEX IF NOT EXISTS idx_bp_track_date ON bet_plans(track, race_date);

            CREATE TABLE IF NOT EXISTS odds_snapshots (
                snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                track           TEXT NOT NULL,
                race_date       TEXT NOT NULL,
                race_number     INTEGER NOT NULL,
                post            INTEGER,
                horse_name      TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                odds_raw        TEXT,
                odds_decimal    REAL,
                source          TEXT NOT NULL DEFAULT 'morning_line',
                captured_at     TEXT NOT NULL,
                UNIQUE(session_id, track, race_date, race_number, normalized_name, source)
            );
            CREATE INDEX IF NOT EXISTS idx_os_lookup
                ON odds_snapshots(track, race_date, race_number, normalized_name, source);

            CREATE TABLE IF NOT EXISTS trainer_bucket_stats (
                stat_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                trainer      TEXT NOT NULL,
                bucket_type  TEXT NOT NULL,
                bucket_value TEXT,
                starts       INTEGER DEFAULT 0,
                wins         INTEGER DEFAULT 0,
                itm          INTEGER DEFAULT 0,
                roi          REAL DEFAULT 0.0,
                updated_at   TEXT,
                UNIQUE(trainer, bucket_type)
            );
            """
        )
        self.conn.commit()
        self._ensure_columns()

    def _get_table_columns(self, table_name: str) -> set:
        """Return set of column names for a table (works on both backends)."""
        if self.db_backend == "postgres":
            cur = self.conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                (table_name,),
            )
            return {row[0] for row in cur.fetchall()}
        else:
            cur = self.conn.execute(f"PRAGMA table_info({table_name})")
            return {row[1] for row in cur.fetchall()}

    def _ensure_columns(self) -> None:
        # sessions table
        existing = self._get_table_columns("sessions")
        desired = {
            "parser_used": "TEXT",
            "horses_count": "INTEGER",
            "total_races": "INTEGER",
            "parser_meta": "TEXT",
        }
        for column, col_type in desired.items():
            if column not in existing:
                self.conn.execute(f"ALTER TABLE sessions ADD COLUMN {column} {col_type}")
        # result_entries table — add session_id if missing
        re_cols = self._get_table_columns("result_entries")
        if re_cols and "session_id" not in re_cols:
            self.conn.execute("ALTER TABLE result_entries ADD COLUMN session_id TEXT")
        # bet_plans table — add engine_version and plan_type if missing
        bp_cols = self._get_table_columns("bet_plans")
        if bp_cols and "engine_version" not in bp_cols:
            self.conn.execute("ALTER TABLE bet_plans ADD COLUMN engine_version TEXT DEFAULT ''")
        if bp_cols and "plan_type" not in bp_cols:
            self.conn.execute("ALTER TABLE bet_plans ADD COLUMN plan_type TEXT NOT NULL DEFAULT 'session'")
        # result_predictions — add post column for post-based matching
        rp_cols = self._get_table_columns("result_predictions")
        if rp_cols and "post" not in rp_cols:
            self.conn.execute("ALTER TABLE result_predictions ADD COLUMN post INTEGER")
        # result_races — add status column for parse error tracking
        rr_cols = self._get_table_columns("result_races")
        if rr_cols and "status" not in rr_cols:
            self.conn.execute("ALTER TABLE result_races ADD COLUMN status TEXT DEFAULT 'OK'")
        self.conn.commit()
        self._renormalize_names()

    def _renormalize_names(self) -> None:
        """Re-apply _normalize_name to all stored normalized_name columns.

        Needed after updating the normalization rules (e.g. adding Unicode
        quote stripping).  Idempotent — skips rows already correct.
        """
        for table, name_col, id_col in [
            ("sheets_horses", "horse_name", "horse_id"),
            ("brisnet_horses", "horse_name", "brisnet_id"),
            ("result_entries", "horse_name", "entry_result_id"),
        ]:
            rows = self.conn.execute(
                f"SELECT {id_col}, {name_col}, normalized_name FROM {table}"
            ).fetchall()
            updated = 0
            for r in rows:
                expected = self._normalize_name(r[name_col] or "")
                if r["normalized_name"] != expected:
                    self.conn.execute(
                        f"UPDATE {table} SET normalized_name = ? WHERE {id_col} = ?",
                        (expected, r[id_col]),
                    )
                    updated += 1
            if updated:
                self.conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Uppercase, strip punctuation (incl. Unicode quotes), collapse whitespace."""
        n = (name or "").upper().strip()
        # Strip ASCII and Unicode apostrophes/quotes and common punctuation
        n = re.sub(r"[\u2018\u2019\u201C\u201D'\-.,()\"']", "", n)
        n = re.sub(r"\s+", " ", n)
        return n

    @staticmethod
    def _pred_entry_join(pred: str = "rp", entry: str = "er",
                         join_type: str = "LEFT JOIN") -> str:
        """JOIN result_predictions to result_entries: post first, name fallback."""
        return f"""
            {join_type} result_entries {entry}
                ON {pred}.track = {entry}.track
                AND {pred}.race_date = {entry}.race_date
                AND {pred}.race_number = {entry}.race_number
                AND (
                    ({pred}.post IS NOT NULL AND {pred}.post = {entry}.post)
                    OR ({pred}.post IS NULL AND {pred}.normalized_name = {entry}.normalized_name)
                )
        """

    @staticmethod
    def _normalize_track(track: str) -> str:
        """Normalize track to short code. Handles both full names and codes."""
        t = (track or "").strip()
        if not t:
            return ""
        # Already a short code (2-4 uppercase letters)?
        if len(t) <= 5 and t.upper() == t:
            return t.upper()
        # Full name lookup
        code = _TRACK_FULL_TO_CODE.get(t)
        if code:
            return code
        # Case-insensitive search
        t_lower = t.lower()
        for full_name, code in _TRACK_FULL_TO_CODE.items():
            if full_name.lower() == t_lower:
                return code
        # Return as-is (uppercase)
        return t.upper()

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        """Normalize various date formats to MM/DD/YYYY when possible."""
        d = (date_str or "").strip()
        if not d or d == "Unknown Date":
            return d

        import re
        from datetime import datetime as _dt

        # Already MM/DD/YYYY
        if re.match(r"\d{2}/\d{2}/\d{4}$", d):
            return d

        # "February 19, 2026" or "Jan 5, 2025"
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y"):
            try:
                parsed = _dt.strptime(d, fmt)
                return parsed.strftime("%m/%d/%Y")
            except ValueError:
                pass

        # "29JAN" or "06JUN25" — short Ragozin format
        m = re.match(r"(\d{1,2})([A-Z]{3})(\d{2,4})?$", d, re.IGNORECASE)
        if m:
            day = int(m.group(1))
            mon_str = m.group(2).capitalize()
            year_str = m.group(3)
            months = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                       "Jul": 7, "Jly": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
            month = months.get(mon_str)
            if month:
                if year_str:
                    yr = int(year_str)
                    if yr < 100:
                        yr += 2000
                else:
                    from datetime import date as _date
                    yr = _date.today().year
                try:
                    return f"{month:02d}/{day:02d}/{yr}"
                except Exception:
                    pass

        return d  # return as-is if no pattern matches

    @staticmethod
    def _age_from_restrictions(restr: str) -> int:
        """Extract age from BRISNET race restrictions string."""
        restr = (restr or "").lower()
        m = re.search(r"(\d)yo", restr)
        if m:
            return int(m.group(1))
        if "3&up" in restr or "3 & up" in restr:
            return 3
        if "4&up" in restr or "4 & up" in restr:
            return 4
        return 0

    def resolve_alias(self, normalized_name: str) -> str:
        """If normalized_name is a known alias, return canonical; else return as-is."""
        row = self.conn.execute(
            "SELECT canonical_name FROM horse_aliases WHERE alias_name = ?",
            (normalized_name,),
        ).fetchone()
        return row[0] if row else normalized_name

    def add_alias(self, canonical: str, alias: str) -> bool:
        """Add a horse alias mapping. Returns True if inserted, False if exists."""
        canon_norm = self._normalize_name(canonical)
        alias_norm = self._normalize_name(alias)
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO horse_aliases(canonical_name, alias_name) VALUES(?, ?)",
                (canon_norm, alias_norm),
            )
            self.conn.commit()
            return self.conn.total_changes > 0
        except Exception:
            return False

    def list_aliases(self) -> List[Dict[str, str]]:
        rows = self.conn.execute(
            "SELECT canonical_name, alias_name FROM horse_aliases ORDER BY canonical_name"
        ).fetchall()
        return [{"canonical_name": r[0], "alias_name": r[1]} for r in rows]

    @staticmethod
    def _fuzzy_strip(name: str) -> str:
        """Extra-aggressive normalization for fuzzy matching.

        Strips country suffixes (-FR, -IR, -GB, -JPN, etc.) in addition
        to the standard normalization.
        """
        n = Persistence._normalize_name(name)
        # Strip trailing country codes (common in Ragozin sheets)
        n = re.sub(r"\s*-\s*[A-Z]{2,3}$", "", n)
        return n

    def suggest_matches(
        self, brisnet_name: str, track: str, race_date: str,
        race_number: int = 0, limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Suggest sheets_horses that might match a brisnet horse name.

        Uses fuzzy-stripped names and substring matching. Returns a list
        of ``{horse_name, normalized_name, horse_id, score}`` dicts
        sorted by descending score.  Score meanings:
          100 = exact fuzzy match (after stripping suffixes/quotes)
           80 = one name contains the other
           60 = shared-word overlap (Jaccard on words)
        """
        b_fuzzy = self._fuzzy_strip(brisnet_name)
        b_words = set(b_fuzzy.split())

        # Get candidate sheets horses from same track+date (optionally same race)
        where = "track = ? AND race_date = ?"
        params: list = [self._normalize_track(track), self._normalize_date(race_date)]
        if race_number:
            where += " AND race_number = ?"
            params.append(race_number)

        rows = self.conn.execute(
            f"SELECT horse_id, horse_name, normalized_name FROM sheets_horses WHERE {where}",
            params,
        ).fetchall()

        scored = []
        for r in rows:
            s_fuzzy = self._fuzzy_strip(r["horse_name"])
            if b_fuzzy == s_fuzzy:
                scored.append({**dict(r), "score": 100})
            elif b_fuzzy in s_fuzzy or s_fuzzy in b_fuzzy:
                scored.append({**dict(r), "score": 80})
            else:
                s_words = set(s_fuzzy.split())
                overlap = b_words & s_words
                union = b_words | s_words
                if overlap and len(overlap) / len(union) >= 0.4:
                    jaccard = int(len(overlap) / len(union) * 100)
                    scored.append({**dict(r), "score": jaccard})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def get_unmatched_with_suggestions(
        self, track: str = "", race_date: str = "",
    ) -> List[Dict[str, Any]]:
        """Return unmatched brisnet horses with fuzzy match suggestions."""
        track = self._normalize_track(track) if track else ""
        race_date = self._normalize_date(race_date) if race_date else ""

        where = "bh.brisnet_id NOT IN (SELECT brisnet_id FROM reconciliation)"
        params: list = []
        if track:
            where += " AND bh.track = ?"
            params.append(track)
        if race_date:
            where += " AND bh.race_date = ?"
            params.append(race_date)

        rows = self.conn.execute(f"""
            SELECT bh.brisnet_id, bh.horse_name, bh.normalized_name,
                   bh.race_number, bh.post, bh.track, bh.race_date
            FROM brisnet_horses bh WHERE {where}
            ORDER BY bh.race_number, bh.post
        """, params).fetchall()

        result = []
        for r in rows:
            suggestions = self.suggest_matches(
                r["horse_name"], r["track"], r["race_date"],
                race_number=r["race_number"],
            )
            result.append({
                "brisnet_id": r["brisnet_id"],
                "horse_name": r["horse_name"],
                "normalized_name": r["normalized_name"],
                "race_number": r["race_number"],
                "post": r["post"],
                "suggestions": suggestions,
            })
        return result

    def _prepare_session_row(self, row) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        data = dict(row)
        meta_raw = data.get("parser_meta")
        if meta_raw:
            try:
                data["parser_meta"] = json.loads(meta_raw)
            except json.JSONDecodeError:
                data["parser_meta"] = {}
        else:
            data["parser_meta"] = {}
        return data

    # ------------------------------------------------------------------
    # Session CRUD (existing)
    # ------------------------------------------------------------------

    def record_session(
        self,
        session_id: str,
        track_name: str,
        race_date: str,
        pdf_name: str,
        *,
        parser_used: str = "traditional",
        horses_count: int = 0,
        total_races: int = 0,
        parser_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        meta_json = json.dumps(parser_meta or {})
        self.conn.execute(
            """
            INSERT INTO sessions(session_id, track_name, race_date, pdf_name, created_at, parser_used, horses_count, total_races, parser_meta)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                track_name=excluded.track_name,
                race_date=excluded.race_date,
                pdf_name=excluded.pdf_name,
                parser_used=excluded.parser_used,
                horses_count=excluded.horses_count,
                total_races=excluded.total_races,
                parser_meta=excluded.parser_meta
            """,
            (
                session_id,
                track_name,
                race_date,
                pdf_name,
                now,
                parser_used,
                horses_count,
                total_races,
                meta_json,
            ),
        )
        self.conn.commit()

    def list_sessions(self) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT session_id, track_name, race_date, pdf_name, created_at, parser_used, horses_count, total_races, parser_meta FROM sessions ORDER BY created_at DESC"
        )
        rows = cur.fetchall()
        return [self._prepare_session_row(row) for row in rows if row]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT session_id, track_name, race_date, pdf_name, created_at, parser_used, horses_count, total_races, parser_meta FROM sessions WHERE session_id=?",
            (session_id,),
        )
        return self._prepare_session_row(cur.fetchone())

    def save_race_inputs(self, session_id: str, race_no: str, data: Dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO race_inputs(session_id, race_no, data_json, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(session_id, race_no) DO UPDATE SET
                data_json=excluded.data_json,
                updated_at=excluded.updated_at
            """,
            (session_id, race_no, json.dumps(data), now),
        )
        self.conn.commit()

    def load_race_inputs(self, session_id: str, race_no: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT data_json FROM race_inputs WHERE session_id=? AND race_no=?",
            (session_id, race_no),
        )
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def save_race_outputs(self, session_id: str, race_no: str, data: Dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO race_outputs(session_id, race_no, data_json, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(session_id, race_no) DO UPDATE SET
                data_json=excluded.data_json,
                updated_at=excluded.updated_at
            """,
            (session_id, race_no, json.dumps(data), now),
        )
        self.conn.commit()

    def load_race_outputs(self, session_id: str, race_no: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT data_json FROM race_outputs WHERE session_id=? AND race_no=?",
            (session_id, race_no),
        )
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def save_race_results(self, session_id: str, race_no: str, data: Dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO race_results(session_id, race_no, data_json, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(session_id, race_no) DO UPDATE SET
                data_json=excluded.data_json,
                updated_at=excluded.updated_at
            """,
            (session_id, race_no, json.dumps(data), now),
        )
        self.conn.commit()

    def load_race_results(self, session_id: str, race_no: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT data_json FROM race_results WHERE session_id=? AND race_no=?",
            (session_id, race_no),
        )
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return None

    # ==================================================================
    # Upload tracking
    # ==================================================================

    def record_upload(
        self,
        upload_id: str,
        source_type: str,
        pdf_filename: str,
        pdf_hash: str,
        track: str,
        race_date: str,
        session_id: Optional[str] = None,
        horses_count: int = 0,
    ) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO uploads(upload_id, source_type, pdf_filename, pdf_hash,
                                track, race_date, session_id, uploaded_at, horses_count)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(upload_id) DO NOTHING
            """,
            (upload_id, source_type, pdf_filename, pdf_hash,
             self._normalize_track(track), race_date, session_id, now, horses_count),
        )
        self.conn.commit()

    def get_upload_by_hash(self, pdf_hash: str) -> Optional[Dict[str, Any]]:
        if not pdf_hash:
            return None
        cur = self.conn.execute(
            "SELECT * FROM uploads WHERE pdf_hash = ?", (pdf_hash,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_uploads(self, source_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if source_type:
            cur = self.conn.execute(
                "SELECT * FROM uploads WHERE source_type = ? ORDER BY uploaded_at DESC",
                (source_type,),
            )
        else:
            cur = self.conn.execute(
                "SELECT * FROM uploads ORDER BY uploaded_at DESC"
            )
        return [dict(row) for row in cur.fetchall()]

    # ==================================================================
    # Sheets (Ragozin) ingestion
    # ==================================================================

    def ingest_sheets_horse(
        self,
        upload_id: str,
        horse_dict: Dict[str, Any],
        track: str,
        race_date: str,
    ) -> Optional[int]:
        """Insert a Ragozin horse + lines. Returns horse_id or None if duplicate."""
        name = horse_dict.get("horse_name", "")
        norm = self._normalize_name(name)
        track_code = self._normalize_track(track)
        race_date = self._normalize_date(race_date)
        race_num = horse_dict.get("race_number", 0) or 0

        top_fig = None
        tf_raw = horse_dict.get("top_fig", "")
        if tf_raw:
            try:
                top_fig = float(str(tf_raw))
            except (ValueError, TypeError):
                pass

        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO sheets_horses(
                upload_id, horse_name, normalized_name, race_number, post,
                sex, age, top_fig, track, race_date, sire, dam, trainer
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                upload_id, name, norm, race_num,
                str(horse_dict.get("post", "")),
                horse_dict.get("sex", ""),
                horse_dict.get("age", 0) or 0,
                top_fig,
                track_code,
                race_date,
                horse_dict.get("sire", ""),
                horse_dict.get("dam", ""),
                horse_dict.get("trainer", ""),
            ),
        )

        if cur.rowcount == 0:
            return None  # duplicate

        horse_id = cur.lastrowid

        # Insert lines
        for idx, line in enumerate(horse_dict.get("lines", [])):
            parsed_fig = line.get("parsed_figure")
            if parsed_fig is not None:
                try:
                    parsed_fig = float(parsed_fig)
                except (ValueError, TypeError):
                    parsed_fig = None
            flags = json.dumps(line.get("flags", []))

            self.conn.execute(
                """
                INSERT OR IGNORE INTO sheets_lines(
                    horse_id, line_index, parsed_figure, raw_text, surface,
                    track, flags, data_text, race_date, race_type
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    horse_id, idx, parsed_fig,
                    line.get("raw_text", ""),
                    line.get("surface", "DIRT"),
                    line.get("track", ""),
                    flags,
                    line.get("data_text", ""),
                    line.get("race_date", ""),
                    line.get("race_type", ""),
                ),
            )

        self.conn.commit()
        return horse_id

    def ingest_sheets_card(
        self, upload_id: str, parsed_json: Dict[str, Any]
    ) -> Dict[str, int]:
        """Ingest an entire parsed Ragozin card JSON."""
        track = parsed_json.get("track_name", "")
        race_date = parsed_json.get("race_date", "")
        inserted = 0
        skipped = 0

        for h in parsed_json.get("horses", []):
            name = h.get("horse_name", "")
            if not name or name in ("No Data Found", "Unknown"):
                continue
            hid = self.ingest_sheets_horse(upload_id, h, track, race_date)
            if hid is not None:
                inserted += 1
            else:
                skipped += 1

        # Update upload record with final count
        self.conn.execute(
            "UPDATE uploads SET horses_count = ? WHERE upload_id = ?",
            (inserted, upload_id),
        )
        self.conn.commit()
        return {"inserted": inserted, "skipped": skipped}

    # ==================================================================
    # BRISNET ingestion
    # ==================================================================

    def ingest_brisnet_horse(
        self,
        upload_id: str,
        horse_dict: Dict[str, Any],
        race_dict: Dict[str, Any],
    ) -> Optional[int]:
        """Insert a BRISNET horse + running lines + workouts. Returns brisnet_id or None."""
        name = horse_dict.get("horse_name", "") or horse_dict.get("name", "")
        norm = self._normalize_name(name)
        track_code = self._normalize_track(race_dict.get("track", ""))
        race_date = self._normalize_date(
            race_dict.get("race_date", "") or race_dict.get("date", ""))
        race_num = horse_dict.get("race_number", 0) or race_dict.get("race_number", 0) or 0
        restrictions = race_dict.get("restrictions", "")
        age = self._age_from_restrictions(restrictions)

        prime_power = None
        pp_raw = horse_dict.get("prime_power")
        if pp_raw is not None:
            try:
                prime_power = float(pp_raw)
            except (ValueError, TypeError):
                pass

        surface_records = horse_dict.get("surface_records")
        if isinstance(surface_records, dict):
            surface_records = json.dumps(surface_records)
        elif surface_records is None:
            surface_records = ""

        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO brisnet_horses(
                upload_id, horse_name, normalized_name, post, race_number,
                runstyle, runstyle_rating, prime_power, sire, dam, breeder,
                weight, life_starts, life_record, life_earnings, life_speed,
                surface_records, track, race_date, restrictions, age
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                upload_id, name, norm,
                horse_dict.get("post", 0) or 0,
                race_num,
                horse_dict.get("runstyle", ""),
                horse_dict.get("runstyle_rating", 0) or 0,
                prime_power,
                horse_dict.get("sire", "") or "",
                horse_dict.get("dam", "") or "",
                horse_dict.get("breeder", "") or "",
                horse_dict.get("weight", 0) or 0,
                horse_dict.get("life_starts", 0) or 0,
                horse_dict.get("life_record", "") or "",
                horse_dict.get("life_earnings", "") or "",
                horse_dict.get("life_speed", 0) or 0,
                surface_records,
                track_code,
                race_date,
                restrictions,
                age,
            ),
        )

        if cur.rowcount == 0:
            return None  # duplicate

        brisnet_id = cur.lastrowid

        # Insert running lines
        for idx, line in enumerate(horse_dict.get("lines", []) or horse_dict.get("running_lines", [])):
            speed = None
            sp_raw = line.get("parsed_figure") or line.get("speed")
            if sp_raw is not None:
                try:
                    speed = int(sp_raw)
                except (ValueError, TypeError):
                    pass

            self.conn.execute(
                """
                INSERT OR IGNORE INTO brisnet_lines(
                    brisnet_id, line_index, race_date, track, surface,
                    speed, race_type, raw_text
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    brisnet_id, idx,
                    line.get("race_date", ""),
                    line.get("track", ""),
                    line.get("surface", ""),
                    speed,
                    line.get("race_type", ""),
                    line.get("raw_text", ""),
                ),
            )

        # Insert workouts
        for wo in horse_dict.get("workouts", []) or []:
            self.conn.execute(
                """
                INSERT INTO brisnet_workouts(
                    brisnet_id, workout_date, track, distance,
                    surface, time, rank_letter, rank
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    brisnet_id,
                    wo.get("date", "") or wo.get("workout_date", ""),
                    wo.get("track", ""),
                    wo.get("distance", ""),
                    wo.get("surface", ""),
                    wo.get("time", ""),
                    wo.get("rank_letter", ""),
                    wo.get("rank", ""),
                ),
            )

        self.conn.commit()
        return brisnet_id

    def ingest_brisnet_card(
        self,
        upload_id: str,
        pipeline_json: Dict[str, Any],
        races_json: Dict[str, Any],
    ) -> Dict[str, int]:
        """Ingest an entire parsed BRISNET card."""
        # Build race info lookup from races_json
        race_info: Dict[int, Dict] = {}
        track = pipeline_json.get("track_name", "") or pipeline_json.get("track", "")
        race_date = pipeline_json.get("race_date", "") or pipeline_json.get("date", "")

        for race in races_json.get("races", []):
            rn = race.get("race_number", 0)
            conds = race.get("conditions", {})
            if isinstance(conds, str):
                conds = {}
            race_info[rn] = {
                "track": track,
                "race_date": race_date,
                "race_number": rn,
                "restrictions": conds.get("restrictions", ""),
            }

        inserted = 0
        skipped = 0

        for h in pipeline_json.get("horses", []):
            name = h.get("horse_name", "") or h.get("name", "")
            if not name or name in ("No Data Found", "Unknown"):
                continue
            rn = h.get("race_number", 0) or 0
            r_info = race_info.get(rn, {
                "track": track, "race_date": race_date,
                "race_number": rn, "restrictions": "",
            })
            bid = self.ingest_brisnet_horse(upload_id, h, r_info)
            if bid is not None:
                inserted += 1
            else:
                skipped += 1

        self.conn.execute(
            "UPDATE uploads SET horses_count = ? WHERE upload_id = ?",
            (inserted, upload_id),
        )
        self.conn.commit()
        return {"inserted": inserted, "skipped": skipped}

    # ==================================================================
    # Reconciliation
    # ==================================================================

    def reconcile(self, upload_id: str = "") -> Dict[str, int]:
        """Match sheets horses <-> brisnet horses across 4 tiers.

        HIGH:
          1) (track, race_date, race_number, post) exact
          2) (track, race_date, race_number, normalized_name) when post missing
        MED:
          3) (normalized_name, track, race_date)
        LOW:
          4) normalized_name only — but only if unique on both sides
        Aliases are resolved before name matching in tiers 2-4.
        """
        now = datetime.utcnow().isoformat()
        new_matches = 0

        # --- Build alias lookup (alias -> canonical) ---
        alias_rows = self.conn.execute(
            "SELECT canonical_name, alias_name FROM horse_aliases"
        ).fetchall()
        alias_map: Dict[str, str] = {r[1]: r[0] for r in alias_rows}

        # Apply aliases: update normalized_name in-memory for matching.
        # We use a temp table approach: create a view-like mapping.
        # Simpler: just do the alias resolution in SQL via COALESCE + LEFT JOIN.

        # Helper: already-reconciled sheets horse_ids
        def _reconciled_sheets_ids() -> set:
            return {r[0] for r in self.conn.execute(
                "SELECT DISTINCT horse_id FROM reconciliation"
            ).fetchall()}

        # --- Pass 1: HIGH — track + date + race + post (exact) ---
        rows = self.conn.execute(
            """
            SELECT sh.horse_id, bh.brisnet_id
            FROM sheets_horses sh
            JOIN brisnet_horses bh
                ON sh.track = bh.track
                AND sh.race_date = bh.race_date
                AND sh.race_number = bh.race_number
                AND CAST(sh.post AS TEXT) = CAST(bh.post AS TEXT)
            WHERE sh.track != '' AND sh.race_date != ''
                AND sh.post != '' AND sh.post != '0'
                AND NOT EXISTS (
                    SELECT 1 FROM reconciliation r
                    WHERE r.horse_id = sh.horse_id
                )
            """
        ).fetchall()

        for row in rows:
            self.conn.execute(
                """INSERT OR IGNORE INTO reconciliation
                   (horse_id, brisnet_id, match_method, confidence, created_at)
                   VALUES (?, ?, 'track_date_race_post', 'high', ?)""",
                (row["horse_id"], row["brisnet_id"], now),
            )
            new_matches += 1

        # --- Pass 2: HIGH — track + date + race + name (post missing) ---
        rows2 = self.conn.execute(
            """
            SELECT sh.horse_id, sh.normalized_name, bh.brisnet_id, bh.normalized_name AS bn
            FROM sheets_horses sh
            JOIN brisnet_horses bh
                ON sh.track = bh.track
                AND sh.race_date = bh.race_date
                AND sh.race_number = bh.race_number
            WHERE sh.track != '' AND sh.race_date != ''
                AND NOT EXISTS (
                    SELECT 1 FROM reconciliation r
                    WHERE r.horse_id = sh.horse_id
                )
            """
        ).fetchall()

        for row in rows2:
            sh_name = alias_map.get(row["normalized_name"], row["normalized_name"])
            bh_name = alias_map.get(row["bn"], row["bn"])
            if sh_name == bh_name:
                self.conn.execute(
                    """INSERT OR IGNORE INTO reconciliation
                       (horse_id, brisnet_id, match_method, confidence, created_at)
                       VALUES (?, ?, 'track_date_race_name', 'high', ?)""",
                    (row["horse_id"], row["brisnet_id"], now),
                )
                new_matches += 1

        # --- Pass 2.5: HIGH — track + date + race + fuzzy-stripped name ---
        for row in rows2:
            if row["horse_id"] in _reconciled_sheets_ids():
                continue
            sh_fuzzy = self._fuzzy_strip(row["normalized_name"])
            bh_fuzzy = self._fuzzy_strip(row["bn"])
            sh_resolved = alias_map.get(sh_fuzzy, sh_fuzzy)
            bh_resolved = alias_map.get(bh_fuzzy, bh_fuzzy)
            if sh_resolved == bh_resolved:
                self.conn.execute(
                    """INSERT OR IGNORE INTO reconciliation
                       (horse_id, brisnet_id, match_method, confidence, created_at)
                       VALUES (?, ?, 'fuzzy_strip_race', 'high', ?)""",
                    (row["horse_id"], row["brisnet_id"], now),
                )
                new_matches += 1

        # --- Pass 3: MED — normalized_name + track + date ---
        rows3 = self.conn.execute(
            """
            SELECT sh.horse_id, sh.normalized_name, bh.brisnet_id, bh.normalized_name AS bn
            FROM sheets_horses sh
            JOIN brisnet_horses bh
                ON sh.track = bh.track
                AND sh.race_date = bh.race_date
            WHERE sh.track != '' AND sh.race_date != ''
                AND NOT EXISTS (
                    SELECT 1 FROM reconciliation r
                    WHERE r.horse_id = sh.horse_id
                )
            """
        ).fetchall()

        for row in rows3:
            sh_name = alias_map.get(row["normalized_name"], row["normalized_name"])
            bh_name = alias_map.get(row["bn"], row["bn"])
            if sh_name == bh_name:
                self.conn.execute(
                    """INSERT OR IGNORE INTO reconciliation
                       (horse_id, brisnet_id, match_method, confidence, created_at)
                       VALUES (?, ?, 'name_track_date', 'medium', ?)""",
                    (row["horse_id"], row["brisnet_id"], now),
                )
                new_matches += 1

        # --- Pass 3.5: MED — fuzzy-stripped name + track + date ---
        for row in rows3:
            if row["horse_id"] in _reconciled_sheets_ids():
                continue
            sh_fuzzy = self._fuzzy_strip(row["normalized_name"])
            bh_fuzzy = self._fuzzy_strip(row["bn"])
            sh_resolved = alias_map.get(sh_fuzzy, sh_fuzzy)
            bh_resolved = alias_map.get(bh_fuzzy, bh_fuzzy)
            if sh_resolved == bh_resolved:
                self.conn.execute(
                    """INSERT OR IGNORE INTO reconciliation
                       (horse_id, brisnet_id, match_method, confidence, created_at)
                       VALUES (?, ?, 'fuzzy_strip_name', 'medium', ?)""",
                    (row["horse_id"], row["brisnet_id"], now),
                )
                new_matches += 1

        # --- Pass 4: LOW — normalized_name only, unique on both sides ---
        reconciled_sh = _reconciled_sheets_ids()
        reconciled_bh = {r[0] for r in self.conn.execute(
            "SELECT DISTINCT brisnet_id FROM reconciliation"
        ).fetchall()}

        # Sheets horses not yet reconciled, grouped by resolved name
        unmatched_sh = self.conn.execute(
            "SELECT horse_id, normalized_name FROM sheets_horses"
        ).fetchall()
        sh_by_name: Dict[str, List[int]] = {}
        for r in unmatched_sh:
            if r["horse_id"] in reconciled_sh:
                continue
            resolved = alias_map.get(r["normalized_name"], r["normalized_name"])
            sh_by_name.setdefault(resolved, []).append(r["horse_id"])

        # BRISNET horses not yet reconciled, grouped by resolved name
        unmatched_bh = self.conn.execute(
            "SELECT brisnet_id, normalized_name FROM brisnet_horses"
        ).fetchall()
        bh_by_name: Dict[str, List[int]] = {}
        for r in unmatched_bh:
            if r["brisnet_id"] in reconciled_bh:
                continue
            resolved = alias_map.get(r["normalized_name"], r["normalized_name"])
            bh_by_name.setdefault(resolved, []).append(r["brisnet_id"])

        # Only match when BOTH sides have exactly 1 horse for that name
        for name, sh_ids in sh_by_name.items():
            if len(sh_ids) == 1 and name in bh_by_name and len(bh_by_name[name]) == 1:
                self.conn.execute(
                    """INSERT OR IGNORE INTO reconciliation
                       (horse_id, brisnet_id, match_method, confidence, created_at)
                       VALUES (?, ?, 'name_only_unique', 'low', ?)""",
                    (sh_ids[0], bh_by_name[name][0], now),
                )
                new_matches += 1

        self.conn.commit()

        existing = self.conn.execute(
            "SELECT COUNT(*) FROM reconciliation"
        ).fetchone()[0]

        return {"new_matches": new_matches, "existing": existing}

    # ==================================================================
    # Query methods for studies
    # ==================================================================

    def load_all_sheets_horses(
        self,
        track: Optional[str] = None,
        min_lines: int = 0,
    ) -> List[Dict[str, Any]]:
        """Load all Ragozin horses with lines, in JSON-compatible shape."""
        where = "WHERE 1=1"
        params: List[Any] = []
        if track:
            where += " AND sh.track = ?"
            params.append(self._normalize_track(track))

        horses_rows = self.conn.execute(
            f"""
            SELECT sh.horse_id, sh.horse_name, sh.normalized_name,
                   sh.race_number, sh.age, sh.track, sh.race_date,
                   sh.sex, sh.top_fig, sh.sire, sh.dam, sh.post
            FROM sheets_horses sh
            {where}
            ORDER BY sh.horse_id
            """,
            params,
        ).fetchall()

        result = []
        for hr in horses_rows:
            lines = self.conn.execute(
                """
                SELECT line_index, parsed_figure, surface, track, flags,
                       data_text, race_date, race_type, raw_text
                FROM sheets_lines
                WHERE horse_id = ?
                ORDER BY line_index
                """,
                (hr["horse_id"],),
            ).fetchall()

            if len(lines) < min_lines:
                continue

            horse_dict: Dict[str, Any] = {
                "horse_name": hr["horse_name"],
                "race_number": hr["race_number"],
                "age": hr["age"] or 0,
                "lines": [
                    {
                        "parsed_figure": l["parsed_figure"],
                        "fig": str(l["parsed_figure"] or ""),
                        "surface": l["surface"] or "DIRT",
                        "track": l["track"] or "",
                        "flags": json.loads(l["flags"]) if l["flags"] else [],
                        "data_text": l["data_text"] or "",
                        "race_date": l["race_date"] or "",
                        "race_type": l["race_type"] or "",
                        "raw_text": l["raw_text"] or "",
                    }
                    for l in lines
                ],
            }
            result.append(horse_dict)

        return result

    def load_enriched_horses(
        self, track: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict]]:
        """Load Ragozin horses with matched BRISNET enrichment.

        Returns (all_horses, enrichment_dict) compatible with
        poly_dirt_study.merge_enrichment().
        """
        all_horses = self.load_all_sheets_horses(track=track)
        enrichment: Dict[str, Dict] = {}

        # Fetch all reconciliation + brisnet data in one pass
        recon_rows = self.conn.execute(
            """
            SELECT r.horse_id, r.confidence, bh.*
            FROM reconciliation r
            JOIN brisnet_horses bh ON r.brisnet_id = bh.brisnet_id
            """
        ).fetchall()

        # Build lookup: horse_id -> brisnet enrichment
        recon_by_horse: Dict[int, Dict] = {}
        for row in recon_rows:
            recon_by_horse[row["horse_id"]] = dict(row)

        # Map horse_name -> horse_id (need this for lookup)
        name_to_id: Dict[str, int] = {}
        for row in self.conn.execute(
            "SELECT horse_id, normalized_name FROM sheets_horses"
        ).fetchall():
            name_to_id[row["normalized_name"]] = row["horse_id"]

        for horse in all_horses:
            norm = self._normalize_name(horse["horse_name"])
            horse_id = name_to_id.get(norm)
            if horse_id and horse_id in recon_by_horse:
                recon = recon_by_horse[horse_id]
                brisnet_id = recon["brisnet_id"]

                # Fetch BRISNET running line dates/surfaces
                blines = self.conn.execute(
                    """
                    SELECT race_date, surface
                    FROM brisnet_lines
                    WHERE brisnet_id = ?
                    ORDER BY line_index
                    """,
                    (brisnet_id,),
                ).fetchall()

                rl_dates: List[Any] = []
                rl_surfaces: List[str] = []
                for bl in blines:
                    rl_dates.append(bl["race_date"])  # raw string, caller parses
                    rl_surfaces.append(bl["surface"] or "DIRT")

                enr = {
                    "runstyle": recon.get("runstyle") or "",
                    "sire": recon.get("sire") or "",
                    "dam": recon.get("dam") or "",
                    "age": recon.get("age") or 0,
                    "race_number": recon.get("race_number") or 0,
                    "post": recon.get("post") or 0,
                    "running_line_dates": rl_dates,
                    "running_line_surfaces": rl_surfaces,
                    "match_confidence": recon.get("confidence", "medium"),
                }
                horse["_enrichment"] = enr
                horse["_match_confidence"] = enr["match_confidence"]
                enrichment[norm] = enr

        return all_horses, enrichment

    # ==================================================================
    # Statistics
    # ==================================================================

    def get_session_stats(self, upload_id: str) -> Dict[str, Any]:
        """Stats scoped to a single upload/session and its reconciliation impact."""
        # Also accept session_id that maps to multiple upload_ids (primary + _sec)
        upload_ids = [r[0] for r in self.conn.execute(
            "SELECT upload_id FROM uploads WHERE upload_id = ? OR session_id = ?",
            (upload_id, upload_id),
        ).fetchall()]
        if not upload_ids:
            return {"error": "upload not found"}

        placeholders = ",".join("?" * len(upload_ids))

        sh = self.conn.execute(
            f"SELECT COUNT(*) FROM sheets_horses WHERE upload_id IN ({placeholders})",
            upload_ids,
        ).fetchone()[0]

        bh = self.conn.execute(
            f"SELECT COUNT(*) FROM brisnet_horses WHERE upload_id IN ({placeholders})",
            upload_ids,
        ).fetchone()[0]

        # Reconciled pairs involving horses from these uploads
        rp_rows = self.conn.execute(
            f"""
            SELECT r.confidence, COUNT(*) as cnt
            FROM reconciliation r
            WHERE r.horse_id IN (
                SELECT horse_id FROM sheets_horses WHERE upload_id IN ({placeholders})
            )
            OR r.brisnet_id IN (
                SELECT brisnet_id FROM brisnet_horses WHERE upload_id IN ({placeholders})
            )
            GROUP BY r.confidence
            """,
            upload_ids + upload_ids,
        ).fetchall()
        confidence_breakdown = {r["confidence"]: r["cnt"] for r in rp_rows}
        rp = sum(confidence_breakdown.values())

        total_session_horses = sh + bh
        coverage_pct = (rp / total_session_horses * 100) if total_session_horses > 0 else 0.0

        # Unmatched from this session (top 20 each)
        unmatched_sh = self.conn.execute(
            f"""
            SELECT sh.normalized_name, sh.track, sh.race_date
            FROM sheets_horses sh
            WHERE sh.upload_id IN ({placeholders})
            AND NOT EXISTS (SELECT 1 FROM reconciliation r WHERE r.horse_id = sh.horse_id)
            ORDER BY sh.normalized_name LIMIT 20
            """,
            upload_ids,
        ).fetchall()

        unmatched_bh = self.conn.execute(
            f"""
            SELECT bh.normalized_name, bh.track, bh.race_date
            FROM brisnet_horses bh
            WHERE bh.upload_id IN ({placeholders})
            AND NOT EXISTS (SELECT 1 FROM reconciliation r WHERE r.brisnet_id = bh.brisnet_id)
            ORDER BY bh.normalized_name LIMIT 20
            """,
            upload_ids,
        ).fetchall()

        return {
            "upload_ids": upload_ids,
            "sheets_horses": sh,
            "brisnet_horses": bh,
            "reconciled_pairs": rp,
            "coverage_pct": round(coverage_pct, 1),
            "confidence_breakdown": confidence_breakdown,
            "unmatched_sheets": [
                {"name": r["normalized_name"], "track": r["track"], "date": r["race_date"]}
                for r in unmatched_sh
            ],
            "unmatched_brisnet": [
                {"name": r["normalized_name"], "track": r["track"], "date": r["race_date"]}
                for r in unmatched_bh
            ],
        }

    def get_db_stats(self) -> Dict[str, Any]:
        """Aggregate statistics for the Streamlit panel."""
        sh = self.conn.execute("SELECT COUNT(*) FROM sheets_horses").fetchone()[0]
        sl = self.conn.execute("SELECT COUNT(*) FROM sheets_lines").fetchone()[0]
        bh = self.conn.execute("SELECT COUNT(*) FROM brisnet_horses").fetchone()[0]
        bl = self.conn.execute("SELECT COUNT(*) FROM brisnet_lines").fetchone()[0]
        rp = self.conn.execute("SELECT COUNT(*) FROM reconciliation").fetchone()[0]
        up = self.conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]

        tracks_rows = self.conn.execute(
            """
            SELECT DISTINCT track FROM (
                SELECT track FROM sheets_horses WHERE track != ''
                UNION
                SELECT track FROM brisnet_horses WHERE track != ''
            )
            """
        ).fetchall()
        tracks = sorted(row[0] for row in tracks_rows)

        coverage_pct = 0.0
        if sh > 0:
            coverage_pct = (rp / sh) * 100

        # Confidence breakdown
        conf_rows = self.conn.execute(
            "SELECT confidence, COUNT(*) as cnt FROM reconciliation GROUP BY confidence"
        ).fetchall()
        confidence_breakdown = {r["confidence"]: r["cnt"] for r in conf_rows}

        # Unmatched sheets names (top 20)
        unmatched_sheets = self.conn.execute(
            """
            SELECT sh.normalized_name, sh.track, sh.race_date
            FROM sheets_horses sh
            WHERE NOT EXISTS (
                SELECT 1 FROM reconciliation r WHERE r.horse_id = sh.horse_id
            )
            ORDER BY sh.normalized_name
            LIMIT 20
            """
        ).fetchall()
        unmatched_sheets_list = [
            {"name": r["normalized_name"], "track": r["track"], "date": r["race_date"]}
            for r in unmatched_sheets
        ]

        # Unmatched brisnet names (top 20)
        unmatched_brisnet = self.conn.execute(
            """
            SELECT bh.normalized_name, bh.track, bh.race_date
            FROM brisnet_horses bh
            WHERE NOT EXISTS (
                SELECT 1 FROM reconciliation r WHERE r.brisnet_id = bh.brisnet_id
            )
            ORDER BY bh.normalized_name
            LIMIT 20
            """
        ).fetchall()
        unmatched_brisnet_list = [
            {"name": r["normalized_name"], "track": r["track"], "date": r["race_date"]}
            for r in unmatched_brisnet
        ]

        # Collision warnings: same normalized_name mapping to multiple reconciliation rows
        collisions = self.conn.execute(
            """
            SELECT sh.normalized_name, COUNT(DISTINCT r.brisnet_id) as cnt
            FROM reconciliation r
            JOIN sheets_horses sh ON r.horse_id = sh.horse_id
            GROUP BY sh.normalized_name
            HAVING cnt > 1
            """
        ).fetchall()
        collision_warnings = [r["normalized_name"] for r in collisions]

        return {
            "sheets_horses": sh,
            "sheets_lines": sl,
            "brisnet_horses": bh,
            "brisnet_lines": bl,
            "reconciled_pairs": rp,
            "uploads_count": up,
            "tracks": tracks,
            "coverage_pct": round(coverage_pct, 1),
            "confidence_breakdown": confidence_breakdown,
            "unmatched_sheets": unmatched_sheets_list,
            "unmatched_brisnet": unmatched_brisnet_list,
            "collision_warnings": collision_warnings,
        }

    # ==================================================================
    # Race results
    # ==================================================================

    def insert_race_result(
        self, track: str, race_date: str, race_number: int,
        surface: str = "", distance: str = "",
        winner_post: int = 0, winner_name: str = "",
    ) -> Optional[int]:
        """Insert or update a race result row. Returns result_id."""
        track = self._normalize_track(track)
        race_date = self._normalize_date(race_date)
        now = datetime.utcnow().isoformat()
        cur = self.conn.execute(
            """
            INSERT INTO result_races(track, race_date, race_number, surface,
                                     distance, winner_post, winner_name, imported_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track, race_date, race_number) DO UPDATE SET
                surface=excluded.surface, distance=excluded.distance,
                winner_post=excluded.winner_post, winner_name=excluded.winner_name,
                imported_at=excluded.imported_at
            """,
            (track, race_date, race_number, surface, distance,
             winner_post, winner_name, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def mark_race_parse_error(
        self, track: str, race_date: str, race_number: int,
    ) -> None:
        """Flag a race as having a parse error (e.g. no winner)."""
        track = self._normalize_track(track)
        race_date = self._normalize_date(race_date)
        self.conn.execute(
            "UPDATE result_races SET status = 'PARSE_ERROR' WHERE track = ? AND race_date = ? AND race_number = ?",
            (track, race_date, race_number),
        )
        self.conn.commit()

    def insert_entry_result(
        self, track: str, race_date: str, race_number: int, post: int,
        horse_name: str = "", finish_pos: int = 0,
        beaten_lengths: float = None, odds: float = None,
        win_payoff: float = None, place_payoff: float = None,
        show_payoff: float = None, session_id: str = "",
    ) -> Optional[int]:
        """Insert or update an entry result row. Returns entry_result_id."""
        track = self._normalize_track(track)
        race_date = self._normalize_date(race_date)
        norm = self._normalize_name(horse_name)
        cur = self.conn.execute(
            """
            INSERT INTO result_entries(track, race_date, race_number, post,
                horse_name, normalized_name, finish_pos, beaten_lengths,
                odds, win_payoff, place_payoff, show_payoff, session_id)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track, race_date, race_number, post) DO UPDATE SET
                horse_name=excluded.horse_name,
                normalized_name=excluded.normalized_name,
                finish_pos=excluded.finish_pos,
                beaten_lengths=excluded.beaten_lengths,
                odds=excluded.odds,
                win_payoff=excluded.win_payoff,
                place_payoff=excluded.place_payoff,
                show_payoff=excluded.show_payoff,
                session_id=COALESCE(excluded.session_id, result_entries.session_id)
            """,
            (track, race_date, race_number, post, horse_name, norm,
             finish_pos, beaten_lengths, odds,
             win_payoff, place_payoff, show_payoff, session_id or None),
        )
        self.conn.commit()
        return cur.lastrowid

    def link_results_to_entries(
        self, track: str = "", race_date: str = "",
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Link result_entries to brisnet/sheets horses.

        Strategy (brisnet-first since they have post positions):
          Pass 1: exact (track, date, race, post) → brisnet_horses
          Pass 2: reconciliation → sheets_horses from brisnet link
          Pass 3: name fallback (with aliases) → sheets_horses for unlinked
          Pass 4: reconciliation → brisnet from sheets link (reverse)

        When *session_id* is provided, scope matching to uploads from that
        session (upload_id = session_id or session_id + '_sec').
        """
        track = self._normalize_track(track) if track else ""
        race_date = self._normalize_date(race_date) if race_date else ""
        linked_brisnet = 0
        linked_sheets = 0
        unmatched: List[Dict[str, str]] = []

        # Build WHERE clauses for result_entries scope
        # er_where: aliased as "er" (for SELECT/JOIN)
        # up_where: unaliased (for UPDATE ... WHERE on result_entries)
        er_parts = ["1=1"]
        up_parts = ["1=1"]
        re_params: List[Any] = []
        if track:
            er_parts.append("er.track = ?")
            up_parts.append("track = ?")
            re_params.append(track)
        if race_date:
            er_parts.append("er.race_date = ?")
            up_parts.append("race_date = ?")
            re_params.append(race_date)
        if session_id:
            er_parts.append("er.session_id = ?")
            up_parts.append("session_id = ?")
            re_params.append(session_id)
        re_where = " AND ".join(er_parts)
        up_where = " AND ".join(up_parts)

        # Optional: scope brisnet source to session uploads
        bh_session_filter = ""
        bh_session_params: List[Any] = []
        if session_id:
            bh_session_filter = "AND bh.upload_id IN (?, ?)"
            bh_session_params = [session_id, f"{session_id}_sec"]

        # ---- Pass 1: exact post match → brisnet_horses ----
        sql1 = f"""
            SELECT er.entry_result_id, bh.brisnet_id
            FROM result_entries er
            JOIN brisnet_horses bh
                ON er.track = bh.track
                AND er.race_date = bh.race_date
                AND er.race_number = bh.race_number
                AND er.post = bh.post
                {bh_session_filter}
            WHERE er.brisnet_horse_id IS NULL AND {re_where}
        """
        rows = self.conn.execute(sql1, bh_session_params + re_params).fetchall()
        for r in rows:
            self.conn.execute(
                "UPDATE result_entries SET brisnet_horse_id = ? WHERE entry_result_id = ?",
                (r["brisnet_id"], r["entry_result_id"]),
            )
            linked_brisnet += 1

        # ---- Pass 2: reconciliation → sheets from brisnet ----
        self.conn.execute(f"""
            UPDATE result_entries SET sheets_horse_id = (
                SELECT rc.horse_id FROM reconciliation rc
                WHERE rc.brisnet_id = result_entries.brisnet_horse_id
                LIMIT 1
            )
            WHERE brisnet_horse_id IS NOT NULL
              AND sheets_horse_id IS NULL
              AND {up_where}
        """, re_params)
        linked_sheets += self.conn.execute(
            "SELECT changes()").fetchone()[0]

        # ---- Pass 3: name fallback (with aliases) for still-unlinked ----
        unlinked = self.conn.execute(f"""
            SELECT er.entry_result_id, er.normalized_name, er.track, er.race_date,
                   er.race_number, er.post
            FROM result_entries er
            WHERE er.sheets_horse_id IS NULL AND er.brisnet_horse_id IS NULL
              AND {re_where}
        """, re_params).fetchall()

        for er in unlinked:
            norm = er["normalized_name"]
            resolved = self.resolve_alias(norm)
            # Try sheets by name + track + date
            sh = self.conn.execute("""
                SELECT horse_id FROM sheets_horses
                WHERE normalized_name = ? AND track = ? AND race_date = ?
                LIMIT 1
            """, (resolved, er["track"], er["race_date"])).fetchone()
            if sh:
                self.conn.execute(
                    "UPDATE result_entries SET sheets_horse_id = ? WHERE entry_result_id = ?",
                    (sh["horse_id"], er["entry_result_id"]),
                )
                linked_sheets += 1
            else:
                # Try brisnet by name + track + date
                bh = self.conn.execute("""
                    SELECT brisnet_id FROM brisnet_horses
                    WHERE normalized_name = ? AND track = ? AND race_date = ?
                    LIMIT 1
                """, (resolved, er["track"], er["race_date"])).fetchone()
                if bh:
                    self.conn.execute(
                        "UPDATE result_entries SET brisnet_horse_id = ? WHERE entry_result_id = ?",
                        (bh["brisnet_id"], er["entry_result_id"]),
                    )
                    linked_brisnet += 1
                else:
                    reason = "no matching horse in DB"
                    unmatched.append({
                        "race": er["race_number"], "post": er["post"],
                        "name": norm, "reason": reason,
                    })

        # ---- Pass 4: fill in missing cross-links via reconciliation ----
        # brisnet → sheets
        self.conn.execute(f"""
            UPDATE result_entries SET sheets_horse_id = (
                SELECT rc.horse_id FROM reconciliation rc
                WHERE rc.brisnet_id = result_entries.brisnet_horse_id
                LIMIT 1
            )
            WHERE brisnet_horse_id IS NOT NULL
              AND sheets_horse_id IS NULL
              AND {up_where}
        """, re_params)
        # sheets → brisnet
        self.conn.execute(f"""
            UPDATE result_entries SET brisnet_horse_id = (
                SELECT rc.brisnet_id FROM reconciliation rc
                WHERE rc.horse_id = result_entries.sheets_horse_id
                LIMIT 1
            )
            WHERE sheets_horse_id IS NOT NULL
              AND brisnet_horse_id IS NULL
              AND {up_where}
        """, re_params)

        self.conn.commit()

        total_linked = linked_brisnet + linked_sheets
        total_entries = self.conn.execute(
            f"SELECT COUNT(*) FROM result_entries er WHERE {re_where}", re_params
        ).fetchone()[0]
        any_linked = self.conn.execute(
            f"""SELECT COUNT(*) FROM result_entries er
                WHERE (sheets_horse_id IS NOT NULL OR brisnet_horse_id IS NOT NULL)
                  AND {re_where}""", re_params
        ).fetchone()[0]
        link_rate = round(any_linked / total_entries * 100, 1) if total_entries else 0.0

        return {
            "linked": total_linked,
            "linked_brisnet": linked_brisnet,
            "linked_sheets": linked_sheets,
            "total_entries": total_entries,
            "any_linked": any_linked,
            "link_rate": link_rate,
            "unmatched": unmatched[:20],
        }

    def get_results_stats(
        self, track: str = "", date_from: str = "", date_to: str = "",
    ) -> Dict[str, Any]:
        """ROI and results statistics, optionally filtered."""
        where_parts = ["1=1"]
        params: List[Any] = []
        if track:
            where_parts.append("er.track = ?")
            params.append(self._normalize_track(track))
        if date_from:
            where_parts.append("er.race_date >= ?")
            params.append(self._normalize_date(date_from))
        if date_to:
            where_parts.append("er.race_date <= ?")
            params.append(self._normalize_date(date_to))
        where = " AND ".join(where_parts)

        # Total races with results
        total_races = self.conn.execute(
            f"SELECT COUNT(DISTINCT track || race_date || race_number) FROM result_entries er WHERE {where}",
            params,
        ).fetchone()[0]

        # Total entries with odds
        total_with_odds = self.conn.execute(
            f"SELECT COUNT(*) FROM result_entries er WHERE odds IS NOT NULL AND {where}",
            params,
        ).fetchone()[0]

        total_entries = self.conn.execute(
            f"SELECT COUNT(*) FROM result_entries er WHERE {where}",
            params,
        ).fetchone()[0]

        linked_entries = self.conn.execute(
            f"SELECT COUNT(*) FROM result_entries er WHERE sheets_horse_id IS NOT NULL AND {where}",
            params,
        ).fetchone()[0]

        linking_rate = (linked_entries / total_entries * 100) if total_entries > 0 else 0.0

        # ROI by pick rank (1 = top pick)
        roi_by_rank = self._calc_roi_by_rank(where, params)

        # ROI by projection_type (cycle label)
        roi_by_cycle = self._calc_roi_by_cycle(where, params)

        # Available tracks in results
        track_rows = self.conn.execute(
            "SELECT DISTINCT track FROM result_entries ORDER BY track"
        ).fetchall()
        available_tracks = [r["track"] for r in track_rows if r["track"]]

        return {
            "total_races": total_races,
            "total_entries": total_entries,
            "total_with_odds": total_with_odds,
            "linked_entries": linked_entries,
            "linking_rate": round(linking_rate, 1),
            "roi_by_rank": roi_by_rank,
            "roi_by_cycle": roi_by_cycle,
            "available_tracks": available_tracks,
        }

    # ------------------------------------------------------------------
    # Results Inbox helpers
    # ------------------------------------------------------------------

    def get_unattached_results(self) -> List[Dict[str, Any]]:
        """Return result_races rows where at least one entry has no session_id."""
        rows = self.conn.execute("""
            SELECT rr.track, rr.race_date, rr.race_number,
                   rr.surface, rr.distance,
                   COUNT(er.entry_result_id) AS total_entries,
                   SUM(CASE WHEN er.session_id IS NOT NULL AND er.session_id != ''
                            THEN 1 ELSE 0 END) AS attached,
                   SUM(CASE WHEN er.session_id IS NULL OR er.session_id = ''
                            THEN 1 ELSE 0 END) AS unattached
            FROM result_races rr
            LEFT JOIN result_entries er
                ON rr.track = er.track AND rr.race_date = er.race_date
                AND rr.race_number = er.race_number
            GROUP BY rr.track, rr.race_date, rr.race_number
            HAVING unattached > 0
            ORDER BY rr.race_date DESC, rr.track, rr.race_number
        """).fetchall()
        return [dict(r) for r in rows]

    def get_sessions_for_track_date(
        self, track: str, race_date: str,
    ) -> List[Dict[str, Any]]:
        """Find sessions matching a given track and date."""
        track = self._normalize_track(track)
        race_date = self._normalize_date(race_date)
        rows = self.conn.execute("""
            SELECT session_id, track_name, race_date, created_at,
                   parser_used, horses_count
            FROM sessions
            WHERE track_name = ? AND race_date = ?
            ORDER BY created_at DESC
        """, (track, race_date)).fetchall()
        return [dict(r) for r in rows]

    def attach_results_to_session(
        self, track: str, race_date: str, session_id: str,
    ) -> Dict[str, Any]:
        """Set session_id on unlinked result_entries and re-link."""
        track = self._normalize_track(track)
        race_date = self._normalize_date(race_date)
        cur = self.conn.execute("""
            UPDATE result_entries SET session_id = ?
            WHERE track = ? AND race_date = ?
              AND (session_id IS NULL OR session_id = '')
        """, (session_id, track, race_date))
        updated = cur.rowcount
        self.conn.commit()

        link_result = self.link_results_to_entries(
            track=track, race_date=race_date, session_id=session_id,
        )
        self._propagate_predictions_to_results(track, race_date)
        return {"updated_entries": updated, "link_result": link_result}

    def _calc_roi_by_rank(self, where: str, params: list) -> List[Dict]:
        """ROI for $2 win bets on each pick rank."""
        rows = self.conn.execute(
            f"""
            SELECT er.pick_rank,
                   COUNT(*) as bets,
                   SUM(CASE WHEN er.finish_pos = 1 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN er.finish_pos = 1 AND er.win_payoff IS NOT NULL
                            THEN er.win_payoff ELSE 0 END) as payoff_sum
            FROM result_entries er
            WHERE er.pick_rank IS NOT NULL AND er.pick_rank > 0 AND {where}
            GROUP BY er.pick_rank
            ORDER BY er.pick_rank
            """,
            params,
        ).fetchall()
        result = []
        for r in rows:
            bets = r["bets"]
            cost = bets * 2.0
            payoff = r["payoff_sum"] or 0
            roi = ((payoff - cost) / cost * 100) if cost > 0 else 0
            result.append({
                "rank": r["pick_rank"],
                "bets": bets,
                "wins": r["wins"],
                "win_pct": round(r["wins"] / bets * 100, 1) if bets > 0 else 0,
                "roi_pct": round(roi, 1),
            })
        return result

    def _calc_roi_by_cycle(self, where: str, params: list) -> List[Dict]:
        """ROI for $2 win bets grouped by cycle/projection type."""
        rows = self.conn.execute(
            f"""
            SELECT er.projection_type,
                   COUNT(*) as bets,
                   SUM(CASE WHEN er.finish_pos = 1 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN er.finish_pos = 1 AND er.win_payoff IS NOT NULL
                            THEN er.win_payoff ELSE 0 END) as payoff_sum
            FROM result_entries er
            WHERE er.projection_type IS NOT NULL AND er.projection_type != '' AND {where}
            GROUP BY er.projection_type
            ORDER BY er.projection_type
            """,
            params,
        ).fetchall()
        result = []
        for r in rows:
            bets = r["bets"]
            cost = bets * 2.0
            payoff = r["payoff_sum"] or 0
            roi = ((payoff - cost) / cost * 100) if cost > 0 else 0
            result.append({
                "cycle": r["projection_type"],
                "bets": bets,
                "wins": r["wins"],
                "win_pct": round(r["wins"] / bets * 100, 1) if bets > 0 else 0,
                "roi_pct": round(roi, 1),
            })
        return result

    # ==================================================================
    # Predictions
    # ==================================================================

    def save_predictions(
        self, session_id: str, track: str, race_date: str,
        race_number: int, projections: List[Dict[str, Any]],
    ) -> int:
        """Persist a batch of engine projections for one race.

        *projections* is a list of dicts (already sorted by bias_score desc),
        each with at least: name, projection_type, bias_score, raw_score,
        confidence, projected_low, projected_high, tags (list),
        new_top_setup (bool), bounce_risk (bool), tossed (bool).

        Returns the number of rows inserted/updated.
        """
        track = self._normalize_track(track)
        race_date = self._normalize_date(race_date)
        now = datetime.utcnow().isoformat()
        count = 0
        for rank, p in enumerate(projections, 1):
            norm = self._normalize_name(p["name"])
            tags_json = json.dumps(p.get("tags", []))
            raw_post = p.get("post")
            try:
                post_int = int(raw_post) if raw_post else None
            except (ValueError, TypeError):
                post_int = None
            self.conn.execute(
                """
                INSERT INTO result_predictions(
                    session_id, track, race_date, race_number,
                    horse_name, normalized_name, pick_rank, projection_type,
                    bias_score, raw_score, confidence,
                    projected_low, projected_high, tags,
                    new_top_setup, bounce_risk, tossed, post, saved_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id, track, race_date, race_number, normalized_name)
                DO UPDATE SET
                    pick_rank=excluded.pick_rank,
                    projection_type=excluded.projection_type,
                    bias_score=excluded.bias_score,
                    raw_score=excluded.raw_score,
                    confidence=excluded.confidence,
                    projected_low=excluded.projected_low,
                    projected_high=excluded.projected_high,
                    tags=excluded.tags,
                    new_top_setup=excluded.new_top_setup,
                    bounce_risk=excluded.bounce_risk,
                    tossed=excluded.tossed,
                    post=excluded.post,
                    saved_at=excluded.saved_at
                """,
                (
                    session_id, track, race_date, race_number,
                    p["name"], norm, rank, p.get("projection_type", "NEUTRAL"),
                    p.get("bias_score", 0), p.get("raw_score", 0),
                    p.get("confidence", 0),
                    p.get("projected_low", 0), p.get("projected_high", 0),
                    tags_json,
                    int(bool(p.get("new_top_setup"))),
                    int(bool(p.get("bounce_risk"))),
                    int(bool(p.get("tossed"))),
                    post_int,
                    now,
                ),
            )
            count += 1
        self.conn.commit()

        # Propagate pick_rank + projection_type into result_entries when
        # results already exist for this card.
        self._propagate_predictions_to_results(track, race_date)
        return count

    def _propagate_predictions_to_results(
        self, track: str, race_date: str,
    ) -> int:
        """Copy pick_rank/projection_type from predictions into result_entries.

        Matches by post first (reliable), falls back to normalized_name.
        """
        cur = self.conn.execute("""
            UPDATE result_entries SET
                pick_rank = (
                    SELECT rp.pick_rank FROM result_predictions rp
                    WHERE rp.track = result_entries.track
                      AND rp.race_date = result_entries.race_date
                      AND rp.race_number = result_entries.race_number
                      AND (
                          (rp.post IS NOT NULL AND rp.post = result_entries.post)
                          OR (rp.post IS NULL AND rp.normalized_name = result_entries.normalized_name)
                      )
                    LIMIT 1
                ),
                projection_type = (
                    SELECT rp.projection_type FROM result_predictions rp
                    WHERE rp.track = result_entries.track
                      AND rp.race_date = result_entries.race_date
                      AND rp.race_number = result_entries.race_number
                      AND (
                          (rp.post IS NOT NULL AND rp.post = result_entries.post)
                          OR (rp.post IS NULL AND rp.normalized_name = result_entries.normalized_name)
                      )
                    LIMIT 1
                )
            WHERE track = ? AND race_date = ?
              AND (
                  post IN (SELECT rp2.post FROM result_predictions rp2 WHERE rp2.track = ? AND rp2.race_date = ? AND rp2.post IS NOT NULL)
                  OR normalized_name IN (SELECT rp2.normalized_name FROM result_predictions rp2 WHERE rp2.track = ? AND rp2.race_date = ? AND rp2.post IS NULL)
              )
        """, (track, race_date, track, race_date, track, race_date))
        self.conn.commit()
        return cur.rowcount

    def get_predictions_vs_results(
        self, track: str = "", race_date: str = "", session_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Join predictions with results for display."""
        where_parts = ["1=1"]
        params: list = []
        if track:
            where_parts.append("rp.track = ?")
            params.append(self._normalize_track(track))
        if race_date:
            where_parts.append("rp.race_date = ?")
            params.append(self._normalize_date(race_date))
        if session_id:
            where_parts.append("rp.session_id = ?")
            params.append(session_id)
        where = " AND ".join(where_parts)

        rows = self.conn.execute(f"""
            SELECT rp.race_number, rp.horse_name, rp.pick_rank,
                   rp.projection_type, rp.bias_score, rp.confidence,
                   rp.projected_low, rp.projected_high, rp.tags,
                   rp.new_top_setup, rp.bounce_risk, rp.tossed,
                   er.finish_pos, er.odds, er.win_payoff,
                   er.place_payoff, er.show_payoff, er.post,
                   CASE WHEN rp.post IS NOT NULL AND rp.post = er.post THEN 'POST'
                        WHEN rp.post IS NULL AND er.entry_result_id IS NOT NULL THEN 'NAME'
                        ELSE NULL END AS match_method
            FROM result_predictions rp
            {self._pred_entry_join("rp", "er", "LEFT JOIN")}
            WHERE {where}
            ORDER BY rp.race_number, rp.pick_rank
        """, params).fetchall()

        return [dict(r) for r in rows]

    def get_dashboard_predictions_vs_results(
        self, track: str = "", race_date: str = "", session_id: str = "",
    ) -> Dict[str, Any]:
        """Per-race predicted top-2 vs actual top-2 for dashboard display."""
        # If no session specified, find the latest one for this track/date
        if not session_id:
            sid_where = ["1=1"]
            sid_params: list = []
            if track:
                sid_where.append("track = ?")
                sid_params.append(self._normalize_track(track))
            if race_date:
                sid_where.append("race_date = ?")
                sid_params.append(self._normalize_date(race_date))
            row = self.conn.execute(
                f"SELECT session_id FROM result_predictions WHERE {' AND '.join(sid_where)} ORDER BY saved_at DESC LIMIT 1",
                sid_params,
            ).fetchone()
            if row:
                session_id = row[0]

        where_parts = ["1=1"]
        params: list = []
        if track:
            where_parts.append("rp.track = ?")
            params.append(self._normalize_track(track))
        if race_date:
            where_parts.append("rp.race_date = ?")
            params.append(self._normalize_date(race_date))
        if session_id:
            where_parts.append("rp.session_id = ?")
            params.append(session_id)
        where = " AND ".join(where_parts)

        # Predictions joined with their result finish positions
        pred_rows = self.conn.execute(f"""
            SELECT rp.race_number, rp.horse_name, rp.pick_rank,
                   er.finish_pos, er.odds,
                   er.horse_name AS result_horse_name,
                   CASE WHEN rp.post IS NOT NULL AND rp.post = er.post THEN 'POST'
                        WHEN rp.post IS NULL AND er.entry_result_id IS NOT NULL THEN 'NAME'
                        ELSE NULL END AS match_method
            FROM result_predictions rp
            {self._pred_entry_join("rp", "er", "LEFT JOIN")}
            WHERE {where}
            ORDER BY rp.race_number, rp.pick_rank
        """, params).fetchall()

        if not pred_rows:
            return {"races": [], "summary": {}}

        # Get actual winners/runners-up from result_entries
        # Build where for result_entries using same track/date/session filters
        re_where_parts = ["1=1"]
        re_params: list = []
        if track:
            re_where_parts.append("re.track = ?")
            re_params.append(self._normalize_track(track))
        if race_date:
            re_where_parts.append("re.race_date = ?")
            re_params.append(self._normalize_date(race_date))
        if session_id:
            # Join to result_predictions to scope by session
            re_where_parts.append("""re.race_number IN (
                SELECT DISTINCT rp2.race_number FROM result_predictions rp2
                WHERE rp2.session_id = ? AND rp2.track = re.track AND rp2.race_date = re.race_date
            )""")
            re_params.append(session_id)
        re_where = " AND ".join(re_where_parts)

        result_rows = self.conn.execute(f"""
            SELECT re.race_number, re.horse_name, re.finish_pos, re.odds
            FROM result_entries re
            LEFT JOIN result_races rr
                ON re.track = rr.track AND re.race_date = rr.race_date
                AND re.race_number = rr.race_number
            WHERE {re_where} AND re.finish_pos IN (1, 2)
              AND COALESCE(rr.status, 'OK') != 'PARSE_ERROR'
            ORDER BY re.race_number, re.finish_pos
        """, re_params).fetchall()

        # Index actual finishers by race
        winners: Dict[int, Dict] = {}
        runners_up: Dict[int, Dict] = {}
        for r in result_rows:
            rd = dict(r)
            rn = rd["race_number"]
            if rd["finish_pos"] == 1 and rn not in winners:
                winners[rn] = rd
            elif rd["finish_pos"] == 2 and rn not in runners_up:
                runners_up[rn] = rd

        # Group predictions by race
        from collections import defaultdict
        preds_by_race: Dict[int, list] = defaultdict(list)
        for r in pred_rows:
            preds_by_race[dict(r)["race_number"]].append(dict(r))

        races = []
        exact_wins = 0
        top2_hits = 0
        exact_2nds = 0

        for rn in sorted(preds_by_race.keys()):
            preds = preds_by_race[rn]
            pick1 = next((p for p in preds if p["pick_rank"] == 1), None)
            pick2 = next((p for p in preds if p["pick_rank"] == 2), None)
            winner = winners.get(rn)
            runner_up = runners_up.get(rn)

            hit_1 = False
            hit_2 = False
            top2_hit = False

            if pick1 and pick1.get("finish_pos") == 1:
                hit_1 = True
                exact_wins += 1
            if pick1 and pick1.get("finish_pos") in (1, 2):
                top2_hit = True
                top2_hits += 1
            if pick2 and pick2.get("finish_pos") == 2:
                hit_2 = True
                exact_2nds += 1

            race_data = {
                "race_number": rn,
                "predicted_1": pick1["horse_name"] if pick1 else None,
                "predicted_1_finish": pick1.get("finish_pos") if pick1 else None,
                "predicted_2": pick2["horse_name"] if pick2 else None,
                "predicted_2_finish": pick2.get("finish_pos") if pick2 else None,
                "winner": winner["horse_name"] if winner else None,
                "winner_odds": winner.get("odds") if winner else None,
                "runner_up": runner_up["horse_name"] if runner_up else None,
                "hit_1": hit_1,
                "hit_2": hit_2,
                "top2_hit": top2_hit,
                "match_method": pick1.get("match_method") if pick1 else None,
                "result_horse_name": pick1.get("result_horse_name") if pick1 else None,
            }
            races.append(race_data)

        total = len(races)
        summary = {
            "total_races": total,
            "exact_wins": exact_wins,
            "exact_2nds": exact_2nds,
            "top2_hits": top2_hits,
            "win_rate": round(exact_wins / total * 100, 1) if total else 0,
            "top2_rate": round(top2_hits / total * 100, 1) if total else 0,
        }

        return {"races": races, "summary": summary}

    def get_all_predictions_for_date(
        self, race_date: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return all predictions for a date, grouped by track.

        LEFT JOINs result_entries for odds/outcome data.
        """
        race_date = self._normalize_date(race_date)
        rows = self.conn.execute("""
            SELECT rp.session_id, rp.track, rp.race_number, rp.horse_name,
                   rp.normalized_name, rp.pick_rank, rp.projection_type,
                   rp.bias_score, rp.raw_score, rp.confidence,
                   rp.projected_low, rp.projected_high, rp.tags,
                   rp.new_top_setup, rp.bounce_risk, rp.tossed,
                   er.finish_pos, er.odds AS result_odds, er.win_payoff, er.post
            FROM result_predictions rp
            {self._pred_entry_join("rp", "er", "LEFT JOIN")}
            WHERE rp.race_date = ?
            ORDER BY rp.track, rp.race_number, rp.pick_rank
        """, (race_date,)).fetchall()

        by_track: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d.get("tags") or "[]")
            by_track.setdefault(d["track"], []).append(d)
        return by_track

    def get_prediction_roi(
        self, track: str = "", race_date: str = "", session_id: str = "",
    ) -> Dict[str, Any]:
        """ROI stats from predictions table joined with results."""
        where_parts = ["1=1"]
        params: list = []
        if track:
            where_parts.append("rp.track = ?")
            params.append(self._normalize_track(track))
        if race_date:
            where_parts.append("rp.race_date = ?")
            params.append(self._normalize_date(race_date))
        if session_id:
            where_parts.append("rp.session_id = ?")
            params.append(session_id)
        where = " AND ".join(where_parts)

        def _roi_query(group_col, group_alias):
            return self.conn.execute(f"""
                SELECT {group_col} as grp,
                       COUNT(*) as bets,
                       SUM(CASE WHEN er.finish_pos = 1 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN er.finish_pos = 1 AND er.win_payoff IS NOT NULL
                                THEN er.win_payoff ELSE 0 END) as payoff
                FROM result_predictions rp
                {self._pred_entry_join("rp", "er", "JOIN")}
                WHERE {where}
                GROUP BY grp ORDER BY grp
            """, params).fetchall()

        def _build_roi(rows, label_key):
            out = []
            for r in rows:
                b = r["bets"]
                cost = b * 2.0
                pay = r["payoff"] or 0
                roi = ((pay - cost) / cost * 100) if cost > 0 else 0
                out.append({
                    label_key: r["grp"],
                    "bets": b, "wins": r["wins"],
                    "win_pct": round(r["wins"] / b * 100, 1) if b else 0,
                    "roi_pct": round(roi, 1),
                })
            return out

        roi_rank = _build_roi(_roi_query("rp.pick_rank", "rank"), "rank")
        roi_cycle = _build_roi(_roi_query("rp.projection_type", "cycle"), "cycle")
        roi_conf = _build_roi(
            _roi_query(
                """CASE WHEN rp.confidence >= 0.8 THEN 'HIGH (80%+)'
                        WHEN rp.confidence >= 0.5 THEN 'MED (50-79%)'
                        ELSE 'LOW (<50%)' END""",
                "bucket",
            ),
            "bucket",
        )

        total_preds = self.conn.execute(
            f"SELECT COUNT(*) FROM result_predictions rp WHERE {where}", params
        ).fetchone()[0]
        matched = self.conn.execute(f"""
            SELECT COUNT(*) FROM result_predictions rp
            {self._pred_entry_join("rp", "er", "JOIN")}
            WHERE {where}
        """, params).fetchone()[0]

        return {
            "total_predictions": total_preds,
            "matched_with_results": matched,
            "match_rate": round(matched / total_preds * 100, 1) if total_preds else 0,
            "roi_by_rank": roi_rank,
            "roi_by_cycle": roi_cycle,
            "roi_by_confidence": roi_conf,
        }

    def get_prediction_roi_detailed(
        self, track: str = "", race_date: str = "", session_id: str = "",
    ) -> Dict[str, Any]:
        """Extended ROI stats with finer-grained buckets.

        Returns ROI sliced by:
        - cycle_label (projection_type)
        - confidence bucket (0-59, 60-74, 75-84, 85+)
        - odds bucket (<=2-1, 5/2-4-1, 9/2-8-1, 9-1+)
        - surface (Dirt/Turf/Poly) + distance bucket (Sprint/Route)
        """
        where_parts = ["1=1"]
        params: list = []
        if track:
            where_parts.append("rp.track = ?")
            params.append(self._normalize_track(track))
        if race_date:
            where_parts.append("rp.race_date = ?")
            params.append(self._normalize_date(race_date))
        if session_id:
            where_parts.append("rp.session_id = ?")
            params.append(session_id)
        where = " AND ".join(where_parts)

        base_join = f"""
            FROM result_predictions rp
            {self._pred_entry_join("rp", "er", "JOIN")}
            LEFT JOIN result_races rr
                ON er.track = rr.track AND er.race_date = rr.race_date
                AND er.race_number = rr.race_number
            WHERE {where}
        """

        def _roi_query(group_expr):
            return self.conn.execute(f"""
                SELECT {group_expr} as grp,
                       COUNT(*) as bets,
                       SUM(CASE WHEN er.finish_pos = 1 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN er.finish_pos = 1 AND er.win_payoff IS NOT NULL
                                THEN er.win_payoff ELSE 0 END) as payoff
                {base_join}
                GROUP BY grp ORDER BY grp
            """, params).fetchall()

        def _build(rows, label_key):
            out = []
            for r in rows:
                b = r["bets"]
                cost = b * 2.0
                pay = r["payoff"] or 0
                roi = ((pay - cost) / cost * 100) if cost > 0 else 0
                out.append({
                    label_key: r["grp"] or "UNKNOWN",
                    "N": b, "wins": r["wins"],
                    "win_pct": round(r["wins"] / b * 100, 1) if b else 0,
                    "roi_pct": round(roi, 1),
                })
            return out

        # 1. By cycle label
        roi_cycle = _build(
            _roi_query("rp.projection_type"), "cycle"
        )

        # 2. By confidence bucket (finer: 0-59, 60-74, 75-84, 85+)
        roi_conf = _build(
            _roi_query("""
                CASE WHEN rp.confidence >= 0.85 THEN '85+'
                     WHEN rp.confidence >= 0.75 THEN '75-84'
                     WHEN rp.confidence >= 0.60 THEN '60-74'
                     ELSE '0-59' END
            """),
            "confidence",
        )

        # 3. By odds bucket
        roi_odds = _build(
            _roi_query("""
                CASE WHEN er.odds IS NULL THEN 'No Odds'
                     WHEN er.odds <= 2.0 THEN '<=2-1'
                     WHEN er.odds <= 4.0 THEN '5/2-4/1'
                     WHEN er.odds <= 8.0 THEN '9/2-8/1'
                     ELSE '9/1+' END
            """),
            "odds",
        )

        # 4. By surface
        roi_surface = _build(
            _roi_query("""
                CASE WHEN UPPER(COALESCE(rr.surface, '')) LIKE '%TURF%' THEN 'Turf'
                     WHEN UPPER(COALESCE(rr.surface, '')) IN ('POLY', 'ALL WEATHER', 'SYNTHETIC') THEN 'Poly'
                     WHEN COALESCE(rr.surface, '') != '' THEN 'Dirt'
                     ELSE 'Unknown' END
            """),
            "surface",
        )

        # 5. By distance bucket (Sprint < 8f, Route >= 8f)
        roi_distance = _build(
            _roi_query("""
                CASE WHEN COALESCE(rr.distance, '') = '' THEN 'Unknown'
                     WHEN LOWER(rr.distance) LIKE '%mile%' THEN 'Route'
                     WHEN CAST(REPLACE(REPLACE(rr.distance, ' Furlongs', ''), ' Furlong', '') AS REAL) >= 8 THEN 'Route'
                     ELSE 'Sprint' END
            """),
            "distance",
        )

        # 6. By tags (comma-separated in rp.tags — aggregate in Python)
        tag_rows = self.conn.execute(f"""
            SELECT rp.tags,
                   CASE WHEN er.finish_pos = 1 THEN 1 ELSE 0 END as won,
                   CASE WHEN er.finish_pos = 1 AND er.win_payoff IS NOT NULL
                        THEN er.win_payoff ELSE 0 END as payoff
            {base_join}
        """, params).fetchall()

        tag_agg: Dict[str, Dict[str, Any]] = {}
        for row in tag_rows:
            raw_tags = row["tags"] or ""
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            if not tags:
                tags = ["NO_TAGS"]
            for tag in tags:
                if tag not in tag_agg:
                    tag_agg[tag] = {"bets": 0, "wins": 0, "payoff": 0.0}
                tag_agg[tag]["bets"] += 1
                tag_agg[tag]["wins"] += row["won"]
                tag_agg[tag]["payoff"] += row["payoff"]

        roi_tags = []
        for tag, stats in sorted(tag_agg.items()):
            b = stats["bets"]
            cost = b * 2.0
            pay = stats["payoff"]
            roi = ((pay - cost) / cost * 100) if cost > 0 else 0
            roi_tags.append({
                "tag": tag, "N": b, "wins": stats["wins"],
                "win_pct": round(stats["wins"] / b * 100, 1) if b else 0,
                "roi_pct": round(roi, 1),
            })

        # 7. By pick rank (top1 vs top2 vs top3+)
        roi_rank = _build(
            _roi_query("""
                CASE WHEN rp.pick_rank = 1 THEN 'Top Pick'
                     WHEN rp.pick_rank = 2 THEN '2nd Pick'
                     WHEN rp.pick_rank = 3 THEN '3rd Pick'
                     ELSE '4th+' END
            """),
            "rank",
        )

        # Summary counts
        total_preds = self.conn.execute(
            f"SELECT COUNT(*) FROM result_predictions rp WHERE {where}", params
        ).fetchone()[0]
        matched = self.conn.execute(f"""
            SELECT COUNT(*) {base_join}
        """, params).fetchone()[0]

        return {
            "total_predictions": total_preds,
            "matched_with_results": matched,
            "match_rate": round(matched / total_preds * 100, 1) if total_preds else 0,
            "roi_by_cycle": roi_cycle,
            "roi_by_confidence": roi_conf,
            "roi_by_odds": roi_odds,
            "roi_by_surface": roi_surface,
            "roi_by_distance": roi_distance,
            "roi_by_tags": roi_tags,
            "roi_by_rank": roi_rank,
        }

    def get_clv_summary(
        self, track: str = "", race_date: str = "", session_id: str = "",
    ) -> Dict[str, Any]:
        """Compute Closing Line Value stats.

        Joins predictions with result_entries (closing odds) and
        odds_snapshots (morning line) to measure ML → closing shift.
        Positive CLV = line moved toward our pick.
        """
        where_parts = ["1=1"]
        params: list = []
        if track:
            where_parts.append("rp.track = ?")
            params.append(self._normalize_track(track))
        if race_date:
            where_parts.append("rp.race_date = ?")
            params.append(self._normalize_date(race_date))
        if session_id:
            where_parts.append("rp.session_id = ?")
            params.append(session_id)
        where = " AND ".join(where_parts)

        rows = self.conn.execute(f"""
            SELECT rp.pick_rank, rp.confidence, rp.projection_type,
                   er.odds AS closing_odds, er.finish_pos,
                   os.odds_decimal AS ml_odds
            FROM result_predictions rp
            {self._pred_entry_join("rp", "er", "JOIN")}
            LEFT JOIN odds_snapshots os
                ON rp.track = os.track AND rp.race_date = os.race_date
                AND rp.race_number = os.race_number
                AND rp.normalized_name = os.normalized_name
                AND os.source = 'morning_line'
            WHERE {where}
              AND er.odds IS NOT NULL AND er.odds > 0
        """, params).fetchall()

        clv_rows = []
        for r in rows:
            closing = r["closing_odds"]
            ml = r["ml_odds"]
            implied_final = 1.0 / (closing + 1.0)
            implied_ml = 1.0 / (ml + 1.0) if ml and ml > 0 else None
            clv = (implied_ml - implied_final) if implied_ml is not None else None
            clv_rows.append({
                "pick_rank": r["pick_rank"],
                "confidence": r["confidence"],
                "projection_type": r["projection_type"],
                "clv": clv,
                "won": r["finish_pos"] == 1,
            })

        with_clv = [r for r in clv_rows if r["clv"] is not None]
        beat_count = sum(1 for r in with_clv if r["clv"] > 0)
        total_clv = len(with_clv)
        avg_clv = sum(r["clv"] for r in with_clv) / total_clv if total_clv else 0

        # By confidence bucket
        buckets: Dict[str, list] = {"0-59": [], "60-74": [], "75-84": [], "85+": []}
        for r in with_clv:
            c = r["confidence"] * 100
            if c >= 85:
                buckets["85+"].append(r)
            elif c >= 75:
                buckets["75-84"].append(r)
            elif c >= 60:
                buckets["60-74"].append(r)
            else:
                buckets["0-59"].append(r)

        clv_by_conf = []
        for label, bkt in buckets.items():
            if bkt:
                avg = sum(r["clv"] for r in bkt) / len(bkt)
                beat = sum(1 for r in bkt if r["clv"] > 0)
                clv_by_conf.append({
                    "confidence": label, "N": len(bkt),
                    "avg_clv": round(avg * 100, 2),
                    "beat_pct": round(beat / len(bkt) * 100, 1),
                })

        # By cycle
        cycle_groups: Dict[str, list] = {}
        for r in with_clv:
            cycle_groups.setdefault(r["projection_type"], []).append(r)
        clv_by_cycle = []
        for cyc, grp in sorted(cycle_groups.items()):
            avg = sum(r["clv"] for r in grp) / len(grp)
            beat = sum(1 for r in grp if r["clv"] > 0)
            clv_by_cycle.append({
                "cycle": cyc, "N": len(grp),
                "avg_clv": round(avg * 100, 2),
                "beat_pct": round(beat / len(grp) * 100, 1),
            })

        return {
            "total_with_clv": total_clv,
            "total_without_ml": len(clv_rows) - total_clv,
            "avg_clv_pct": round(avg_clv * 100, 2),
            "beat_closing_pct": round(beat_count / total_clv * 100, 1) if total_clv else 0,
            "clv_by_confidence": clv_by_conf,
            "clv_by_cycle": clv_by_cycle,
        }

    def get_calibration_data(
        self, min_n: int = 30, track_filter: str = "",
    ) -> Dict[str, Any]:
        """ROI grouped by (track, surface, distance) with threshold recommendations."""
        where_parts = ["1=1"]
        params: list = []
        if track_filter:
            where_parts.append("rp.track = ?")
            params.append(self._normalize_track(track_filter))
        where = " AND ".join(where_parts)

        surface_expr = """
            CASE WHEN UPPER(COALESCE(rr.surface, '')) LIKE '%TURF%' THEN 'Turf'
                 WHEN UPPER(COALESCE(rr.surface, '')) IN ('POLY', 'ALL WEATHER', 'SYNTHETIC') THEN 'Poly'
                 WHEN COALESCE(rr.surface, '') != '' THEN 'Dirt'
                 ELSE 'Unknown' END
        """
        distance_expr = """
            CASE WHEN COALESCE(rr.distance, '') = '' THEN 'Unknown'
                 WHEN LOWER(rr.distance) LIKE '%mile%' THEN 'Route'
                 WHEN CAST(REPLACE(REPLACE(rr.distance, ' Furlongs', ''), ' Furlong', '') AS REAL) >= 8 THEN 'Route'
                 ELSE 'Sprint' END
        """

        base_join = f"""
            FROM result_predictions rp
            {self._pred_entry_join("rp", "er", "JOIN")}
            LEFT JOIN result_races rr
                ON er.track = rr.track AND er.race_date = rr.race_date
                AND er.race_number = rr.race_number
            WHERE {where}
        """

        # Main bucket query: ROI by (track, surface, distance)
        bucket_rows = self.conn.execute(f"""
            SELECT rp.track,
                   {surface_expr} AS surface_cat,
                   {distance_expr} AS dist_cat,
                   COUNT(*) AS N,
                   SUM(CASE WHEN er.finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN er.finish_pos = 1 AND er.win_payoff IS NOT NULL
                            THEN er.win_payoff ELSE 0 END) AS payoff
            {base_join}
            GROUP BY rp.track, surface_cat, dist_cat
            ORDER BY rp.track, surface_cat, dist_cat
        """, params).fetchall()

        buckets = []
        for r in bucket_rows:
            n = r["N"]
            cost = n * 2.0
            pay = r["payoff"] or 0
            roi = ((pay - cost) / cost * 100) if cost > 0 else 0
            buckets.append({
                "track": r["track"],
                "surface": r["surface_cat"],
                "distance": r["dist_cat"],
                "label": f"{r['track']} {r['surface_cat']} {r['dist_cat']}",
                "N": n,
                "wins": r["wins"],
                "win_pct": round(r["wins"] / n * 100, 1) if n else 0,
                "roi_pct": round(roi, 1),
            })

        # Threshold grid search per qualifying bucket
        conf_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        odds_thresholds = [2.0, 3.0, 4.0, 5.0, 8.0]
        recommendations = []

        for b in buckets:
            if b["N"] < min_n:
                continue
            tv, sv, dv = b["track"], b["surface"], b["distance"]

            best_conf_roi, best_conf_thresh, best_conf_n = -999.0, 0.60, 0
            for ct in conf_thresholds:
                row = self.conn.execute(f"""
                    SELECT COUNT(*) AS N,
                           SUM(CASE WHEN er.finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
                           SUM(CASE WHEN er.finish_pos = 1 AND er.win_payoff IS NOT NULL
                                    THEN er.win_payoff ELSE 0 END) AS payoff
                    {base_join}
                    AND rp.track = ? AND {surface_expr} = ? AND {distance_expr} = ?
                    AND rp.confidence >= ?
                """, params + [tv, sv, dv, ct]).fetchone()
                n = row["N"]
                if n >= min_n:
                    cost = n * 2.0
                    pay = row["payoff"] or 0
                    roi = ((pay - cost) / cost * 100) if cost > 0 else 0
                    if roi > best_conf_roi:
                        best_conf_roi, best_conf_thresh, best_conf_n = roi, ct, n

            best_odds_roi, best_odds_thresh, best_odds_n = -999.0, 2.0, 0
            for ot in odds_thresholds:
                row = self.conn.execute(f"""
                    SELECT COUNT(*) AS N,
                           SUM(CASE WHEN er.finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
                           SUM(CASE WHEN er.finish_pos = 1 AND er.win_payoff IS NOT NULL
                                    THEN er.win_payoff ELSE 0 END) AS payoff
                    {base_join}
                    AND rp.track = ? AND {surface_expr} = ? AND {distance_expr} = ?
                    AND er.odds >= ?
                """, params + [tv, sv, dv, ot]).fetchone()
                n = row["N"]
                if n >= min_n:
                    cost = n * 2.0
                    pay = row["payoff"] or 0
                    roi = ((pay - cost) / cost * 100) if cost > 0 else 0
                    if roi > best_odds_roi:
                        best_odds_roi, best_odds_thresh, best_odds_n = roi, ot, n

            recommendations.append({
                "label": b["label"],
                "track": tv, "surface": sv, "distance": dv,
                "total_N": b["N"], "total_roi": b["roi_pct"],
                "best_conf_threshold": best_conf_thresh,
                "best_conf_roi": round(best_conf_roi, 1),
                "best_conf_n": best_conf_n,
                "best_odds_threshold": best_odds_thresh,
                "best_odds_roi": round(best_odds_roi, 1),
                "best_odds_n": best_odds_n,
            })

        return {"buckets": buckets, "recommendations": recommendations, "min_n": min_n}

    def get_detailed_roi(
        self, track_filter: str = "", min_n: int = 5,
    ) -> Dict[str, Any]:
        """ROI breakdowns by cycle (projection_type), confidence tier, odds bucket,
        and pick rank. Returns top leaks and tuning suggestions."""
        where_parts = ["er.finish_pos IS NOT NULL"]
        params: list = []
        if track_filter:
            where_parts.append("rp.track = ?")
            params.append(self._normalize_track(track_filter))
        where = " AND ".join(where_parts)

        base_join = f"""
            FROM result_predictions rp
            {self._pred_entry_join("rp", "er", "JOIN")}
            WHERE {where}
        """

        # --- By cycle (projection_type) ---
        cycle_rows = self.conn.execute(f"""
            SELECT rp.projection_type AS cycle,
                   COUNT(*) AS N,
                   SUM(CASE WHEN er.finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN er.finish_pos = 1 AND er.win_payoff IS NOT NULL
                            THEN er.win_payoff ELSE 0 END) AS payoff
            {base_join}
            AND rp.pick_rank = 1
            GROUP BY rp.projection_type
            ORDER BY N DESC
        """, params).fetchall()
        by_cycle = []
        for r in cycle_rows:
            n = r["N"]
            cost = n * 2.0
            pay = r["payoff"] or 0
            roi = ((pay - cost) / cost * 100) if cost > 0 else 0
            by_cycle.append({
                "cycle": r["cycle"], "N": n, "wins": r["wins"],
                "win_pct": round(r["wins"] / n * 100, 1) if n else 0,
                "roi_pct": round(roi, 1),
            })

        # --- By confidence tier (top pick only) ---
        conf_tiers = [
            ("High (>=70%)", 0.70, 1.01),
            ("Medium (55-69%)", 0.55, 0.70),
            ("Low (<55%)", 0.0, 0.55),
        ]
        by_confidence = []
        for label, lo, hi in conf_tiers:
            row = self.conn.execute(f"""
                SELECT COUNT(*) AS N,
                       SUM(CASE WHEN er.finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN er.finish_pos = 1 AND er.win_payoff IS NOT NULL
                                THEN er.win_payoff ELSE 0 END) AS payoff
                {base_join}
                AND rp.pick_rank = 1
                AND rp.confidence >= ? AND rp.confidence < ?
            """, params + [lo, hi]).fetchone()
            n = row["N"]
            if n > 0:
                cost = n * 2.0
                pay = row["payoff"] or 0
                roi = ((pay - cost) / cost * 100) if cost > 0 else 0
                by_confidence.append({
                    "tier": label, "N": n, "wins": row["wins"],
                    "win_pct": round(row["wins"] / n * 100, 1),
                    "roi_pct": round(roi, 1),
                })

        # --- By odds bucket (top pick only) ---
        odds_buckets = [
            ("Even-2/1", 0.0, 3.0),
            ("5/2-4/1", 3.0, 5.0),
            ("9/2-7/1", 5.0, 8.0),
            ("8/1-14/1", 8.0, 15.0),
            ("15/1+", 15.0, 9999.0),
        ]
        by_odds = []
        for label, lo, hi in odds_buckets:
            row = self.conn.execute(f"""
                SELECT COUNT(*) AS N,
                       SUM(CASE WHEN er.finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN er.finish_pos = 1 AND er.win_payoff IS NOT NULL
                                THEN er.win_payoff ELSE 0 END) AS payoff
                {base_join}
                AND rp.pick_rank = 1
                AND er.odds >= ? AND er.odds < ?
            """, params + [lo, hi]).fetchone()
            n = row["N"]
            if n > 0:
                cost = n * 2.0
                pay = row["payoff"] or 0
                roi = ((pay - cost) / cost * 100) if cost > 0 else 0
                by_odds.append({
                    "bucket": label, "N": n, "wins": row["wins"],
                    "win_pct": round(row["wins"] / n * 100, 1),
                    "roi_pct": round(roi, 1),
                })

        # --- By pick rank (1-5) ---
        rank_rows = self.conn.execute(f"""
            SELECT rp.pick_rank AS rank,
                   COUNT(*) AS N,
                   SUM(CASE WHEN er.finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN er.finish_pos = 1 AND er.win_payoff IS NOT NULL
                            THEN er.win_payoff ELSE 0 END) AS payoff
            {base_join}
            AND rp.pick_rank <= 5
            GROUP BY rp.pick_rank
            ORDER BY rp.pick_rank
        """, params).fetchall()
        by_rank = []
        for r in rank_rows:
            n = r["N"]
            cost = n * 2.0
            pay = r["payoff"] or 0
            roi = ((pay - cost) / cost * 100) if cost > 0 else 0
            by_rank.append({
                "rank": r["rank"], "N": n, "wins": r["wins"],
                "win_pct": round(r["wins"] / n * 100, 1) if n else 0,
                "roi_pct": round(roi, 1),
            })

        # --- Top leaks: worst ROI spots ---
        leaks = []
        all_rows = by_cycle + [
            {**c, "cycle": c.get("tier", c.get("bucket", ""))} for c in by_confidence + by_odds
        ]
        for row in all_rows:
            if row["N"] >= min_n and row["roi_pct"] < -20:
                leaks.append({
                    "spot": row.get("cycle") or row.get("tier") or row.get("bucket", ""),
                    "N": row["N"], "roi_pct": row["roi_pct"],
                    "win_pct": row["win_pct"],
                })
        leaks.sort(key=lambda x: x["roi_pct"])

        # --- Tuning suggestions ---
        suggestions = []
        for c in by_cycle:
            if c["N"] >= min_n:
                if c["roi_pct"] > 10:
                    suggestions.append(f"Keep betting {c['cycle']} (ROI {c['roi_pct']:+.1f}%, N={c['N']})")
                elif c["roi_pct"] < -30:
                    suggestions.append(f"Consider avoiding {c['cycle']} top picks (ROI {c['roi_pct']:+.1f}%, N={c['N']})")
        for o in by_odds:
            if o["N"] >= min_n and o["roi_pct"] > 10:
                suggestions.append(f"Profitable in {o['bucket']} range (ROI {o['roi_pct']:+.1f}%, N={o['N']})")
        for c in by_confidence:
            if c["N"] >= min_n and c["roi_pct"] > 10:
                suggestions.append(f"{c['tier']} confidence is profitable (ROI {c['roi_pct']:+.1f}%, N={c['N']})")

        return {
            "by_cycle": by_cycle,
            "by_confidence": by_confidence,
            "by_odds": by_odds,
            "by_rank": by_rank,
            "leaks": leaks[:5],
            "suggestions": suggestions,
        }

    def get_all_bets(
        self, track: str = "", race_date: str = "", session_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Return every prediction joined with outcome for audit/export."""
        where_parts = ["1=1"]
        params: list = []
        if track:
            where_parts.append("rp.track = ?")
            params.append(self._normalize_track(track))
        if race_date:
            where_parts.append("rp.race_date = ?")
            params.append(self._normalize_date(race_date))
        if session_id:
            where_parts.append("rp.session_id = ?")
            params.append(session_id)
        where = " AND ".join(where_parts)

        rows = self.conn.execute(f"""
            SELECT rp.session_id, rp.track, rp.race_date, rp.race_number,
                   rp.horse_name, rp.pick_rank, rp.projection_type,
                   rp.bias_score, rp.confidence,
                   rp.projected_low, rp.projected_high,
                   rp.new_top_setup, rp.bounce_risk, rp.tossed,
                   er.post, er.finish_pos, er.odds,
                   er.win_payoff, er.place_payoff, er.show_payoff,
                   rr.surface, rr.distance
            FROM result_predictions rp
            {self._pred_entry_join("rp", "er", "LEFT JOIN")}
            LEFT JOIN result_races rr
                ON er.track = rr.track AND er.race_date = rr.race_date
                AND er.race_number = rr.race_number
            WHERE {where}
            ORDER BY rp.race_date, rp.race_number, rp.pick_rank
        """, params).fetchall()

        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Bet Plans
    # ------------------------------------------------------------------

    def save_bet_plan(
        self, session_id: str, track: str, race_date: str,
        settings_dict: Dict[str, Any], plan_dict: Dict[str, Any],
        total_risk: float, paper_mode: bool, engine_version: str = "",
        plan_type: str = "session",
    ) -> int:
        """Persist a generated bet plan. Returns the plan_id."""
        track = self._normalize_track(track)
        race_date = self._normalize_date(race_date)
        now = datetime.utcnow().isoformat()
        cur = self.conn.execute(
            """
            INSERT INTO bet_plans(
                session_id, track, race_date, settings_json, plan_json,
                total_risk, paper_mode, engine_version, plan_type, created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                session_id, track, race_date,
                json.dumps(settings_dict), json.dumps(plan_dict),
                total_risk, int(paper_mode), engine_version, plan_type, now,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def load_bet_plans(
        self, session_id: str = "", track: str = "", race_date: str = "",
    ) -> List[Dict[str, Any]]:
        """Load stored bet plans with optional filters."""
        where_parts = ["1=1"]
        params: list = []
        if session_id:
            where_parts.append("session_id = ?")
            params.append(session_id)
        if track:
            where_parts.append("track = ?")
            params.append(self._normalize_track(track))
        if race_date:
            where_parts.append("race_date = ?")
            params.append(self._normalize_date(race_date))
        where = " AND ".join(where_parts)

        rows = self.conn.execute(
            f"""SELECT plan_id, session_id, track, race_date,
                       settings_json, plan_json, total_risk,
                       paper_mode, engine_version, created_at
                FROM bet_plans WHERE {where}
                ORDER BY created_at DESC""",
            params,
        ).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            d["settings"] = json.loads(d.pop("settings_json"))
            d["plan"] = json.loads(d.pop("plan_json"))
            d["paper_mode"] = bool(d["paper_mode"])
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Odds snapshots
    # ------------------------------------------------------------------

    def save_odds_snapshots(
        self, session_id: str, track: str, race_date: str,
        snapshots: List[Dict[str, Any]], source: str = "morning_line",
    ) -> int:
        """Persist ML odds from BRISNET upload.

        Each snapshot dict: {race_number, post, horse_name, odds_raw, odds_decimal}
        Returns row count.
        """
        track = self._normalize_track(track)
        race_date = self._normalize_date(race_date)
        now = datetime.utcnow().isoformat()
        count = 0
        for s in snapshots:
            norm = self._normalize_name(s.get("horse_name", ""))
            self.conn.execute(
                """
                INSERT INTO odds_snapshots(
                    session_id, track, race_date, race_number, post,
                    horse_name, normalized_name, odds_raw, odds_decimal,
                    source, captured_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id, track, race_date, race_number, normalized_name, source)
                DO UPDATE SET
                    odds_raw=excluded.odds_raw,
                    odds_decimal=excluded.odds_decimal,
                    post=excluded.post,
                    captured_at=excluded.captured_at
                """,
                (
                    session_id, track, race_date,
                    s.get("race_number", 0), s.get("post"),
                    s.get("horse_name", ""), norm,
                    s.get("odds_raw", ""), s.get("odds_decimal"),
                    source, now,
                ),
            )
            count += 1
        self.conn.commit()
        return count

    def get_odds_snapshots(
        self, track: str, race_date: str,
        source: str = "morning_line",
    ) -> Dict[Tuple[int, str], float]:
        """Return {(race_number, normalized_name): odds_decimal} for quick lookup."""
        track = self._normalize_track(track)
        race_date = self._normalize_date(race_date)
        rows = self.conn.execute(
            """
            SELECT race_number, normalized_name, odds_decimal
            FROM odds_snapshots
            WHERE track = ? AND race_date = ? AND source = ?
              AND odds_decimal IS NOT NULL
            """,
            (track, race_date, source),
        ).fetchall()
        return {(r["race_number"], r["normalized_name"]): r["odds_decimal"] for r in rows}

    def get_odds_snapshots_full(
        self, track: str, race_date: str,
        source: str = "morning_line",
    ) -> List[Dict[str, Any]]:
        """Return full odds snapshot rows for a card (includes odds_raw)."""
        track = self._normalize_track(track)
        race_date = self._normalize_date(race_date)
        rows = self.conn.execute(
            """SELECT race_number, post, horse_name, normalized_name,
                      odds_raw, odds_decimal
               FROM odds_snapshots
               WHERE track = ? AND race_date = ? AND source = ?""",
            (track, race_date, source),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_odds_for_date(
        self, race_date: str, source: str = "morning_line",
    ) -> Dict[tuple, dict]:
        """Return {(track, race_number, normalized_name): odds_dict} for an entire date."""
        race_date = self._normalize_date(race_date)
        rows = self.conn.execute("""
            SELECT track, race_number, post, normalized_name, odds_raw, odds_decimal
            FROM odds_snapshots
            WHERE race_date = ? AND source = ?
        """, (race_date, source)).fetchall()
        return {
            (r["track"], r["race_number"], r["normalized_name"]): dict(r)
            for r in rows
        }

    def evaluate_bet_plan_roi(self, plan_id: int) -> Dict[str, Any]:
        """Compute realized ROI for a stored bet plan by joining tickets
        against result_entries using tiered matching.

        Match priority per horse:
          HIGH — (track, race_date, race_number, post)
          MED  — (track, race_date, race_number, normalized_name)
          LOW  — normalized horse_name from selection text

        Returns dict with plan_id, total_wagered, total_returned, roi_pct,
        ticket_results (list of per-ticket outcomes), resolved/unresolved
        counts, and evaluated_at.
        """
        row = self.conn.execute(
            "SELECT * FROM bet_plans WHERE plan_id = ?", (plan_id,)
        ).fetchone()
        if not row:
            return {"error": f"plan_id {plan_id} not found"}

        plan_data = json.loads(row["plan_json"])
        track = row["track"]
        race_date = row["race_date"]

        # Pre-load all result entries for this card for efficient lookups
        all_entries = self.conn.execute(
            """SELECT post, horse_name, normalized_name, finish_pos, odds,
                      win_payoff, race_number
               FROM result_entries
               WHERE track = ? AND race_date = ?""",
            (track, race_date),
        ).fetchall()

        # Build lookup structures per race
        # by_post:  {race_number: {post: entry_dict}}
        # by_name:  {race_number: {normalized_name: entry_dict}}
        by_post: Dict[int, Dict[int, dict]] = {}
        by_name: Dict[int, Dict[str, dict]] = {}
        for e in all_entries:
            rn = e["race_number"]
            ed = dict(e)
            if rn not in by_post:
                by_post[rn] = {}
                by_name[rn] = {}
            if e["post"] is not None:
                by_post[rn][e["post"]] = ed
            if e["normalized_name"]:
                by_name[rn][e["normalized_name"]] = ed

        def _resolve_horse(race_num: int, details: dict, sel_name: str):
            """Resolve a single horse selection to a result entry.

            Returns (entry_dict_or_None, match_tier: str, reason: str).
            """
            post = details.get("post")
            stored_norm = details.get("normalized_name", "")
            sel_norm = self._normalize_name(sel_name)

            race_posts = by_post.get(race_num, {})
            race_names = by_name.get(race_num, {})

            # No results for this race at all
            if not race_posts and not race_names:
                return None, "none", "no results for race"

            # HIGH: match by post
            if post and isinstance(post, int) and post in race_posts:
                return race_posts[post], "high", "matched by post"

            # MED: match by normalized_name from ticket details
            if stored_norm and stored_norm in race_names:
                return race_names[stored_norm], "med", "matched by normalized_name"

            # LOW: normalize the selection text itself
            if sel_norm and sel_norm in race_names:
                return race_names[sel_norm], "low", "matched by selection name"

            return None, "none", "unresolved"

        total_wagered = 0.0
        total_returned = 0.0
        ticket_results = []
        resolved_count = 0
        unresolved_count = 0

        for rp in plan_data.get("race_plans", []):
            race_num = rp.get("race_number", 0)
            for t in rp.get("tickets", []):
                cost = t.get("cost", 0)
                total_wagered += cost
                bet_type = t.get("bet_type", "")
                selections = t.get("selections", [])
                details = t.get("details", {})
                returned = 0.0
                outcome = "no_result"
                match_tier = "none"
                match_reason = ""

                if bet_type == "WIN" and selections:
                    entry, match_tier, match_reason = _resolve_horse(
                        race_num, details, selections[0],
                    )
                    if entry and entry.get("finish_pos") is not None:
                        if entry["finish_pos"] == 1:
                            payoff = entry.get("win_payoff") or 0
                            returned = (cost / 2.0) * payoff if payoff else 0
                            outcome = "won"
                        else:
                            outcome = "lost"
                        resolved_count += 1
                    elif entry:
                        # Entry exists but no finish position
                        outcome = "no_result"
                        match_reason = "entry found but no finish_pos"
                        unresolved_count += 1
                    else:
                        unresolved_count += 1

                elif bet_type == "EXACTA" and len(selections) >= 2:
                    structure = details.get("structure", "key")
                    horses_ids = details.get("horses", {})

                    # Resolve each selection to finish position
                    resolved_finishes = {}
                    all_matched = True
                    for sel in selections:
                        h_ids = horses_ids.get(sel, {})
                        entry, tier, reason = _resolve_horse(race_num, h_ids, sel)
                        if entry and entry.get("finish_pos") is not None:
                            resolved_finishes[sel] = entry["finish_pos"]
                        else:
                            all_matched = False

                    if not resolved_finishes:
                        outcome = "no_result"
                        match_tier = "none"
                        match_reason = "no horses resolved"
                        unresolved_count += 1
                    else:
                        match_tier = "high"
                        if structure == "key":
                            top_name = selections[0]
                            top_fin = resolved_finishes.get(top_name)
                            if top_fin == 1:
                                for u in selections[1:]:
                                    if resolved_finishes.get(u) == 2:
                                        outcome = "won"
                                        break
                                else:
                                    outcome = "lost"
                            elif top_fin is not None:
                                outcome = "lost"
                            else:
                                outcome = "no_result"
                                match_reason = "top horse unresolved"
                        else:  # saver
                            under_name = details.get("under", "")
                            under_fin = resolved_finishes.get(under_name)
                            if under_fin == 2:
                                for o in details.get("overs", []):
                                    if resolved_finishes.get(o) == 1:
                                        outcome = "won"
                                        break
                                else:
                                    outcome = "lost"
                            elif under_fin is not None:
                                outcome = "lost"
                            else:
                                outcome = "no_result"
                                match_reason = "under horse unresolved"

                        if outcome in ("won", "lost"):
                            resolved_count += 1
                        else:
                            unresolved_count += 1

                    # Exacta payoff not tracked per combination
                    if outcome == "won":
                        returned = 0

                total_returned += returned
                ticket_results.append({
                    "race_number": race_num,
                    "bet_type": bet_type,
                    "selections": selections,
                    "cost": cost,
                    "outcome": outcome,
                    "returned": returned,
                    "match_tier": match_tier,
                    "match_reason": match_reason,
                })

        total_tickets = resolved_count + unresolved_count
        roi_pct = ((total_returned - total_wagered) / total_wagered * 100) if total_wagered > 0 else 0

        return {
            "plan_id": plan_id,
            "track": track,
            "race_date": race_date,
            "total_wagered": round(total_wagered, 2),
            "total_returned": round(total_returned, 2),
            "roi_pct": round(roi_pct, 1),
            "resolved": resolved_count,
            "unresolved": unresolved_count,
            "total_tickets": total_tickets,
            "ticket_results": ticket_results,
            "evaluated_at": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Trainer bucket stats
    # ------------------------------------------------------------------

    def upsert_trainer_bucket(
        self, trainer: str, bucket_type: str, starts: int, wins: int,
        itm: int, roi: float, bucket_value: str = None,
    ) -> None:
        """Insert or update a single trainer bucket stat row."""
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """INSERT INTO trainer_bucket_stats
                   (trainer, bucket_type, bucket_value, starts, wins, itm, roi, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(trainer, bucket_type) DO UPDATE SET
                   starts=excluded.starts, wins=excluded.wins,
                   itm=excluded.itm, roi=excluded.roi,
                   bucket_value=excluded.bucket_value,
                   updated_at=excluded.updated_at""",
            (trainer, bucket_type, bucket_value, starts, wins, itm, roi, now),
        )
        self.conn.commit()

    def get_trainer_buckets(self, trainer: str) -> List[Dict]:
        """Return all bucket stats for a trainer."""
        cur = self.conn.execute(
            """SELECT bucket_type, bucket_value, starts, wins, itm, roi, updated_at
               FROM trainer_bucket_stats WHERE trainer = ?""",
            (trainer,),
        )
        return [
            {
                "bucket_type": r[0], "bucket_value": r[1],
                "starts": r[2], "wins": r[3], "itm": r[4],
                "roi": r[5], "updated_at": r[6],
            }
            for r in cur.fetchall()
        ]

    def get_trainer_buckets_bulk(self, trainers: List[str]) -> Dict[str, List[Dict]]:
        """Return bucket stats for multiple trainers in one query."""
        if not trainers:
            return {}
        placeholders = ",".join("?" for _ in trainers)
        cur = self.conn.execute(
            f"""SELECT trainer, bucket_type, bucket_value, starts, wins, itm, roi, updated_at
                FROM trainer_bucket_stats WHERE trainer IN ({placeholders})""",
            tuple(trainers),
        )
        result: Dict[str, List[Dict]] = {t: [] for t in trainers}
        for r in cur.fetchall():
            result.setdefault(r[0], []).append({
                "bucket_type": r[1], "bucket_value": r[2],
                "starts": r[3], "wins": r[4], "itm": r[5],
                "roi": r[6], "updated_at": r[7],
            })
        return result
