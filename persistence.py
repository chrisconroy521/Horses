"""SQLite persistence for handicapping sessions and cumulative horse database."""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


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
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
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
            """
        )
        self.conn.commit()
        self._ensure_columns()

    def _ensure_columns(self) -> None:
        # sessions table
        cur = self.conn.execute("PRAGMA table_info(sessions)")
        existing = {row[1] for row in cur.fetchall()}
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
        cur2 = self.conn.execute("PRAGMA table_info(result_entries)")
        re_cols = {row[1] for row in cur2.fetchall()}
        if re_cols and "session_id" not in re_cols:
            self.conn.execute("ALTER TABLE result_entries ADD COLUMN session_id TEXT")
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
        except sqlite3.IntegrityError:
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

    def _prepare_session_row(self, row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
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

        return {
            "total_races": total_races,
            "total_entries": total_entries,
            "total_with_odds": total_with_odds,
            "linked_entries": linked_entries,
            "linking_rate": round(linking_rate, 1),
            "roi_by_rank": roi_by_rank,
            "roi_by_cycle": roi_by_cycle,
        }

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
