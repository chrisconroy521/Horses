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

            /* ============================================================
               Cumulative horse database tables
               ============================================================ */

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
            """
        )
        self.conn.commit()
        self._ensure_session_columns()

    def _ensure_session_columns(self) -> None:
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
        self.conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Uppercase, strip punctuation, collapse whitespace."""
        n = (name or "").upper().strip()
        n = re.sub(r"['\-.,()]", "", n)
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
        """Match sheets horses <-> brisnet horses.

        Pass 1: (track, race_date, race_number, post) -> 'high'
        Pass 2: (normalized_name, track, race_date) -> 'medium'
        """
        now = datetime.utcnow().isoformat()
        new_matches = 0

        # Pass 1: High confidence — track + date + race_number + post
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
                    WHERE r.horse_id = sh.horse_id AND r.brisnet_id = bh.brisnet_id
                )
            """
        ).fetchall()

        for row in rows:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO reconciliation
                    (horse_id, brisnet_id, match_method, confidence, created_at)
                VALUES (?, ?, 'track_date_race_post', 'high', ?)
                """,
                (row["horse_id"], row["brisnet_id"], now),
            )
            new_matches += 1

        # Pass 2: Medium confidence — normalized_name + track + race_date
        # Only for sheets horses not yet reconciled
        rows2 = self.conn.execute(
            """
            SELECT sh.horse_id, bh.brisnet_id
            FROM sheets_horses sh
            JOIN brisnet_horses bh
                ON sh.normalized_name = bh.normalized_name
                AND sh.track = bh.track
                AND sh.race_date = bh.race_date
            WHERE sh.track != '' AND sh.race_date != ''
                AND NOT EXISTS (
                    SELECT 1 FROM reconciliation r
                    WHERE r.horse_id = sh.horse_id
                )
            """
        ).fetchall()

        for row in rows2:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO reconciliation
                    (horse_id, brisnet_id, match_method, confidence, created_at)
                VALUES (?, ?, 'normalized_name_track_date', 'medium', ?)
                """,
                (row["horse_id"], row["brisnet_id"], now),
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

        return {
            "sheets_horses": sh,
            "sheets_lines": sl,
            "brisnet_horses": bh,
            "brisnet_lines": bl,
            "reconciled_pairs": rp,
            "uploads_count": up,
            "tracks": tracks,
            "coverage_pct": round(coverage_pct, 1),
        }
