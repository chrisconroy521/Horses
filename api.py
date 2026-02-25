from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import uvicorn
import os
import json
import uuid
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

import re as _re
import fitz as _fitz
from dataclasses import asdict
from ragozin_parser import RagozinParser, RaceData
from gpt_parser_alternative import GPTRagozinParserAlternative
from merge import merge_parsed_sessions, merge_primary_secondary
from brisnet_parser import BrisnetParser, to_pipeline_json as brisnet_to_pipeline, parse_odds_decimal
from persistence import Persistence
from bet_builder import (
    BetSettings, build_day_plan, day_plan_to_dict, day_plan_to_text, day_plan_to_csv,
    build_daily_wins, DailyWinCandidate,
    MultiRaceStrategy, build_multi_race_plan, multi_race_plan_to_dict,
    build_daily_double_plan, daily_double_plan_to_dict,
    DualModeSettings, build_dual_mode_day_plan, dual_mode_plan_to_dict,
)
import hashlib
import subprocess as _subprocess
import dotenv
dotenv.load_dotenv()

# Cumulative horse database
import sys as _sys
print("[api] Connecting to database...", flush=True, file=_sys.stderr)
_db = Persistence(Path("horses.db"))
print(f"[api] Database ready (backend={_db.db_backend})", flush=True, file=_sys.stderr)

# Engine version — git short hash at startup
def _get_engine_version() -> str:
    try:
        h = _subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent),
            stderr=_subprocess.DEVNULL,
        ).decode().strip()
        return h
    except Exception:
        return "unknown"

_ENGINE_VERSION = _get_engine_version()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ragozin Sheets Parser API", version="1.0.0")

# Add CORS middleware
_allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:8502"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session middleware (required by authlib for OAuth state)
from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("JWT_SECRET", "dev-insecure-secret"),
)

# ---------------------------------------------------------------------------
# Auth: Google OAuth routes + token middleware
# ---------------------------------------------------------------------------
from auth import AUTH_ENABLED, decode_token

if AUTH_ENABLED:
    from auth import get_oauth, create_token, is_email_allowed, APP_URL

    @app.get("/auth/google")
    async def auth_google(request: Request):
        """Redirect to Google OAuth consent screen."""
        oauth = get_oauth()
        redirect_uri = str(request.url_for("auth_callback"))
        return await oauth.google.authorize_redirect(request, redirect_uri)

    @app.get("/auth/callback")
    async def auth_callback(request: Request):
        """Handle Google OAuth callback, mint JWT, redirect to frontend."""
        oauth = get_oauth()
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get("userinfo", {})
        email = (user_info.get("email") or "").lower()
        if not is_email_allowed(email):
            return RedirectResponse(f"{APP_URL}?error=unauthorized")
        jwt_token = create_token(email, user_info.get("name", ""))
        return RedirectResponse(f"{APP_URL}?token={jwt_token}")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Protect API endpoints with Bearer token when auth is enabled."""
    path = request.url.path
    # Always allow health, root, docs, and auth routes
    if (
        path in ("/", "/health", "/docs", "/openapi.json", "/redoc")
        or path.startswith("/auth/")
    ):
        return await call_next(request)
    # Skip auth when not configured (local dev)
    if not AUTH_ENABLED:
        return await call_next(request)
    # Check Bearer token
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401, content={"detail": "Not authenticated"}
        )
    try:
        decode_token(auth_header.split(" ", 1)[1])
    except Exception:
        return JSONResponse(
            status_code=401, content={"detail": "Invalid or expired token"}
        )
    return await call_next(request)


# Initialize parsers
parser = RagozinParser()
brisnet_parser = BrisnetParser()
gpt_parser = None


def detect_pdf_type(pdf_path: str) -> str:
    """Detect PDF type from content of first 3 pages.

    Returns one of:
      'sheets'     – Ragozin-style sheets (page markers like ``CD  p1``)
      'brisnet'    – BRISNET Ultimate PPs w/ QuickPlay
      'brisnet_pp' – BRISNET PPs without QuickPlay
      'unknown'    – unrecognised
    """
    try:
        doc = _fitz.open(pdf_path)
        pages_to_check = min(3, len(doc))
        combined = ''
        for i in range(pages_to_check):
            combined += doc[i].get_text()[:2000] + '\n'
        doc.close()
        lower = combined.lower()

        # Equibase results chart detection (before brisnet check)
        has_mutuel = '$2 mutuel prices' in lower or '$2 mutuel' in lower
        has_race_header = _re.search(
            r'(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|'
            r'TENTH|ELEVENTH|TWELFTH|THIRTEENTH|FOURTEENTH)\s+RACE',
            combined,
        )
        if has_mutuel and has_race_header:
            return 'equibase_chart'

        is_brisnet = 'brisnet.com' in lower or 'ultimate pp' in lower
        has_quickplay = 'quickplay' in lower or 'QuickPlay' in combined

        if is_brisnet and has_quickplay:
            return 'brisnet'
        if is_brisnet:
            return 'brisnet_pp'

        # Ragozin sheets markers: track codes followed by page refs like "CD  p1"
        if _re.search(r'[A-Z]{2,3}\s+p\d', combined):
            return 'sheets'
    except Exception:
        pass
    return 'unknown'

# Valid PDF types that can be used as primary / secondary uploads
PRIMARY_TYPES = frozenset({'sheets', 'brisnet'})
SECONDARY_TYPES = frozenset({'brisnet', 'brisnet_pp'})

# Try to initialize GPT parser if API key is available
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        gpt_parser = GPTRagozinParserAlternative(api_key=api_key)
        logger.info("GPT-enhanced parser initialized successfully")
    else:
        logger.warning("OpenAI API key not found. GPT parser will not be available")
except Exception as e:
    logger.warning(f"Failed to initialize GPT parser: {e}")

# Create directories for file storage
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory storage for parsed data (in production, use a database)
parsed_races: Dict[str, Dict[str, Any]] = {}

# New dual-upload session storage
sessions: Dict[str, Dict[str, Any]] = {}


def _trad_to_dict(race_data: RaceData) -> Dict[str, Any]:
    """Convert traditional RaceData to the JSON-output dict schema."""
    horses_dict = []
    for horse in race_data.horses:
        # Build lines from past_performances (new) or fallback to single line
        lines = []
        if horse.past_performances:
            for pp in horse.past_performances:
                line = {
                    'raw_text': pp.get('raw_text', ''),
                    'fig': str(pp.get('parsed_figure', '')),
                    'parsed_figure': pp.get('parsed_figure'),
                    'flags': pp.get('flags', []),
                    'surface': pp.get('surface', 'DIRT'),
                    'surface_type': pp.get('surface', 'DIRT'),
                    'track': pp.get('track', race_data.track_name),
                    'data_text': pp.get('data_text', ''),
                    'race_type': 'Unknown',
                    'race_date': horse.race_date,
                    'notes': '',
                    'race_analysis': '',
                }
                lines.append(line)
        if not lines:
            # Fallback: single line from old-style fields
            lines = [{
                'fig': str(horse.figure) if horse.figure is not None else '',
                'parsed_figure': horse.figure,
                'flags': [],
                'surface': horse.track_surface,
                'track': race_data.track_name,
                'race_type': 'Unknown',
                'race_date': horse.race_date,
                'notes': '',
                'race_analysis': '',
            }]

        horse_dict = {
            'horse_name': horse.horse_name,
            'race_number': horse.race_number,
            'sex': 'Unknown',
            'age': getattr(horse, 'age', 0),
            'breeder_owner': 'Unknown',
            'foal_date': 'Unknown',
            'reg_code': 'Unknown',
            'races': len(lines),
            'top_fig': str(horse.figure) if horse.figure is not None else '',
            'horse_analysis': f"Ragozin Figure: {horse.figure}",
            'performance_trend': f"{len(lines)} past performances",
            'lines': lines,
        }
        horses_dict.append(horse_dict)
    return {
        'track_name': race_data.track_name,
        'race_date': race_data.race_date,
        'race_number': race_data.race_number,
        'surface': race_data.surface,
        'distance': race_data.distance,
        'weather': race_data.weather,
        'race_conditions': race_data.race_conditions,
        'horses': horses_dict,
    }


def _gpt_to_dict(horse_performance) -> Dict[str, Any]:
    """Convert GPT HorsePastPerformance to the JSON-output dict schema."""
    return asdict(horse_performance)


# ---------------------------------------------------------------------------
# Startup recovery
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _recover_sessions():
    """Rebuild sessions dict from DB and *_primary.json files on disk."""
    # DB-based recovery (works on Railway where files are ephemeral)
    try:
        for sess in _db.list_sessions():
            sid = sess.get("session_id")
            if sid and sid not in sessions:
                sessions[sid] = {
                    "session_id": sid,
                    "track": sess.get("track_name", ""),
                    "date": sess.get("race_date", ""),
                    "has_primary": True,
                    "has_secondary": False,
                    "primary_pdf_type": sess.get("parser_used", ""),
                    "primary_pdf_filename": sess.get("pdf_name", ""),
                    "primary_horses_count": sess.get("horses_count", 0),
                    "created_at": sess.get("created_at", ""),
                }
                logger.info(f"Recovered session {sid} from database")
    except Exception as e:
        logger.warning(f"DB-based session recovery failed: {e}")

    # File-based recovery (local dev, overrides DB data with richer metadata)
    for path in OUTPUT_DIR.glob("*_primary.json"):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            meta = data.get('_session_meta')
            if meta and meta.get('session_id'):
                sid = meta['session_id']
                sessions[sid] = meta
                logger.info(f"Recovered session {sid} from {path.name}")
        except Exception as e:
            logger.warning(f"Could not recover session from {path.name}: {e}")

    # Patch sessions that have a _secondary.json on disk but meta missed it
    for path in OUTPUT_DIR.glob("*_secondary.json"):
        sid = path.stem.replace('_secondary', '')
        if sid in sessions and not sessions[sid].get('has_secondary'):
            try:
                with open(path, 'r') as f:
                    sec_data = json.load(f)
                sec_horses = sec_data.get('horses', [])
                sessions[sid].update({
                    "has_secondary": True,
                    "secondary_pdf_type": sec_data.get('parse_source', 'brisnet'),
                    "secondary_pdf_filename": sec_data.get('pdf_filename') or path.name,
                    "secondary_output_file": str(path),
                    "secondary_horses_count": len(sec_horses),
                })
                logger.info(f"Patched secondary info for session {sid} from {path.name}")
            except Exception as e:
                logger.warning(f"Could not patch secondary for {sid}: {e}")

    # Also rebuild legacy parsed_races from non-session JSON files
    for path in OUTPUT_DIR.glob("*.json"):
        name = path.stem
        if name.endswith('_primary') or name.endswith('_secondary'):
            continue
        # Skip if already in parsed_races (shouldn't happen on fresh start)
        if name in parsed_races:
            continue
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            horses_list = data.get('horses', [])
            parsed_races[name] = {
                "id": name,
                "original_filename": "recovered",
                "pdf_path": "",
                "pdf_type": data.get('pdf_type', 'unknown'),
                "track_name": data.get('track_name', ''),
                "race_date": data.get('race_date', ''),
                "race_number": data.get('race_number', ''),
                "surface": data.get('surface', ''),
                "distance": data.get('distance', ''),
                "weather": data.get('weather', ''),
                "horses_count": len(horses_list),
                "total_races": sum(len(h.get('lines', [])) for h in horses_list),
                "tracks_count": 0,
                "surfaces_count": 0,
                "date_range": "N/A",
                "parsed_at": "",
                "analysis_date": "",
                "analysis_time": "",
                "processing_duration": "",
                "parser_used": data.get('parse_source', 'unknown'),
                "gpt_attempted": False,
                "fallback_to_traditional": False,
            }
        except Exception:
            pass

    logger.info(f"Startup recovery: {len(sessions)} session(s), {len(parsed_races)} legacy race(s)")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _build_session_meta(
    session_id: str, track: str, date: str,
    pdf_type: str, filename: str, pdf_path: str,
    output_file: str, horses_count: int, races_count: int,
    parser_used: str,
) -> Dict[str, Any]:
    """Create a new session dict for the sessions store."""
    now = pd.Timestamp.now()
    return {
        "session_id": session_id,
        "track": track,
        "date": date,
        "created_at": str(now),
        # Primary
        "primary_pdf_type": pdf_type,
        "primary_pdf_filename": filename,
        "primary_pdf_path": pdf_path,
        "primary_output_file": output_file,
        "primary_horses_count": horses_count,
        "primary_races_count": races_count,
        "primary_parser_used": parser_used,
        "primary_parsed_at": str(now),
        # Secondary (None until uploaded)
        "secondary_pdf_type": None,
        "secondary_pdf_filename": None,
        "secondary_pdf_path": None,
        "secondary_output_file": None,
        "secondary_horses_count": None,
        "secondary_parsed_at": None,
        # Flags
        "has_primary": True,
        "has_secondary": False,
        "merge_coverage": None,
        # Legacy compat fields (so /races still works)
        "id": session_id,
        "original_filename": filename,
        "pdf_type": pdf_type,
        "track_name": track,
        "race_date": date,
        "horses_count": horses_count,
        "total_races": races_count,
        "parser_used": parser_used,
        "analysis_date": now.strftime("%Y-%m-%d"),
        "analysis_time": now.strftime("%H:%M:%S"),
        "parsed_at": str(now),
        "processing_duration": "",
        "pdf_path": pdf_path,
        "gpt_attempted": False,
        "fallback_to_traditional": False,
    }


def _load_session_data(session_id: str, source: str = "merged") -> Dict[str, Any]:
    """Load JSON data for a session.

    ``source`` can be ``'primary'``, ``'secondary'``, or ``'merged'`` (default).
    Merged loads both and calls ``merge_primary_secondary`` if secondary exists.
    """
    primary_path = OUTPUT_DIR / f"{session_id}_primary.json"
    secondary_path = OUTPUT_DIR / f"{session_id}_secondary.json"

    if source == "secondary":
        if not secondary_path.exists():
            return None
        with open(secondary_path, 'r') as f:
            data = json.load(f)
        data.pop('_session_meta', None)
        return data

    if not primary_path.exists():
        return None
    with open(primary_path, 'r') as f:
        primary = json.load(f)
    primary.pop('_session_meta', None)

    if source == "primary" or not secondary_path.exists():
        return primary

    # merged
    with open(secondary_path, 'r') as f:
        secondary = json.load(f)
    secondary.pop('_session_meta', None)

    return merge_primary_secondary(primary, secondary)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Ragozin Sheets Parser API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint with DB status."""
    db_status = "unknown"
    db_backend = getattr(_db, "db_backend", "unknown") if _db else "not_initialized"
    try:
        if _db:
            _db.conn.execute("SELECT 1")
            db_status = "connected"
    except Exception as exc:
        db_status = f"error: {exc}"
    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "service": "ragozin-parser",
        "db_backend": db_backend,
        "db_status": db_status,
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), use_gpt: bool = False):
    """
    Upload and parse a racing PDF (auto-detects Ragozin vs BRISNET).
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        file_id = str(uuid.uuid4())
        pdf_path = UPLOAD_DIR / f"{file_id}.pdf"

        with open(pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Uploaded PDF: {file.filename} -> {pdf_path}")

        start_time = pd.Timestamp.now()
        gpt_attempted = False
        gpt_error_text = None
        output_dict = None

        # Auto-detect PDF type
        pdf_type = detect_pdf_type(str(pdf_path))
        logger.info(f"Detected PDF type: {pdf_type}")

        if pdf_type == 'brisnet':
            # --- BRISNET path ---
            logger.info("Running BRISNET parser")
            card = brisnet_parser.parse(str(pdf_path))
            output_dict = brisnet_to_pipeline(card)
            output_dict['parse_source'] = 'brisnet'
            parse_source = "brisnet"
            trad_data = None  # no CSV export for BRISNET
        else:
            # --- Ragozin path (existing logic) ---
            trad_dict = None
            gpt_dict = None

            logger.info("Running traditional parser")
            trad_data = parser.parse_ragozin_sheet(str(pdf_path))
            trad_dict = _trad_to_dict(trad_data)

            if use_gpt and gpt_parser:
                gpt_attempted = True
                try:
                    logger.info("Running GPT-enhanced parser")
                    gpt_result = gpt_parser.parse_ragozin_sheet(str(pdf_path), use_direct_pdf=True)
                    gpt_horses = getattr(gpt_result, 'horses', [])
                    if len(gpt_horses) == 0:
                        raise ValueError("GPT parser returned zero horses")
                    gpt_dict = _gpt_to_dict(gpt_result)
                except Exception as gpt_err:
                    gpt_error_text = str(gpt_err)
                    logger.warning(f"GPT parser failed: {gpt_err}")
                    gpt_dict = None
            elif use_gpt and not gpt_parser:
                logger.warning("GPT parser requested but unavailable")

            if trad_dict and gpt_dict:
                output_dict = merge_parsed_sessions(trad_dict, gpt_dict)
                parse_source = "both"
            elif gpt_dict:
                output_dict = gpt_dict
                output_dict['parse_source'] = 'gpt'
                parse_source = "gpt"
            else:
                output_dict = trad_dict
                output_dict['parse_source'] = 'traditional'
                if gpt_attempted:
                    parse_source = "fallback"
                    output_dict['parse_source'] = 'fallback'
                else:
                    parse_source = "traditional"

        end_time = pd.Timestamp.now()
        processing_duration = str(end_time - start_time).split('.')[0]

        # Compute stats from output
        horses_list = output_dict.get('horses', [])
        horses_count = len(horses_list)
        total_races = sum(len(h.get('lines', [])) for h in horses_list)
        unique_tracks = set()
        unique_surfaces = set()
        date_range = []
        for h in horses_list:
            for line in h.get('lines', []):
                if line.get('track'):
                    unique_tracks.add(line['track'])
                if line.get('surface'):
                    unique_surfaces.add(line['surface'])
                if line.get('race_date'):
                    date_range.append(line['race_date'])

        analysis_timestamp = pd.Timestamp.now()
        analysis_date = analysis_timestamp.strftime("%Y-%m-%d")
        analysis_time = analysis_timestamp.strftime("%H:%M:%S")

        # Write output JSON
        race_id = str(uuid.uuid4())
        json_path = OUTPUT_DIR / f"{race_id}.json"
        csv_path = OUTPUT_DIR / f"{race_id}.csv"

        with open(json_path, 'w') as f:
            json.dump(output_dict, f, indent=2, default=str)

        # CSV from the traditional parser (only for Ragozin)
        if pdf_type != 'brisnet' and trad_data:
            parser.export_to_csv(trad_data, str(csv_path))

        # Store session metadata
        parsed_races[race_id] = {
            "id": race_id,
            "original_filename": file.filename,
            "pdf_path": str(pdf_path),
            "pdf_type": pdf_type,
            "track_name": output_dict.get('track_name', ''),
            "race_date": output_dict.get('race_date', ''),
            "race_number": output_dict.get('race_number', ''),
            "surface": output_dict.get('surface', ''),
            "distance": output_dict.get('distance', ''),
            "weather": output_dict.get('weather', ''),
            "horses_count": horses_count,
            "total_races": total_races,
            "tracks_count": len(unique_tracks),
            "surfaces_count": len(unique_surfaces),
            "date_range": f"{min(date_range)} to {max(date_range)}" if date_range else "N/A",
            "parsed_at": str(analysis_timestamp),
            "analysis_date": analysis_date,
            "analysis_time": analysis_time,
            "processing_duration": processing_duration,
            "parser_used": parse_source,
            "gpt_attempted": gpt_attempted,
            "fallback_to_traditional": (parse_source == "fallback"),
        }

        return {
            "success": True,
            "race_id": race_id,
            "message": f"Successfully parsed {horses_count} horses",
            "analysis_date": analysis_date,
            "analysis_time": analysis_time,
            "processing_duration": processing_duration,
            "pdf_type": pdf_type,
            "race_info": {
                "track": output_dict.get('track_name', ''),
                "date": output_dict.get('race_date', ''),
                "race_number": output_dict.get('race_number', ''),
                "surface": output_dict.get('surface', ''),
                "distance": output_dict.get('distance', ''),
                "horses_count": horses_count,
                "total_races": total_races,
                "tracks_count": len(unique_tracks),
                "surfaces_count": len(unique_surfaces),
                "date_range": f"{min(date_range)} to {max(date_range)}" if date_range else "N/A",
                "parser_used": parse_source,
            },
            "parser_info": {
                "parser_used": parse_source,
                "pdf_type": pdf_type,
                "gpt_available": gpt_parser is not None,
                "gpt_attempted": gpt_attempted,
                "fallback_to_traditional": (parse_source == "fallback"),
                "gpt_error_text": gpt_error_text,
            },
            "files": {
                "json": str(json_path),
                "csv": str(csv_path) if csv_path.exists() else None,
            },
        }

    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")
# ---------------------------------------------------------------------------
# New dual-upload endpoints
# ---------------------------------------------------------------------------

@app.post("/upload_primary")
async def upload_primary(file: UploadFile = File(...), use_gpt: bool = False):
    """Upload a primary PDF (Ragozin sheets or BRISNET w/ QuickPlay).

    Creates a new session and returns session_id + summary.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    session_id = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{session_id}_primary.pdf"

    with open(pdf_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    pdf_type = detect_pdf_type(str(pdf_path))
    logger.info(f"upload_primary: {file.filename} detected as {pdf_type}")

    # Accept sheets/brisnet as primary, also accept unknown/brisnet_pp with a warning
    if pdf_type not in PRIMARY_TYPES and pdf_type not in SECONDARY_TYPES:
        # Still allow unknown through — parser may handle it
        logger.warning(f"PDF type '{pdf_type}' not in PRIMARY_TYPES, proceeding anyway")

    start_time = pd.Timestamp.now()
    output_dict = None
    parse_source = "unknown"

    if pdf_type in ('brisnet', 'brisnet_pp'):
        card = brisnet_parser.parse(str(pdf_path))
        output_dict = brisnet_to_pipeline(card)
        output_dict['parse_source'] = 'brisnet'
        parse_source = "brisnet"
    else:
        # Ragozin / sheets / unknown → traditional parser
        trad_data = parser.parse_ragozin_sheet(str(pdf_path))
        trad_dict = _trad_to_dict(trad_data)

        gpt_dict = None
        if use_gpt and gpt_parser:
            try:
                gpt_result = gpt_parser.parse_ragozin_sheet(str(pdf_path), use_direct_pdf=True)
                gpt_horses = getattr(gpt_result, 'horses', [])
                if len(gpt_horses) == 0:
                    raise ValueError("GPT parser returned zero horses")
                gpt_dict = _gpt_to_dict(gpt_result)
            except Exception as gpt_err:
                logger.warning(f"GPT parser failed: {gpt_err}")

        if trad_dict and gpt_dict:
            output_dict = merge_parsed_sessions(trad_dict, gpt_dict)
            parse_source = "both"
        elif gpt_dict:
            output_dict = gpt_dict
            output_dict['parse_source'] = 'gpt'
            parse_source = "gpt"
        else:
            output_dict = trad_dict
            output_dict['parse_source'] = 'traditional'
            parse_source = "traditional"

    end_time = pd.Timestamp.now()
    processing_duration = str(end_time - start_time).split('.')[0]

    horses_list = output_dict.get('horses', [])
    horses_count = len(horses_list)
    races_count = len({h.get('race_number', 0) for h in horses_list})

    track = output_dict.get('track_name', '')
    date = output_dict.get('race_date', '')

    # Embed session meta in the JSON for disk recovery
    output_file = str(OUTPUT_DIR / f"{session_id}_primary.json")
    session_meta = _build_session_meta(
        session_id=session_id, track=track, date=date,
        pdf_type=pdf_type, filename=file.filename,
        pdf_path=str(pdf_path), output_file=output_file,
        horses_count=horses_count, races_count=races_count,
        parser_used=parse_source,
    )
    session_meta['processing_duration'] = processing_duration

    output_dict['_session_meta'] = session_meta
    with open(output_file, 'w') as f:
        json.dump(output_dict, f, indent=2, default=str)

    sessions[session_id] = session_meta

    # --- Cumulative DB ingestion ---
    recon_result = None
    try:
        pdf_hash = hashlib.sha256(content).hexdigest()
        _db.record_upload(
            upload_id=session_id, source_type="brisnet" if parse_source == "brisnet" else "sheets",
            pdf_filename=file.filename, pdf_hash=pdf_hash,
            track=track, race_date=date, session_id=session_id,
            horses_count=horses_count,
        )
        if parse_source == "brisnet":
            _db.ingest_brisnet_card(session_id, output_dict, {})
        else:
            _db.ingest_sheets_card(session_id, output_dict)
        recon_result = _db.reconcile(session_id)
    except Exception as db_err:
        logger.warning(f"DB ingestion failed (non-fatal): {db_err}")

    resp = {
        "success": True,
        "session_id": session_id,
        "track": track,
        "date": date,
        "pdf_type": pdf_type,
        "horses_count": horses_count,
        "races_count": races_count,
        "parser_used": parse_source,
        "processing_duration": processing_duration,
    }
    if recon_result:
        resp["reconciliation"] = recon_result
        global_stats = _db.get_db_stats()
        resp["db_coverage_pct"] = global_stats["coverage_pct"]
        resp["db_reconciled_pairs"] = global_stats["reconciled_pairs"]
    return resp


@app.post("/upload_secondary")
async def upload_secondary(session_id: str, file: UploadFile = File(...)):
    """Attach a secondary (enrichment) PDF to an existing session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    if not session.get('has_primary'):
        raise HTTPException(status_code=400, detail="Session has no primary upload")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    pdf_path = UPLOAD_DIR / f"{session_id}_secondary.pdf"
    with open(pdf_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    pdf_type = detect_pdf_type(str(pdf_path))
    logger.info(f"upload_secondary: {file.filename} detected as {pdf_type}")

    # Parse with BRISNET parser (secondary is always BRISNET PPs)
    card = brisnet_parser.parse(str(pdf_path))
    output_dict = brisnet_to_pipeline(card)
    output_dict['parse_source'] = 'brisnet'

    horses_list = output_dict.get('horses', [])
    horses_count = len(horses_list)
    now = pd.Timestamp.now()

    output_file = str(OUTPUT_DIR / f"{session_id}_secondary.json")
    with open(output_file, 'w') as f:
        json.dump(output_dict, f, indent=2, default=str)

    # Compute merge coverage
    primary_data = _load_session_data(session_id, source="primary")
    merge_coverage = None
    if primary_data:
        merged = merge_primary_secondary(primary_data, output_dict)
        stats = merged.get('merge_stats', {})
        merge_coverage = stats.get('coverage')

    # Update session
    session.update({
        "secondary_pdf_type": pdf_type,
        "secondary_pdf_filename": file.filename,
        "secondary_pdf_path": str(pdf_path),
        "secondary_output_file": output_file,
        "secondary_horses_count": horses_count,
        "secondary_parsed_at": str(now),
        "has_secondary": True,
        "merge_coverage": merge_coverage,
    })

    # Persist updated _session_meta back to primary JSON so recovery sees it
    primary_json_path = OUTPUT_DIR / f"{session_id}_primary.json"
    if primary_json_path.exists():
        try:
            with open(primary_json_path, 'r') as f:
                primary_data_on_disk = json.load(f)
            primary_data_on_disk['_session_meta'] = dict(session)
            with open(primary_json_path, 'w') as f:
                json.dump(primary_data_on_disk, f, indent=2, default=str)
            logger.info(f"Persisted secondary metadata to {primary_json_path.name}")
        except Exception as e:
            logger.warning(f"Failed to persist secondary metadata: {e}")

    # --- Cumulative DB ingestion for secondary ---
    recon_result = None
    try:
        sec_upload_id = f"{session_id}_sec"
        pdf_hash = hashlib.sha256(content).hexdigest()
        _db.record_upload(
            upload_id=sec_upload_id, source_type="brisnet",
            pdf_filename=file.filename, pdf_hash=pdf_hash,
            track=session.get("track", ""), race_date=session.get("date", ""),
            session_id=session_id, horses_count=horses_count,
        )
        _db.ingest_brisnet_card(sec_upload_id, output_dict, {})
        recon_result = _db.reconcile(sec_upload_id)

        # Save morning line odds snapshots from BRISNET data
        odds_snapshots = []
        for h in horses_list:
            odds_dec = parse_odds_decimal(h.get('odds', ''))
            if odds_dec is not None:
                odds_snapshots.append({
                    'race_number': h.get('race_number', 0),
                    'post': h.get('post'),
                    'horse_name': h.get('horse_name', ''),
                    'odds_raw': h.get('odds', ''),
                    'odds_decimal': odds_dec,
                })
        if odds_snapshots:
            _db.save_odds_snapshots(
                session_id=session_id,
                track=session.get("track", ""),
                race_date=session.get("date", ""),
                snapshots=odds_snapshots,
                source='morning_line',
            )
            logger.info(f"Saved {len(odds_snapshots)} ML odds snapshots for {session_id}")
    except Exception as db_err:
        logger.warning(f"DB ingestion (secondary) failed (non-fatal): {db_err}")

    resp = {
        "success": True,
        "session_id": session_id,
        "secondary_pdf_type": pdf_type,
        "horses_count": horses_count,
        "merge_coverage": merge_coverage,
    }
    if recon_result:
        resp["reconciliation"] = recon_result
        global_stats = _db.get_db_stats()
        resp["db_coverage_pct"] = global_stats["coverage_pct"]
        resp["db_reconciled_pairs"] = global_stats["reconciled_pairs"]
    return resp


@app.get("/db/stats")
async def db_stats():
    """Return cumulative database statistics."""
    return _db.get_db_stats()


@app.get("/sessions/{session_id}/stats")
async def session_stats(session_id: str):
    """Return reconciliation stats scoped to a single session/upload."""
    result = _db.get_session_stats(session_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/results/stats")
async def results_stats(track: str = "", date_from: str = "", date_to: str = ""):
    """ROI and results statistics, optionally filtered by track/date range."""
    return _db.get_results_stats(track=track, date_from=date_from, date_to=date_to)


@app.post("/predictions/save")
async def save_predictions(payload: dict):
    """Save engine predictions for one race.

    Body: {session_id, track, race_date, race_number, projections: [...]}
    Each projection: {name, projection_type, bias_score, raw_score,
                      confidence, projected_low, projected_high, tags,
                      new_top_setup, bounce_risk, tossed}
    """
    sid = payload.get("session_id", "")
    track = payload.get("track", "")
    race_date = payload.get("race_date", "")
    race_number = int(payload.get("race_number", 0))
    projections = payload.get("projections", [])
    if not sid or not track or not race_date or not race_number or not projections:
        raise HTTPException(status_code=400, detail="Missing required fields")
    count = _db.save_predictions(sid, track, race_date, race_number, projections)
    return {"saved": count, "session_id": sid, "race_number": race_number}


@app.get("/predictions/roi")
async def prediction_roi(track: str = "", date: str = "", session_id: str = ""):
    """ROI from predictions joined with results."""
    return _db.get_prediction_roi(track=track, race_date=date, session_id=session_id)


@app.get("/predictions/roi-detailed")
async def prediction_roi_detailed(track: str = "", date: str = "", session_id: str = ""):
    """Detailed ROI by cycle, confidence bucket, odds bucket, surface, distance."""
    return _db.get_prediction_roi_detailed(track=track, race_date=date, session_id=session_id)


@app.get("/calibration/roi")
async def calibration_roi(track: str = "", min_n: int = 30):
    """ROI grouped by (track, surface, distance) with threshold recommendations."""
    return _db.get_calibration_data(min_n=min_n, track_filter=track)


@app.get("/calibration/detailed-roi")
async def detailed_roi(track: str = "", min_n: int = 5):
    """ROI by cycle, confidence tier, odds bucket, pick rank + leaks/suggestions."""
    return _db.get_detailed_roi(track_filter=track, min_n=min_n)


@app.get("/predictions/export-bets")
async def export_bets(track: str = "", date: str = "", session_id: str = ""):
    """All predictions + outcomes for audit/export."""
    return _db.get_all_bets(track=track, race_date=date, session_id=session_id)


@app.get("/predictions/vs-results")
async def predictions_vs_results(
    track: str = "", date: str = "", session_id: str = "",
):
    """Predictions joined with results for a card."""
    return _db.get_predictions_vs_results(
        track=track, race_date=date, session_id=session_id,
    )


@app.post("/results/upload")
async def upload_results(
    file: UploadFile = File(...), track: str = "", date: str = "",
    session_id: str = "",
):
    """Upload a results CSV or Equibase chart PDF and ingest into DB.

    If *session_id* is provided, results are scoped to that session and
    linking uses the session's uploads for matching.  Track/date are pulled
    from the session metadata when not explicitly provided.

    PDF uploads always return a preview (races, entries, sample_rows,
    confidence, missing_fields).  High-confidence PDFs are auto-ingested;
    low-confidence PDFs require user confirmation via ``/results/confirm-pdf``.
    """
    fname = (file.filename or "").lower()
    is_csv = fname.endswith('.csv')
    is_pdf = fname.endswith('.pdf')
    if not is_csv and not is_pdf:
        raise HTTPException(status_code=400, detail="Only CSV and PDF files are allowed")

    # Pull track/date from session when available
    if session_id and session_id in sessions:
        s = sessions[session_id]
        if not track:
            track = s.get("track", "")
        if not date:
            date = s.get("date", "")

    content = await file.read()
    import tempfile
    suffix = ".csv" if is_csv else ".pdf"
    tmp = Path(tempfile.mktemp(suffix=suffix))
    tmp.write_bytes(content)

    try:
        if is_csv:
            from ingest_results import ingest_csv
            result = ingest_csv(
                _db, str(tmp), track=track, race_date=date,
                session_id=session_id,
            )
            return {"success": True, **result}
        else:
            from ingest_results import ingest_pdf
            result = ingest_pdf(
                _db, str(tmp), track=track, race_date=date,
                session_id=session_id,
            )
            # Always includes preview data; adds ingested/needs_review flags
            if result.get("needs_review"):
                return result
            return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/results/confirm-pdf")
async def confirm_pdf_results(payload: dict):
    """Confirm and ingest user-reviewed PDF results.

    Body: {rows: [...], track: str, date: str, session_id: str,
           program_map: "program"|"post"}

    *program_map*: ``"program"`` uses the ``program`` column as the post
    identifier (extracts leading digits from "1A" etc.); ``"post"``
    (default) uses the ``post`` column as-is.
    """
    rows = payload.get("rows", [])
    track = payload.get("track", "")
    date = payload.get("date", "")
    session_id = payload.get("session_id", "")
    program_map = payload.get("program_map", "post")

    if not rows:
        raise HTTPException(status_code=400, detail="No rows provided")

    try:
        from ingest_results import ingest_rows
        result = ingest_rows(
            _db, rows, track=track, race_date=date,
            session_id=session_id, program_map=program_map,
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/inbox")
async def results_inbox():
    """Return result_races with unattached entries plus matching sessions."""
    unattached = _db.get_unattached_results()

    # For each unique (track, date), find matching sessions
    track_dates = set()
    for r in unattached:
        track_dates.add((r["track"], r["race_date"]))

    matching_sessions: Dict[str, Any] = {}
    for (track, race_date) in track_dates:
        key = f"{track}_{race_date}"
        matches = _db.get_sessions_for_track_date(track, race_date)
        # Also check in-memory sessions dict
        for sid, s in sessions.items():
            if s.get("track", "") == track and s.get("date", "") == race_date:
                if not any(m["session_id"] == sid for m in matches):
                    matches.append({
                        "session_id": sid,
                        "track_name": s.get("track", ""),
                        "race_date": s.get("date", ""),
                        "created_at": s.get("created_at", ""),
                        "parser_used": s.get("primary_pdf_type", ""),
                        "horses_count": s.get("primary_horses_count", 0),
                    })
        matching_sessions[key] = matches

    return {"unattached_races": unattached, "matching_sessions": matching_sessions}


@app.post("/results/attach")
async def attach_results(payload: dict):
    """Attach result_entries to a session.

    Body: {track, race_date, session_id}
    """
    track = payload.get("track", "")
    race_date = payload.get("race_date", "")
    session_id = payload.get("session_id", "")
    if not track or not race_date or not session_id:
        raise HTTPException(status_code=400, detail="track, race_date, session_id required")
    result = _db.attach_results_to_session(track, race_date, session_id)
    return {"success": True, **result}


@app.post("/db/alias")
async def add_alias(payload: dict):
    """Add a horse name alias. Body: {canonical: str, alias: str}"""
    canonical = payload.get("canonical", "")
    alias = payload.get("alias", "")
    if not canonical or not alias:
        raise HTTPException(status_code=400, detail="Both 'canonical' and 'alias' are required")
    _db.add_alias(canonical, alias)
    # Re-run reconciliation to pick up new alias
    result = _db.reconcile()
    return {"status": "ok", "canonical": canonical, "alias": alias, "reconciliation": result}


@app.get("/db/aliases")
async def list_aliases():
    """List all horse name aliases."""
    return _db.list_aliases()


@app.get("/db/unmatched")
async def unmatched_brisnet(track: str = "", date: str = ""):
    """Unmatched brisnet horses with fuzzy match suggestions."""
    return _db.get_unmatched_with_suggestions(track=track, race_date=date)


@app.get("/sessions")
async def list_sessions():
    """List all dual-upload sessions."""
    items = []
    for sid, s in sessions.items():
        items.append({
            "session_id": sid,
            "track": s.get("track", ""),
            "date": s.get("date", ""),
            "has_primary": s.get("has_primary", False),
            "has_secondary": s.get("has_secondary", False),
            "primary_pdf_type": s.get("primary_pdf_type"),
            "secondary_pdf_type": s.get("secondary_pdf_type"),
            "primary_pdf_filename": s.get("primary_pdf_filename"),
            "secondary_pdf_filename": s.get("secondary_pdf_filename"),
            "primary_horses_count": s.get("primary_horses_count"),
            "secondary_horses_count": s.get("secondary_horses_count"),
            "merge_coverage": s.get("merge_coverage"),
            "created_at": s.get("created_at"),
        })
    return {"sessions": items, "total_count": len(items)}


@app.get("/sessions/{session_id}/summary")
async def session_summary(session_id: str):
    """Per-source counts for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]

    result = {
        "session_id": session_id,
        "primary_races": session.get("primary_races_count", 0),
        "primary_horses": session.get("primary_horses_count", 0),
        "secondary_horses": session.get("secondary_horses_count"),
    }

    if session.get("has_secondary"):
        merged = _load_session_data(session_id, source="merged")
        if merged:
            stats = merged.get('merge_stats', {})
            result["matched"] = stats.get("matched", 0)
            result["unmatched"] = stats.get("unmatched", 0)
            result["unmatched_names"] = stats.get("unmatched_names", [])
            result["merge_coverage"] = stats.get("coverage")

    return result


@app.get("/sessions/{session_id}/races")
async def session_races(session_id: str, source: str = "merged"):
    """Get race data for a session.

    Query param ``source``: ``primary``, ``secondary``, or ``merged`` (default).
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if source not in ("primary", "secondary", "merged"):
        raise HTTPException(status_code=400, detail="source must be primary, secondary, or merged")

    data = _load_session_data(session_id, source=source)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No {source} data for this session")

    return {"session_id": session_id, "source": source, "race_data": data}


# ---------------------------------------------------------------------------
# Legacy endpoints (wrappers)
# ---------------------------------------------------------------------------

@app.get("/races")
async def list_races():
    """
    List all parsed races (legacy + sessions combined).
    """
    all_races = list(parsed_races.values()) + list(sessions.values())
    return {
        "races": all_races,
        "total_count": len(all_races)
    }

@app.get("/races/{race_id}")
async def get_race(race_id: str):
    """
    Get detailed information about a specific race.
    Checks sessions first (returns merged if secondary exists), then legacy.
    """
    # Check sessions first
    if race_id in sessions:
        data = _load_session_data(race_id, source="merged")
        if data:
            return {"race_info": sessions[race_id], "race_data": data}

    if race_id not in parsed_races:
        raise HTTPException(status_code=404, detail="Race not found")

    # Load the full race data from JSON file
    json_path = OUTPUT_DIR / f"{race_id}.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            race_data = json.load(f)

        return {
            "race_info": parsed_races[race_id],
            "race_data": race_data
        }
    else:
        raise HTTPException(status_code=404, detail="Race data file not found")

@app.get("/races/{race_id}/horses")
async def get_race_horses(race_id: str):
    """
    Get horses for a specific race.
    Checks sessions first (returns merged if secondary exists), then legacy.
    """
    # Check sessions first
    if race_id in sessions:
        data = _load_session_data(race_id, source="merged")
        if data:
            return {"race_id": race_id, "horses": data.get("horses", [])}

    if race_id not in parsed_races:
        raise HTTPException(status_code=404, detail="Race not found")

    # Load the full race data from JSON file
    json_path = OUTPUT_DIR / f"{race_id}.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            race_data = json.load(f)

        return {
            "race_id": race_id,
            "horses": race_data.get("horses", [])
        }
    else:
        raise HTTPException(status_code=404, detail="Race data file not found")

@app.get("/races/{race_id}/download/{format}")
async def download_race_data(race_id: str, format: str):
    """
    Download race data in specified format (json or csv)
    """
    if race_id not in parsed_races:
        raise HTTPException(status_code=404, detail="Race not found")
    
    if format.lower() not in ["json", "csv"]:
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
    
    file_path = OUTPUT_DIR / f"{race_id}.{format.lower()}"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{format.upper()} file not found")
    
    return {"file_path": str(file_path)}

@app.delete("/races/{race_id}")
async def delete_race(race_id: str):
    """
    Delete a parsed race/session and its associated files.
    Handles both new sessions and legacy parsed_races.
    """
    found = race_id in sessions or race_id in parsed_races
    if not found:
        raise HTTPException(status_code=404, detail="Race not found")

    try:
        # Clean up session files (primary + secondary)
        if race_id in sessions:
            for suffix in ('_primary.pdf', '_secondary.pdf'):
                p = UPLOAD_DIR / f"{race_id}{suffix}"
                if p.exists():
                    p.unlink()
            for suffix in ('_primary.json', '_secondary.json'):
                p = OUTPUT_DIR / f"{race_id}{suffix}"
                if p.exists():
                    p.unlink()
            del sessions[race_id]

        # Clean up legacy files
        if race_id in parsed_races:
            pdf_path = Path(parsed_races[race_id].get("pdf_path", ""))
            json_path = OUTPUT_DIR / f"{race_id}.json"
            csv_path = OUTPUT_DIR / f"{race_id}.csv"

            if pdf_path.exists():
                pdf_path.unlink()
            if json_path.exists():
                json_path.unlink()
            if csv_path.exists():
                csv_path.unlink()

            del parsed_races[race_id]

        return {"success": True, "message": "Race deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting race: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting race: {str(e)}")

@app.get("/parser-status")
async def get_parser_status():
    """
    Get parser status and capabilities
    """
    return {
        "traditional_parser": True,
        "gpt_parser_available": gpt_parser is not None,
        "openai_api_key_set": os.getenv("OPENAI_API_KEY") is not None,
        "symbol_sheet_loaded": gpt_parser.ragozin_context != "" if gpt_parser else False
    }

@app.get("/stats")
async def get_stats():
    """
    Get parsing statistics for the new data structure
    """
    total_sessions = len(parsed_races)
    total_horses = sum(race["horses_count"] for race in parsed_races.values())
    total_individual_races = sum(race.get("total_races", 0) for race in parsed_races.values())
    
    # Parser usage breakdown
    parser_counts = {}
    for race in parsed_races.values():
        parser_used = race.get("parser_used", "unknown")
        parser_counts[parser_used] = parser_counts.get(parser_used, 0) + 1
    
    # Surface breakdown (from individual races)
    surface_counts = {}
    track_counts = {}
    
    # Load detailed data to get accurate surface and track counts
    for race_id, race_info in parsed_races.items():
        json_path = OUTPUT_DIR / f"{race_id}.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    race_data = json.load(f)
                
                for horse in race_data.get("horses", []):
                    # Each horse entry represents one race entry
                    surface = horse.get("track_surface", "unknown")
                    track = race_data.get("track_name", "unknown")
                    surface_counts[surface] = surface_counts.get(surface, 0) + 1
                    track_counts[track] = track_counts.get(track, 0) + 1
            except Exception as e:
                logger.warning(f"Error loading race data for stats: {e}")
    
    return {
        "total_sessions": total_sessions,
        "total_horses": total_horses,
        "total_individual_races": total_individual_races,
        "average_horses_per_session": total_horses / total_sessions if total_sessions > 0 else 0,
        "average_races_per_horse": total_individual_races / total_horses if total_horses > 0 else 0,
        "parser_usage": parser_counts,
        "surface_breakdown": surface_counts,
        "track_breakdown": track_counts
    }

# ---------------------------------------------------------------------------
# Parser diagnostics
# ---------------------------------------------------------------------------

@app.get("/parser/diagnose/{session_id}/{race_number}")
async def parser_diagnose(session_id: str, race_number: int):
    """Dump raw parser diagnostics for a single race.

    Returns per-page classified lines showing how each line was interpreted
    by the Ragozin parser (figure, data, concat_figure_data, noise, etc.).
    """
    pdf_path = Path("uploads") / f"{session_id}_primary.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found for session")
    parser = RagozinParser()
    return parser.diagnose_race(str(pdf_path), race_number)


# ---------------------------------------------------------------------------
# Odds endpoints
# ---------------------------------------------------------------------------

@app.get("/odds/snapshots/{session_id}")
async def get_odds_snapshots_endpoint(session_id: str):
    """Return ML odds snapshots for a session's track/date."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    track = s.get("track", "")
    race_date = s.get("date", "")
    rows = _db.get_odds_snapshots_full(track, race_date, source="morning_line")
    return {"session_id": session_id, "source": "morning_line", "snapshots": rows}


# ---------------------------------------------------------------------------
# Bet Builder endpoints
# ---------------------------------------------------------------------------

@app.post("/bets/build")
async def build_bets(payload: dict):
    """Generate a bet plan from saved predictions.

    Body: {session_id, track, race_date,
           bankroll?, risk_profile?, max_risk_per_race_pct?,
           max_risk_per_day_pct?, min_confidence?, min_odds_a?,
           min_odds_b?, paper_mode?, save?}

    Returns the DayPlan dict (and plan_id if save=true).
    """
    sid = payload.get("session_id", "")
    track = payload.get("track", "")
    race_date = payload.get("race_date", "")
    if not sid or not track or not race_date:
        raise HTTPException(status_code=400, detail="session_id, track, race_date required")

    # Build settings from payload
    settings = BetSettings(
        bankroll=float(payload.get("bankroll", 1000)),
        risk_profile=payload.get("risk_profile", "standard"),
        max_risk_per_race_pct=float(payload.get("max_risk_per_race_pct", 1.5)),
        max_risk_per_day_pct=float(payload.get("max_risk_per_day_pct", 6.0)),
        min_confidence=float(payload.get("min_confidence", 0)),
        min_odds_a=float(payload.get("min_odds_a", 2.0)),
        min_odds_b=float(payload.get("min_odds_b", 4.0)),
        paper_mode=bool(payload.get("paper_mode", True)),
        allow_missing_odds=bool(payload.get("allow_missing_odds", False)),
    )

    # Load predictions for this card
    preds = _db.get_predictions_vs_results(track=track, race_date=race_date, session_id=sid)
    if not preds:
        raise HTTPException(status_code=404, detail="No predictions found for this card")

    # Inject morning line odds (with raw string) where result odds are missing
    ml_snaps = _db.get_odds_snapshots_full(track=track, race_date=race_date, source='morning_line')
    ml_by_name: Dict[tuple, dict] = {}
    for snap in ml_snaps:
        key = (snap["race_number"], snap["normalized_name"])
        ml_by_name[key] = snap
    for p in preds:
        if p.get('odds') is None:
            key = (p.get('race_number', 0), _db._normalize_name(p.get('horse_name', '')))
            snap = ml_by_name.get(key)
            if snap and snap.get('odds_decimal') is not None:
                p['odds'] = snap['odds_decimal']
                p['odds_raw'] = snap.get('odds_raw', '')
                p['odds_source'] = 'morning_line'
            else:
                p['odds_source'] = 'missing'
        else:
            p['odds_source'] = 'result'

    # Group by race_number
    race_projections: Dict[int, list] = {}
    for p in preds:
        rn = p.get("race_number", 0)
        if rn not in race_projections:
            race_projections[rn] = []
        race_projections[rn].append(p)

    # Compute figure quality per race for guardrails
    race_quality: Dict[int, float] = {}
    for rn, projs in race_projections.items():
        non_tossed = [p for p in projs if not p.get("tossed", False)]
        with_figs = sum(1 for p in non_tossed if p.get("bias_score", 0) != 0)
        race_quality[rn] = with_figs / len(non_tossed) if non_tossed else 0.0

    plan = build_day_plan(race_projections, settings, race_quality=race_quality)
    plan_dict = day_plan_to_dict(plan)

    result = {"plan": plan_dict}

    # Optionally persist
    if payload.get("save", True):
        from dataclasses import asdict as _asdict
        plan_id = _db.save_bet_plan(
            session_id=sid, track=track, race_date=race_date,
            settings_dict=_asdict(settings), plan_dict=plan_dict,
            total_risk=plan.total_risk, paper_mode=settings.paper_mode,
            engine_version=_ENGINE_VERSION,
        )
        result["plan_id"] = plan_id

    return result


@app.post("/bets/daily-wins")
async def build_daily_wins_endpoint(payload: dict):
    """Generate cross-track daily WIN bet plan.

    Body: {race_date, bankroll?, risk_profile?, max_risk_per_day_pct?,
           min_confidence?, min_odds_a?, min_overlay?, paper_mode?, max_bets?, save?}
    """
    race_date = payload.get("race_date", "")
    if not race_date:
        raise HTTPException(status_code=400, detail="race_date required")

    settings = BetSettings(
        bankroll=float(payload.get("bankroll", 1000)),
        risk_profile=payload.get("risk_profile", "standard"),
        max_risk_per_day_pct=float(payload.get("max_risk_per_day_pct", 6.0)),
        min_confidence=float(payload.get("min_confidence", 0.65)),
        min_odds_a=float(payload.get("min_odds_a", 2.0)),
        paper_mode=bool(payload.get("paper_mode", True)),
        allow_missing_odds=bool(payload.get("allow_missing_odds", False)),
        min_overlay=float(payload.get("min_overlay", 1.10)),
    )
    max_bets = int(payload.get("max_bets", 10))

    # Load all predictions and odds for the date
    predictions_by_track = _db.get_all_predictions_for_date(race_date)
    if not predictions_by_track:
        raise HTTPException(status_code=404, detail="No predictions found for this date")

    odds_by_key = _db.get_all_odds_for_date(race_date)

    # Build daily WIN candidates
    candidates = build_daily_wins(
        predictions_by_track, odds_by_key, settings, max_bets=max_bets,
    )

    # Serialize candidates
    from dataclasses import asdict as _asdict_local
    candidates_dicts = [_asdict_local(c) for c in candidates]
    total_risk = sum(c.stake for c in candidates)

    result: Dict[str, Any] = {
        "candidates": candidates_dicts,
        "total_risk": total_risk,
        "tracks": list(set(c.track for c in candidates)),
        "bet_count": len(candidates),
    }

    # Optionally persist
    if payload.get("save", True) and candidates:
        plan_dict = {
            "candidates": candidates_dicts,
            "total_risk": total_risk,
            "settings": _asdict_local(settings),
        }
        plan_id = _db.save_bet_plan(
            session_id=f"daily_{race_date}",
            track="ALL",
            race_date=race_date,
            settings_dict=_asdict_local(settings),
            plan_dict=plan_dict,
            total_risk=total_risk,
            paper_mode=settings.paper_mode,
            engine_version=_ENGINE_VERSION,
            plan_type="daily",
        )
        result["plan_id"] = plan_id

    return result


@app.post("/bets/multi-race")
async def build_multi_race(payload: dict):
    """Build Pick 3 or Pick 6 ticket.

    Body: {session_id, track, race_date, start_race, bet_type,
           budget, a_count?, b_count?, c_count?, c_leg_override?, save?}
    """
    sid = payload.get("session_id", "")
    track = payload.get("track", "")
    race_date = payload.get("race_date", "")
    if not sid or not track or not race_date:
        raise HTTPException(status_code=400, detail="session_id, track, race_date required")

    start_race = int(payload.get("start_race", 1))
    bet_type = payload.get("bet_type", "PICK3").upper()
    if bet_type not in ("PICK3", "PICK6"):
        raise HTTPException(status_code=400, detail="bet_type must be PICK3 or PICK6")

    budget = float(payload.get("budget", 48))
    strategy = MultiRaceStrategy(
        a_count=int(payload.get("a_count", 1)),
        b_count=int(payload.get("b_count", 3)),
        c_count=int(payload.get("c_count", 5)),
    )
    c_leg_override = bool(payload.get("c_leg_override", False))
    settings = BetSettings(
        bankroll=float(payload.get("bankroll", 1000)),
        min_confidence=float(payload.get("min_confidence", 0)),
    )

    # Load predictions
    preds = _db.get_predictions_vs_results(track=track, race_date=race_date, session_id=sid)
    if not preds:
        raise HTTPException(status_code=404, detail="No predictions found for this card")

    # Inject ML odds
    ml_snaps = _db.get_odds_snapshots_full(track=track, race_date=race_date, source='morning_line')
    ml_by_name: Dict[tuple, dict] = {}
    for snap in ml_snaps:
        key = (snap["race_number"], snap["normalized_name"])
        ml_by_name[key] = snap
    for p in preds:
        if p.get('odds') is None:
            key = (p.get('race_number', 0), _db._normalize_name(p.get('horse_name', '')))
            snap = ml_by_name.get(key)
            if snap and snap.get('odds_decimal') is not None:
                p['odds'] = snap['odds_decimal']
                p['odds_raw'] = snap.get('odds_raw', '')

    # Group by race_number
    race_projections: Dict[int, list] = {}
    for p in preds:
        rn = p.get("race_number", 0)
        race_projections.setdefault(rn, []).append(p)

    # Compute figure quality per race
    race_quality: Dict[int, float] = {}
    for rn, projs in race_projections.items():
        non_tossed = [p for p in projs if not p.get("tossed", False)]
        with_figs = sum(1 for p in non_tossed if p.get("bias_score", 0) != 0)
        race_quality[rn] = with_figs / len(non_tossed) if non_tossed else 0.0

    plan = build_multi_race_plan(
        race_projections, start_race, bet_type, budget, strategy, settings,
        race_quality=race_quality, c_leg_override=c_leg_override,
    )
    plan_dict = multi_race_plan_to_dict(plan)

    result: Dict[str, Any] = {"plan": plan_dict}

    # Optionally persist
    if payload.get("save", False):
        plan_id = _db.save_bet_plan(
            session_id=sid, track=track, race_date=race_date,
            settings_dict=plan.settings, plan_dict=plan_dict,
            total_risk=plan.cost, paper_mode=True,
            engine_version=_ENGINE_VERSION,
            plan_type="multi_race",
        )
        result["plan_id"] = plan_id

    return result


@app.post("/bets/daily-double")
async def build_daily_double(payload: dict):
    """Build a Daily Double ticket for two consecutive races.

    Body: {session_id, track, race_date, start_race,
           budget, a_count?, b_count?, c_count?, save?}
    """
    sid = payload.get("session_id", "")
    track = payload.get("track", "")
    race_date = payload.get("race_date", "")
    if not sid or not track or not race_date:
        raise HTTPException(status_code=400, detail="session_id, track, race_date required")

    start_race = int(payload.get("start_race", 1))
    budget = float(payload.get("budget", 24))
    strategy = MultiRaceStrategy(
        a_count=int(payload.get("a_count", 1)),
        b_count=int(payload.get("b_count", 3)),
        c_count=int(payload.get("c_count", 5)),
    )
    settings = BetSettings(
        bankroll=float(payload.get("bankroll", 1000)),
        min_confidence=float(payload.get("min_confidence", 0)),
    )

    # Load predictions
    preds = _db.get_predictions_vs_results(track=track, race_date=race_date, session_id=sid)
    if not preds:
        raise HTTPException(status_code=404, detail="No predictions found for this card")

    # Inject ML odds
    ml_snaps = _db.get_odds_snapshots_full(track=track, race_date=race_date, source='morning_line')
    ml_by_name: Dict[tuple, dict] = {}
    for snap in ml_snaps:
        key = (snap["race_number"], snap["normalized_name"])
        ml_by_name[key] = snap
    for p in preds:
        if p.get('odds') is None:
            key = (p.get('race_number', 0), _db._normalize_name(p.get('horse_name', '')))
            snap = ml_by_name.get(key)
            if snap and snap.get('odds_decimal') is not None:
                p['odds'] = snap['odds_decimal']
                p['odds_raw'] = snap.get('odds_raw', '')

    # Group by race_number
    race_projections: Dict[int, list] = {}
    for p in preds:
        rn = p.get("race_number", 0)
        race_projections.setdefault(rn, []).append(p)

    # Compute figure quality per race
    race_quality: Dict[int, float] = {}
    for rn, projs in race_projections.items():
        non_tossed = [p for p in projs if not p.get("tossed", False)]
        with_figs = sum(1 for p in non_tossed if p.get("bias_score", 0) != 0)
        race_quality[rn] = with_figs / len(non_tossed) if non_tossed else 0.0

    plan = build_daily_double_plan(
        race_projections, start_race, budget, settings, strategy,
        race_quality=race_quality,
    )
    plan_dict = daily_double_plan_to_dict(plan)

    result: Dict[str, Any] = {"plan": plan_dict}

    if payload.get("save", False):
        plan_id = _db.save_bet_plan(
            session_id=sid, track=track, race_date=race_date,
            settings_dict=plan.settings, plan_dict=plan_dict,
            total_risk=plan.total_cost, paper_mode=True,
            engine_version=_ENGINE_VERSION,
            plan_type="daily_double",
        )
        result["plan_id"] = plan_id

    return result


@app.post("/bets/dual-mode")
async def build_dual_mode(payload: dict):
    """Generate dual-mode (Profit + Score) betting plan.

    Body: {session_id, track, race_date, mode?,
           bankroll?, risk_profile?, score_budget_pct?,
           profit_min_overlay?, score_min_odds?, score_min_overlay?,
           mandatory_payout?, save?}
    """
    sid = payload.get("session_id", "")
    track = payload.get("track", "")
    race_date = payload.get("race_date", "")
    if not sid or not track or not race_date:
        raise HTTPException(status_code=400, detail="session_id, track, race_date required")

    mode = payload.get("mode", "both")
    if mode not in ("profit", "score", "both"):
        raise HTTPException(status_code=400, detail="mode must be profit, score, or both")

    settings = DualModeSettings(
        bankroll=float(payload.get("bankroll", 1000)),
        risk_profile=payload.get("risk_profile", "standard"),
        score_budget_pct=float(payload.get("score_budget_pct", 0.20)),
        profit_min_overlay=float(payload.get("profit_min_overlay", 1.25)),
        profit_min_odds_a=float(payload.get("profit_min_odds_a", 2.0)),
        profit_min_odds_b=float(payload.get("profit_min_odds_b", 4.0)),
        figure_quality_threshold=float(payload.get("figure_quality_threshold", 0.80)),
        score_min_odds=float(payload.get("score_min_odds", 8.0)),
        score_min_overlay=float(payload.get("score_min_overlay", 1.60)),
        score_max_c_per_leg=int(payload.get("score_max_c_per_leg", 2)),
        score_require_singles=int(payload.get("score_require_singles", 2)),
        mandatory_payout=bool(payload.get("mandatory_payout", False)),
        paper_mode=bool(payload.get("paper_mode", True)),
    )

    # Load predictions
    preds = _db.get_predictions_vs_results(track=track, race_date=race_date, session_id=sid)
    if not preds:
        raise HTTPException(status_code=404, detail="No predictions found for this card")

    # Inject ML odds
    ml_snaps = _db.get_odds_snapshots_full(track=track, race_date=race_date, source='morning_line')
    ml_by_name: Dict[tuple, dict] = {}
    for snap in ml_snaps:
        key = (snap["race_number"], snap["normalized_name"])
        ml_by_name[key] = snap
    for p in preds:
        if p.get('odds') is None:
            key = (p.get('race_number', 0), _db._normalize_name(p.get('horse_name', '')))
            snap = ml_by_name.get(key)
            if snap and snap.get('odds_decimal') is not None:
                p['odds'] = snap['odds_decimal']
                p['odds_raw'] = snap.get('odds_raw', '')

    # Group by race_number
    race_projections: Dict[int, list] = {}
    for p in preds:
        rn = p.get("race_number", 0)
        race_projections.setdefault(rn, []).append(p)

    # Compute figure quality per race
    race_quality: Dict[int, float] = {}
    for rn, projs in race_projections.items():
        non_tossed = [p for p in projs if not p.get("tossed", False)]
        with_figs = sum(1 for p in non_tossed if p.get("bias_score", 0) != 0)
        race_quality[rn] = with_figs / len(non_tossed) if non_tossed else 0.0

    plan = build_dual_mode_day_plan(
        race_projections, settings, mode=mode,
        race_quality=race_quality, track=track,
    )
    plan_dict = dual_mode_plan_to_dict(plan)

    result: Dict[str, Any] = {"plan": plan_dict}

    if payload.get("save", False):
        plan_id = _db.save_bet_plan(
            session_id=sid, track=track, race_date=race_date,
            settings_dict=plan.settings, plan_dict=plan_dict,
            total_risk=plan.total_risk, paper_mode=settings.paper_mode,
            engine_version=_ENGINE_VERSION,
            plan_type="dual_mode",
        )
        result["plan_id"] = plan_id

    return result


@app.get("/bets/plans")
async def list_bet_plans(session_id: str = "", track: str = "", date: str = ""):
    """List stored bet plans."""
    return _db.load_bet_plans(session_id=session_id, track=track, race_date=date)


@app.get("/bets/evaluate/{plan_id}")
async def evaluate_bet_plan(plan_id: int):
    """Evaluate a bet plan against actual results."""
    result = _db.evaluate_bet_plan_roi(plan_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/bets/commander-save")
async def save_commander_plan(payload: dict):
    """Persist a finalised Bet Commander slip.

    Body: {session_id, track, race_date, slip_entries, total_cost, bankroll}
    """
    sid = payload.get("session_id", "")
    track = payload.get("track", "")
    race_date = payload.get("race_date", "")
    if not sid or not track or not race_date:
        raise HTTPException(status_code=400, detail="session_id, track, race_date required")

    slip_entries = payload.get("slip_entries", [])
    if not slip_entries:
        raise HTTPException(status_code=400, detail="slip_entries required")

    total_cost = float(payload.get("total_cost", 0))
    bankroll = float(payload.get("bankroll", 0))

    plan_dict = {
        "slip_entries": slip_entries,
        "total_cost": total_cost,
        "bankroll": bankroll,
        "bet_count": len(slip_entries),
    }
    settings_dict = {
        "bankroll": bankroll,
        "source": "bet_commander",
    }

    plan_id = _db.save_bet_plan(
        session_id=sid, track=track, race_date=race_date,
        settings_dict=settings_dict, plan_dict=plan_dict,
        total_risk=total_cost, paper_mode=True,
        engine_version=_ENGINE_VERSION,
        plan_type="commander",
    )
    return {"plan_id": plan_id, "total_cost": total_cost}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
