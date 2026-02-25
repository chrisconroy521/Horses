from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from brisnet_parser import BrisnetParser, to_pipeline_json as brisnet_to_pipeline
from persistence import Persistence
import hashlib
import dotenv
dotenv.load_dotenv()

# Cumulative horse database
_db = Persistence(Path("horses.db"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ragozin Sheets Parser API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """Rebuild sessions dict from *_primary.json files on disk."""
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
    """Health check endpoint"""
    return {"status": "healthy", "service": "ragozin-parser"}

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
    """Upload a results CSV and ingest into DB.

    If *session_id* is provided, results are scoped to that session and
    linking uses the session's uploads for matching.  Track/date are pulled
    from the session metadata when not explicitly provided.
    """
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    # Pull track/date from session when available
    if session_id and session_id in sessions:
        s = sessions[session_id]
        if not track:
            track = s.get("track", "")
        if not date:
            date = s.get("date", "")

    content = await file.read()
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".csv"))
    tmp.write_bytes(content)

    try:
        from ingest_results import ingest_csv
        result = ingest_csv(
            _db, str(tmp), track=track, race_date=date,
            session_id=session_id,
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp.unlink(missing_ok=True)


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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
