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

from dataclasses import asdict
from ragozin_parser import RagozinParser, RaceData
from gpt_parser_alternative import GPTRagozinParserAlternative
from merge import merge_parsed_sessions
import dotenv
dotenv.load_dotenv()

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
gpt_parser = None

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


def _trad_to_dict(race_data: RaceData) -> Dict[str, Any]:
    """Convert traditional RaceData to the JSON-output dict schema."""
    horses_dict = []
    for horse in race_data.horses:
        race_line = {
            'fig': str(horse.figure) if horse.figure is not None else '',
            'flags': horse.trouble_indicators if horse.trouble_indicators else [],
            'track': race_data.track_name,
            'month': horse.race_date.split('/')[0] if horse.race_date and '/' in horse.race_date else '',
            'surface': horse.track_surface,
            'race_type': 'Unknown',
            'race_date': horse.race_date,
            'notes': (
                f"Finish: {horse.finish_position}, Odds: {horse.odds}, "
                f"Trainer: {horse.trainer}, Jockey: {horse.jockey}"
                if any([horse.finish_position, horse.odds, horse.trainer, horse.jockey])
                else ''
            ),
            'race_analysis': (
                f"Speed: {horse.speed_rating}, Pace: {horse.pace_rating}, Class: {horse.class_rating}"
                if any([horse.speed_rating, horse.pace_rating, horse.class_rating])
                else ''
            ),
        }
        horse_dict = {
            'horse_name': horse.horse_name,
            'sex': 'Unknown',
            'age': 0,
            'breeder_owner': 'Unknown',
            'foal_date': 'Unknown',
            'reg_code': 'Unknown',
            'races': 1,
            'top_fig': str(horse.figure) if horse.figure is not None else '',
            'horse_analysis': f"Ragozin Figure: {horse.figure}, Surface: {horse.track_surface}, Distance: {horse.distance}",
            'performance_trend': (
                f"Weather: {horse.weather_conditions}, Weight: {horse.weight} lbs"
                if horse.weather_conditions or horse.weight
                else 'No trend data available'
            ),
            'lines': [race_line],
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
    Upload and parse a Ragozin PDF sheet
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
        trad_dict = None
        gpt_dict = None

        # 1. Always run traditional parser
        logger.info("Running traditional parser")
        trad_data = parser.parse_ragozin_sheet(str(pdf_path))
        trad_dict = _trad_to_dict(trad_data)

        # 2. Optionally run GPT parser
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

        # 3. Merge or pick single source
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

        # 4. Compute stats from merged output
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

        # 5. Write output JSON
        race_id = str(uuid.uuid4())
        json_path = OUTPUT_DIR / f"{race_id}.json"
        csv_path = OUTPUT_DIR / f"{race_id}.csv"

        with open(json_path, 'w') as f:
            json.dump(output_dict, f, indent=2, default=str)

        # CSV from the traditional parser (always available)
        parser.export_to_csv(trad_data, str(csv_path))

        # 6. Store session metadata
        parsed_races[race_id] = {
            "id": race_id,
            "original_filename": file.filename,
            "pdf_path": str(pdf_path),
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
                "gpt_available": gpt_parser is not None,
                "gpt_attempted": gpt_attempted,
                "fallback_to_traditional": (parse_source == "fallback"),
                "gpt_error_text": gpt_error_text,
            },
            "files": {
                "json": str(json_path),
                "csv": str(csv_path),
            },
        }

    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")
@app.get("/races")
async def list_races():
    """
    List all parsed races
    """
    return {
        "races": list(parsed_races.values()),
        "total_count": len(parsed_races)
    }

@app.get("/races/{race_id}")
async def get_race(race_id: str):
    """
    Get detailed information about a specific race
    """
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
    Get horses for a specific race
    """
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
    Delete a parsed race and its associated files
    """
    if race_id not in parsed_races:
        raise HTTPException(status_code=404, detail="Race not found")
    
    try:
        # Remove files
        pdf_path = Path(parsed_races[race_id]["pdf_path"])
        json_path = OUTPUT_DIR / f"{race_id}.json"
        csv_path = OUTPUT_DIR / f"{race_id}.csv"
        
        if pdf_path.exists():
            pdf_path.unlink()
        if json_path.exists():
            json_path.unlink()
        if csv_path.exists():
            csv_path.unlink()
        
        # Remove from memory
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
