#!/usr/bin/env python3
"""
Alternative GPT-based Ragozin Sheets Parser
Uses PyMuPDF to convert PDF pages to images (no Poppler required)
Enhanced with Ragozin symbol sheet context for better understanding
"""

import os
import json
import base64
import tempfile
import shutil
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import requests
from io import BytesIO

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

try:
    import fitz  # PyMuPDF
    import pdfplumber
    PDF_TEXT_AVAILABLE = True
except ImportError:
    PDF_TEXT_AVAILABLE = False
    print("PDF text extraction not available. Install with: pip install PyMuPDF pdfplumber")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. Install with: pip install Pillow")

@dataclass
class RaceEntry:
    """Enhanced individual race data structure with AI analysis"""
    race_year: int = 0  # Year of the race (e.g., 2023)
    race_index: int = 0  # Index within the year (1 = most recent)
    figure_raw: str = ""  # Full printed figure with symbols (e.g., "P~23+")
    parsed_figure: float = 0.0  # Numeric decimal representation (e.g., 23.25)
    pre_symbols: List[str] = None  # Symbols before figure (e.g., ["P~"])
    post_symbols: List[str] = None  # Symbols after figure (e.g., ["+"])
    distance_bracket: str = ""  # Distance bracket inferred from font style
    surface_type: str = ""  # Surface type based on track label prefix
    track_code: str = ""  # Track code parsed from gray arrow (e.g., "SAR", "BEL")
    date_code: str = ""  # Day from gray arrow (e.g., "31")
    month_label: str = ""  # Vertical month label beside arrow (e.g., "OCT")
    race_class_code: str = ""  # Race class code from far right (e.g., "MS", "AW")
    trouble_indicators: List[str] = None  # Post-symbols like T, D, Z, B, etc.
    ai_analysis: Dict[str, str] = None  # AI analysis with left_side, middle, right_side, full_interpretation
    
    # Legacy fields for compatibility
    fig: str = ""  # Ragozin figure (e.g., "20", "17-", "26+")
    flags: List[str] = None  # List of flags/symbols (e.g., ["~", "+", "F"])
    track: str = ""  # Track code (e.g., "YDM1", "WMSP1")
    month: str = ""  # Month of race (e.g., "JUN", "MAY")
    surface: str = ""  # Surface type (e.g., "dirt", "turf")
    race_type: str = ""  # Race type (e.g., "MSW", "Claiming 20K")
    race_date: str = ""  # Exact race date (e.g., "06/15/24")
    notes: str = ""  # Additional notes
    race_analysis: str = ""  # AI-generated analysis of this race
    
    def __post_init__(self):
        if self.pre_symbols is None:
            self.pre_symbols = []
        if self.post_symbols is None:
            self.post_symbols = []
        if self.trouble_indicators is None:
            self.trouble_indicators = []
        if self.flags is None:
            self.flags = []
        if self.ai_analysis is None:
            self.ai_analysis = {
                "left_side": "",
                "middle": "",
                "right_side": "",
                "full_interpretation": ""
            }

@dataclass
class HorseEntry:
    """Enhanced horse entry data structure with comprehensive metadata and AI analysis"""
    horse_name: str
    sex: str = ""  # "M" or "F"
    foaling_year: int = 0  # Last 2 digits in heading (e.g., 19 = 2019)
    race_number: int = 0  # From heading (e.g., "Race 3")
    sire: str = ""  # Above horse name (e.g., "UNIFIED")
    dam: str = ""  # Below sire (e.g., "MYKINDSAINT")
    damsire: str = ""  # Below dam if available (e.g., "SAINT BALLADO")
    state_bred: str = ""  # State abbreviation (e.g., "KY")
    track_code: str = ""  # (e.g., "CD" for Churchill Downs)
    sheet_page_number: str = ""  # From heading (e.g., "p21")
    
    # Legacy fields for compatibility
    age: int = 0
    breeder_owner: str = ""
    foal_date: str = ""
    reg_code: str = ""
    races: int = 0  # Total number of races
    top_fig: str = ""  # Top Ragozin figure
    lines: List[RaceEntry] = None  # List of race lines
    
    # Enhanced analysis fields
    horse_analysis: str = ""  # AI-generated analysis of this horse's overall performance
    performance_trend: str = ""  # Analysis of performance trends
    
    def __post_init__(self):
        if self.lines is None:
            self.lines = []

@dataclass
class HorsePastPerformance:
    """Horse past performance data structure - represents one horse's complete racing history"""
    horse: HorseEntry = None
    parsed_at: str = ""
    source_file: str = ""
    
    def __post_init__(self):
        if self.horse is None:
            self.horse = HorseEntry(horse_name="Unknown")
    
    # Properties for compatibility with existing API
    @property
    def track_name(self) -> str:
        """Property to maintain compatibility with api.py"""
        return "Individual Horse Past Performance"
    
    @property
    def race_date(self) -> str:
        """Property to maintain compatibility with api.py"""
        return "Multiple Dates"
    
    @property
    def race_number(self) -> int:
        """Property to maintain compatibility with api.py"""
        return 0
    
    @property
    def surface(self) -> str:
        """Property to maintain compatibility with api.py"""
        return "Multiple Surfaces"
    
    @property
    def distance(self) -> str:
        """Property to maintain compatibility with api.py"""
        return "Multiple Distances"
    
    @property
    def weather(self) -> str:
        """Property to maintain compatibility with api.py"""
        return ""
    
    @property
    def horses(self) -> List[HorseEntry]:
        """Property to maintain compatibility with api.py"""
        return [self.horse] if self.horse else []

class GPTRagozinParserAlternative:
    """Alternative GPT-based parser for Ragozin sheets using PyMuPDF"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        # Ensure no trailing slash to prevent double slashes
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if not PDF_TEXT_AVAILABLE:
            print("Warning: PDF text extraction not available. Install with: pip install PyMuPDF pdfplumber")
        
        if not PIL_AVAILABLE:
            print("Warning: PIL not available. Install with: pip install Pillow")
        
        # Initialize Ragozin symbol sheet context
        self.ragozin_context = ""
        self._load_ragozin_context()
    
    def _load_ragozin_context(self, symbol_sheet_path: str = "ragozin-symbol-sheet-the-sheets.pdf"):
        """Load and extract text from the Ragozin symbol sheet PDF for context"""
        if not PDF_TEXT_AVAILABLE:
            print("Warning: PDF text extraction not available. Context will not be loaded.")
            return
        
        if not os.path.exists(symbol_sheet_path):
            print(f"Warning: Ragozin symbol sheet not found at {symbol_sheet_path}")
            return
        
        try:
            # Try PyMuPDF first
            doc = fitz.open(symbol_sheet_path)
            text_content = ""
            for page in doc:
                text_content += page.get_text()
            doc.close()
            
            if text_content.strip():
                self.ragozin_context = text_content.strip()
                print(f"[OK] Loaded Ragozin symbol sheet context ({len(self.ragozin_context)} characters)")
                return
            
        except Exception as e:
            print(f"PyMuPDF failed for symbol sheet: {e}")
        
        try:
            # Fallback to pdfplumber
            with pdfplumber.open(symbol_sheet_path) as pdf:
                text_content = ""
                for page in pdf.pages:
                    text_content += page.extract_text() or ""
                
                if text_content.strip():
                    self.ragozin_context = text_content.strip()
                    print(f"[OK] Loaded Ragozin symbol sheet context with pdfplumber ({len(self.ragozin_context)} characters)")
                    return
                    
        except Exception as e:
            print(f"pdfplumber failed for symbol sheet: {e}")
        
        print("[ERROR] Failed to load Ragozin symbol sheet context")
    
    def create_analysis_prompt(self) -> str:
        """Create the prompt for analyzing Ragozin sheets with enhanced context"""
        return """
        AI PROMPT: Extract and Analyze Horse Performance from Ragozin Sheets

You are an AI system trained to read and analyze horse racing performance from Ragozin Sheets. Each sheet contains one horse’s race history. Your job is to extract and structure each race’s data and provide a detailed AI interpretation for each.
—
1. SHEET STRUCTURE
Each column = 1 year of races (e.g. “2 RACES 23” means 2 races in 2023).
Rows = individual races, most recent at the top.
Horse meta-data appears at the top: name, foaling year, sex, sire/dam info, track, and race number.
Each row contains a performance figure, symbols, track/date, and race-class details.
—
2. EXTRACT HORSE META-DATA
From the top heading of the sheet extract:
horse_name
sex ("M" or "F")
foaling_year (e.g. "19" = 2019)
race_number (e.g. "Race 3")
sire
dam
damsire (if shown)
state_bred (e.g. “KY”)
track_code (e.g. “CD”)
sheet_page_number (e.g. “p21”)
—
3. RACE-LEVEL DATA PER ROW
For each race (row), extract:
race_year (from column header)
race_index (1 = most recent, increasing down)
figure_raw (e.g. “~19+”)
parsed_figure (e.g. 19.25)
pre_symbols (e.g. ["", "P"])
post_symbols (e.g. ["+", "V", "T", "Z"])
distance_bracket (determined by font — see section 4)
surface_type (from pre-symbols: turf, good turf, sloppy, etc.)
track_code (gray arrow e.g. “SAR”)
date_code (e.g. “21”)
month_label (e.g. “JUN”)
race_class_code (far right of the row e.g. “40CD21” or “AWBEL31”)
trainer_initials (letters at far right like “RJK”, “CBr”)
trouble_indicators (post-symbols like V, T, D, etc.)
—
4. DISTANCE FROM FIGURE FONT STYLE
Use font appearance to assign distance:
Font Style	Distance Description
Light italic	< 5 furlongs
Light upright	5 – 5.5 furlongs
Regular upright	6 furlongs
Bold upright	6.5 – 7 furlongs
Condensed bold upright	7.5 – 1 mile 40 yards
Wide bold	1 mile 70 yds – 1⅛ miles
Very wide bold italic	1 3/16 – 1¼ miles
Extra-wide black italic	Over 1¼ miles
—
5. RAGOZIN SYMBOL DICTIONARY
Pre-Figure symbols (surface / pace / condition):
* PolyTrack
= turf
^= good turf
.= yielding turf
:= soft turf
’ wet fast
^ good, slow
. wet, sloppy, muddy
: very bad wet track
.. very slow track
../ ploughed up after freezing
.: heavy track
/ frozen track
r rain
s snow
F 1st lasix
g heavy gusting wind
G very heavy gusting wind
~ approximate
X~ fig. stretch call
P~ adjusted for slow pace
circle = missing number
XX = Did Not Finish
+ is a quarter-point
- minus quarter-point
" is a half-point

Post-Figure symbols (trip / outcome / trouble):
& claimed by... w/initials
 initials in caps = hot trainer
s off poorly <2 lengths
S off poorly 2-4 lengths
D dwelt >4 lengths
P pace too slow (unadjusted)
Z bled
J lost jockey
m mud caulks
b buried race (better than looked)
Q switched off turf
w won race
< bar shoe on
> bar shoe off
( shoes on
) shoes off
R ran off before race
$ bet for no obvious reason
G unruly at the gate
B bore in/bore out
K lame/broke down
k sore
t small trouble
T big trouble
n no lasix used
L back on Lasix
] blinkers off
[ blinkers on
V 4 or more horse-widths wide on final turn
v 3-3.5 horse-widths wide on final turn
Y rail trip on final turn
E Fell during race
u wore bend shoes
f fell back sharply after start
—
6. DETAILED AI INTERPRETATION
For each race, append an “ai_analysis” block with the following style:
Example:
{
  "ai_analysis": {
    "left_side": "“19+” means a Ragozin figure of 19, slightly worse than 19.0, approximately 19.25.",
    "middle": "“V C vet 6/21” suggests the horse raced wide, may have been claimed, and saw a vet on June 21.",
    "right_side": "The race occurred at Churchill Downs (CD) on June 21 in a $40,000 claiming race.",
    "full_interpretation": "Despite breaking poorly and racing wide, the horse posted a 19.25 Ragozin — a solid figure. Vet attention may explain a layoff or performance dip. This was a better effort than the previous 26 figure."
  }
}
—
7. OUTPUT FORMAT (EXAMPLE)
{
  "horse_name": "UNDERHILL’S TAB",
  "sex": "M",
  "foaling_year": 2019,
  "race_number": 3,
  "sire": "UNIFIED",
  "dam": "MYKINDSAINT",
  "damsire": "SAINT BALLADO",
  "state_bred": "KY",
  "track_code": "CD",
  "sheet_page_number": "p21",
  "races": [
    {
      "race_year": 2023,
      "race_index": 1,
      "figure_raw": "~19+",
      "parsed_figure": 19.25,
      "pre_symbols": ["~"],
      "post_symbols": ["+"],
      "distance_bracket": "6 furlongs",
      "surface_type": "dirt",
      "track_code": "CD",
      "date_code": "21",
      "month_label": "JUN",
      "race_class_code": "40CD21",
      "trainer_initials": "RJK",
      "trouble_indicators": ["V", "S"],
      "ai_analysis": {
        "left_side": "“19+” is a Ragozin figure slightly above 19, approx. 19.25.",
        "middle": "Symbols 'V' and 'S' indicate a wide trip and slow break.",
        "right_side": "This was a $40,000 claimer at Churchill on June 21.",
        "full_interpretation": "The horse ran 4+ wide and broke poorly, yet still posted a 19.25 — a good performance. Indicates fitness despite minor trip issues."
      }
    }
  ]
}
"""
    
    def pdf_to_images_pymupdf(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to high-quality images using PyMuPDF"""
        if not PDF_TEXT_AVAILABLE:
            raise ImportError("PyMuPDF not available. Install with: pip install PyMuPDF")
        
        if not PIL_AVAILABLE:
            raise ImportError("PIL not available. Install with: pip install Pillow")
        
        # Create temporary directory for images
        temp_dir = tempfile.mkdtemp(prefix="ragozin_parser_")
        image_paths = []
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                # Get page
                page = doc.load_page(page_num)
                
                # Set high zoom factor for better quality (4x zoom for better OCR)
                mat = fitz.Matrix(4.0, 4.0)  # 4x zoom for high quality
                
                # Render page to image with high DPI
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Save image with high quality
                image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                pix.save(image_path)
                image_paths.append(image_path)
                
                print(f"   Converted page {page_num + 1} to high-quality image")
                print(f"   Image path: {image_path}")
            
            doc.close()
            print(f"[OK] Converted {len(image_paths)} pages to high-quality images")
            return image_paths
            
        except Exception as e:
            # Cleanup on error
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise e
    
    def analyze_pdf_page_direct(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """Analyze a PDF page directly by sending the page as a PDF to GPT Vision"""
        try:
            # Open PDF and extract the specific page
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            
            # Create a new PDF with just this page
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            # Save the single page as a temporary PDF
            temp_pdf_path = tempfile.mktemp(suffix=".pdf")
            new_doc.save(temp_pdf_path)
            new_doc.close()
            doc.close()
            
            # Read the PDF file and encode it
            with open(temp_pdf_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
                base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
            
            # Clean up temporary file
            os.remove(temp_pdf_path)
            
            prompt = self.create_analysis_prompt()
            
            messages = [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is a Ragozin sheet PDF page. Please extract and analyze all race data and return JSON with interpretation."},
                        {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{base64_pdf}"}}
                    ]
                }
            ]
            
            # Prepare request
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 8192, 
                "temperature": 0.1
            }
            
            # Debug: Print the URL being used
            print(f"   [LINK] Making API request to: {self.base_url}")
            
            # Make API request
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Parse response
            content = response.json()["choices"][0]["message"]["content"]
            
            # Debug: Print the raw response
            print(f"[ROCKET] Raw AI response: {content}")
            
            # Extract JSON
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content.strip()
            
            # Debug: Print the extracted JSON
            print(f"   Extracted JSON: {json_str}")
            
            return json.loads(json_str)
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {"error": f"API endpoint not found. Check if the model '{self.model}' is still available. Error: {e}"}
            elif e.response.status_code == 401:
                return {"error": f"Unauthorized. Check your API key. Error: {e}"}
            else:
                return {"error": f"HTTP error {e.response.status_code}: {e}"}
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
    
    def analyze_full_pdf_direct(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze the entire PDF directly by sending it to GPT Vision"""
        try:
            # Read the entire PDF file and encode it
            with open(pdf_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
                base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
            
            prompt = self.create_analysis_prompt()
            
            messages = [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is a complete Ragozin sheet PDF. Please extract and analyze all race data from all pages and return JSON with interpretation."},
                        {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{base64_pdf}"}}
                    ]
                }
            ]
            
            # Prepare request
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 8192, 
                "temperature": 0.1
            }
            
            # Debug: Print the URL being used
            print(f"   [LINK] Making API request to: {self.base_url}")
            
            # Make API request
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Parse response
            content = response.json()["choices"][0]["message"]["content"]
            
            # Debug: Print the raw response
            print(f"[ROCKET] Raw AI response: {content}")
            
            # Extract JSON
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content.strip()
            
            # Debug: Print the extracted JSON
            print(f"   Extracted JSON: {json_str}")
            
            return json.loads(json_str)
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {"error": f"API endpoint not found. Check if the model '{self.model}' is still available. Error: {e}"}
            elif e.response.status_code == 401:
                return {"error": f"Unauthorized. Check your API key. Error: {e}"}
            else:
                return {"error": f"HTTP error {e.response.status_code}: {e}"}
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
    
    def analyze_page(self, image_path: str) -> Dict[str, Any]:
        """Analyze a single page/image using GPT Vision"""
        try:
            # Encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt = self.create_analysis_prompt()
            
            messages = [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is a Ragozin sheet image. Please extract and analyze all race data and return JSON with interpretation."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ]
            # Prepare request
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 8192, 
                "temperature": 0.1
            }
            
            # Debug: Print the URL being used
            print(f"   [LINK] Making API request to: {self.base_url}")
            
            # Make API request
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Parse response
            content = response.json()["choices"][0]["message"]["content"]
            
            # Debug: Print the raw response
            print(f"[ROCKET] Raw AI response: {content}")
            
            # Extract JSON
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content.strip()
            
            # Debug: Print the extracted JSON
            print(f"   Extracted JSON: {json_str}")
            
            return json.loads(json_str)
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {"error": f"API endpoint not found. Check if the model '{self.model}' is still available. Error: {e}"}
            elif e.response.status_code == 401:
                return {"error": f"Unauthorized. Check your API key. Error: {e}"}
            else:
                return {"error": f"HTTP error {e.response.status_code}: {e}"}
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
    
    def parse_ragozin_sheet(self, pdf_path: str, use_direct_pdf: bool = True, use_full_pdf: bool = False) -> HorsePastPerformance:
        """Parse a Ragozin sheet PDF using GPT Vision"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"[SEARCH] Parsing PDF: {pdf_path}")
        
        try:
            # Check if we should analyze the full PDF at once
            if use_full_pdf:
                print(f"   Analyzing entire PDF at once...")
                result = self.analyze_full_pdf_direct(pdf_path)
                
                if "error" in result:
                    print(f"   [WARNING]  Full PDF analysis failed: {result['error']}")
                    return HorsePastPerformance(
                        horse=HorseEntry(horse_name="Analysis Failed"),
                        parsed_at="",
                        source_file=pdf_path
                    )
                
                # Process the result from full PDF analysis
                horse_data = None
                if "horses" in result and isinstance(result["horses"], list) and len(result["horses"]) > 0:
                    horse_data = result["horses"][0]  # Take the first horse
                elif "horse_name" in result:
                    horse_data = result  # Direct horse data
                
                if horse_data:
                    horse_name = horse_data.get("horse_name", "Unknown Horse")
                    
                    # Create horse entry with enhanced metadata
                    horse = HorseEntry(
                        horse_name=horse_name,
                        sex=horse_data.get("sex", ""),
                        foaling_year=horse_data.get("foaling_year", 0),
                        race_number=horse_data.get("race_number", 0),
                        sire=horse_data.get("sire", ""),
                        dam=horse_data.get("dam", ""),
                        damsire=horse_data.get("damsire", ""),
                        state_bred=horse_data.get("state_bred", ""),
                        track_code=horse_data.get("track_code", ""),
                        sheet_page_number=horse_data.get("sheet_page_number", ""),
                        # Legacy fields
                        age=horse_data.get("age", 0),
                        breeder_owner=horse_data.get("breeder_owner", ""),
                        foal_date=horse_data.get("foal_date", ""),
                        reg_code=horse_data.get("reg_code", ""),
                        races=horse_data.get("races", 0),
                        top_fig=horse_data.get("top_fig", ""),
                        horse_analysis=horse_data.get("horse_analysis", ""),
                        performance_trend=horse_data.get("performance_trend", "")
                    )
                    
                    # Add race lines - handle both "races" and "lines" keys
                    race_entries = []
                    if "races" in horse_data and isinstance(horse_data["races"], list):
                        race_entries = horse_data["races"]
                    elif "lines" in horse_data and isinstance(horse_data["lines"], list):
                        race_entries = horse_data["lines"]
                    
                    for line_data in race_entries:
                        race = RaceEntry(
                            # Enhanced fields
                            race_year=line_data.get("race_year", 0),
                            race_index=line_data.get("race_index", 0),
                            figure_raw=line_data.get("figure_raw", ""),
                            parsed_figure=line_data.get("parsed_figure", 0.0),
                            pre_symbols=line_data.get("pre_symbols", []),
                            post_symbols=line_data.get("post_symbols", []),
                            distance_bracket=line_data.get("distance_bracket", ""),
                            surface_type=line_data.get("surface_type", ""),
                            track_code=line_data.get("track_code", ""),
                            date_code=line_data.get("date_code", ""),
                            month_label=line_data.get("month_label", ""),
                            race_class_code=line_data.get("race_class_code", ""),
                            trouble_indicators=line_data.get("trouble_indicators", []),
                            ai_analysis=line_data.get("ai_analysis", {}),
                            # Legacy fields for compatibility
                            fig=line_data.get("fig", ""),
                            flags=line_data.get("flags", []),
                            track=line_data.get("track", ""),
                            month=line_data.get("month", ""),
                            surface=line_data.get("surface", ""),
                            race_type=line_data.get("race_type", ""),
                            race_date=line_data.get("race_date", ""),
                            notes=line_data.get("notes", ""),
                            race_analysis=line_data.get("race_analysis", "")
                        )
                        horse.lines.append(race)
                    
                    # Create horse past performance
                    horse_performance = HorsePastPerformance(
                        horse=horse,
                        parsed_at="",
                        source_file=pdf_path
                    )
                    
                    total_races = len(horse.lines)
                    print(f"[OK] Full PDF parsing completed: {horse_name} with {total_races} races")
                    return horse_performance
                else:
                    print(f"[ERROR] No horse data found in full PDF analysis")
                    return HorsePastPerformance(
                        horse=HorseEntry(horse_name="No Data Found"),
                        parsed_at="",
                        source_file=pdf_path
                    )
            
            # Get PDF info for page-by-page analysis
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            print(f"   Found {total_pages} pages in PDF")
            
            # Analyze all pages (each page may represent one horse or part of a horse's data)
            if total_pages > 0:
                print(f"   Analyzing {total_pages} pages for horse past performance...")
                
                all_horses = []
                combined_race_lines = []
                
                for page_num in range(total_pages):
                    print(f"   Analyzing page {page_num + 1}/{total_pages}...")
                    
                    # Choose analysis method
                    if use_direct_pdf:
                        print(f"   Using direct PDF analysis for page {page_num + 1}")
                        result = self.analyze_pdf_page_direct(pdf_path, page_num)
                    else:
                        print(f"   Using image analysis for page {page_num + 1}")
                        # Convert PDF to images
                        image_paths = self.pdf_to_images_pymupdf(pdf_path)
                        if page_num < len(image_paths):
                            result = self.analyze_page(image_paths[page_num])
                        else:
                            print(f"   [WARNING]  Page {page_num + 1} not found in converted images")
                            continue
                    
                    if "error" in result:
                        print(f"   [WARNING]  Page {page_num + 1} analysis failed: {result['error']}")
                        continue
                    
                    # Extract horse data from this page
                    horse_data = None
                    if "horses" in result and isinstance(result["horses"], list) and len(result["horses"]) > 0:
                        horse_data = result["horses"][0]  # Take the first horse from this page
                    elif "horse_name" in result:
                        horse_data = result  # Direct horse data
                    
                    if horse_data:
                        horse_name = horse_data.get("horse_name", f"Unknown Horse Page {i + 1}")
                        
                        # Create horse entry with enhanced metadata (use data from first page for metadata)
                        if i == 0:
                            horse = HorseEntry(
                                horse_name=horse_name,
                                sex=horse_data.get("sex", ""),
                                foaling_year=horse_data.get("foaling_year", 0),
                                race_number=horse_data.get("race_number", 0),
                                sire=horse_data.get("sire", ""),
                                dam=horse_data.get("dam", ""),
                                damsire=horse_data.get("damsire", ""),
                                state_bred=horse_data.get("state_bred", ""),
                                track_code=horse_data.get("track_code", ""),
                                sheet_page_number=horse_data.get("sheet_page_number", ""),
                                # Legacy fields
                                age=horse_data.get("age", 0),
                                breeder_owner=horse_data.get("breeder_owner", ""),
                                foal_date=horse_data.get("foal_date", ""),
                                reg_code=horse_data.get("reg_code", ""),
                                races=horse_data.get("races", 0),
                                top_fig=horse_data.get("top_fig", ""),
                                horse_analysis=horse_data.get("horse_analysis", ""),
                                performance_trend=horse_data.get("performance_trend", "")
                            )
                            all_horses.append(horse)
                        
                        # Add race lines from this page - handle both "races" and "lines" keys
                        race_entries = []
                        if "races" in horse_data and isinstance(horse_data["races"], list):
                            race_entries = horse_data["races"]
                        elif "lines" in horse_data and isinstance(horse_data["lines"], list):
                            race_entries = horse_data["lines"]
                        
                        for line_data in race_entries:
                            race = RaceEntry(
                                # Enhanced fields
                                race_year=line_data.get("race_year", 0),
                                race_index=line_data.get("race_index", 0),
                                figure_raw=line_data.get("figure_raw", ""),
                                parsed_figure=line_data.get("parsed_figure", 0.0),
                                pre_symbols=line_data.get("pre_symbols", []),
                                post_symbols=line_data.get("post_symbols", []),
                                distance_bracket=line_data.get("distance_bracket", ""),
                                surface_type=line_data.get("surface_type", ""),
                                track_code=line_data.get("track_code", ""),
                                date_code=line_data.get("date_code", ""),
                                month_label=line_data.get("month_label", ""),
                                race_class_code=line_data.get("race_class_code", ""),
                                trouble_indicators=line_data.get("trouble_indicators", []),
                                ai_analysis=line_data.get("ai_analysis", {}),
                                # Legacy fields for compatibility
                                fig=line_data.get("fig", ""),
                                flags=line_data.get("flags", []),
                                track=line_data.get("track", ""),
                                month=line_data.get("month", ""),
                                surface=line_data.get("surface", ""),
                                race_type=line_data.get("race_type", ""),
                                race_date=line_data.get("race_date", ""),
                                notes=line_data.get("notes", ""),
                                race_analysis=line_data.get("race_analysis", "")
                            )
                            combined_race_lines.append(race)
                
                # Combine all race lines into the main horse
                if all_horses:
                    main_horse = all_horses[0]  # Use the first horse as the main entry
                    main_horse.lines.extend(combined_race_lines)
                    
                    # Update total races count
                    main_horse.races = len(main_horse.lines)
                    
                    # Create horse past performance
                    horse_performance = HorsePastPerformance(
                        horse=main_horse,
                        parsed_at="",
                        source_file=pdf_path
                    )
                    
                    total_races = len(main_horse.lines)
                    print(f"[OK] Parsing completed: {main_horse.horse_name} with {total_races} races from {len(image_paths)} pages")
                    return horse_performance
                else:
                    print(f"[ERROR] No horse data found in any page")
                    return HorsePastPerformance(
                        horse=HorseEntry(horse_name="No Data Found"),
                        parsed_at="",
                        source_file=pdf_path
                    )
            else:
                print(f"[ERROR] No pages found in PDF")
                return HorsePastPerformance(
                    horse=HorseEntry(horse_name="No Pages Found"),
                    parsed_at="",
                    source_file=pdf_path
                )
            
            # Clean up temporary images
            for image_path in image_paths:
                if os.path.exists(image_path):
                    os.remove(image_path)
            
            # Create race data
            race_data = RaceData(
                track_name="Multiple Tracks",  # Since we have multiple tracks
                race_number=0,  # Multiple races
                date="Multiple Dates",  # Since we have multiple dates
                surface="Multiple Surfaces",  # Since we have multiple surfaces
                distance="Multiple Distances",  # Since we have multiple distances
                weather="",  # Weather not available in this format
                horses=all_horses
            )
            
            total_races = sum(len(horse.lines) for horse in all_horses)
            print(f"[OK] Parsing completed: {len(all_horses)} horses found with {total_races} total races")
            return race_data
            
        except Exception as e:
            print(f"[ERROR] Parsing failed: {e}")
            raise
    
    def export_to_json(self, horse_performance: HorsePastPerformance, output_path: str) -> None:
        """Export horse past performance data to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(asdict(horse_performance), f, indent=2, default=str)
        print(f"[DISK] Exported to JSON: {output_path}")
    
    def export_to_csv(self, horse_performance: HorsePastPerformance, output_path: str) -> None:
        """Export horse past performance data to CSV file"""
        import pandas as pd
        
        # Convert to DataFrame - one row per race
        races_data = []
        horse = horse_performance.horse
        
        for line in horse.lines:
            races_data.append({
                "horse_name": horse.horse_name,
                "sex": horse.sex,
                "age": horse.age,
                "breeder_owner": horse.breeder_owner,
                "foal_date": horse.foal_date,
                "reg_code": horse.reg_code,
                "total_races": horse.races,
                "top_fig": horse.top_fig,
                "horse_analysis": horse.horse_analysis,
                "performance_trend": horse.performance_trend,
                "fig": line.fig,
                "flags": ",".join(line.flags) if line.flags else "",
                "track": line.track,
                "month": line.month,
                "surface": line.surface,
                "race_type": line.race_type,
                "race_date": line.race_date,
                "notes": line.notes,
                "race_analysis": line.race_analysis
            })
        
        df = pd.DataFrame(races_data)
        df.to_csv(output_path, index=False)
        print(f"[DISK] Exported to CSV: {output_path} ({len(races_data)} race entries)")

def main():
    """Main function for testing"""
    import os
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        print("Please run the API key setup script first:")
        print("   python fix_api_key.py")
        print("   or")
        print("   python set_api_key.py")
        return
    
    # Validate API key format
    if not api_key.startswith("sk-"):
        print("[ERROR] Invalid API key format. OpenAI API keys start with 'sk-'")
        print("Please run the API key setup script:")
        print("   python fix_api_key.py")
        return
    
    print(f"[OK] Using API key: {api_key[:10]}...")
    
    # Initialize parser
    try:
        parser = GPTRagozinParserAlternative(api_key=api_key)
    except Exception as e:
        print(f"[ERROR] Failed to initialize parser: {e}")
        return
    
    # Test with sample PDF
    pdf_path = "cd062525.pdf"
    if not os.path.exists(pdf_path):
        print(f"[ERROR] Test PDF not found: {pdf_path}")
        return
    
    try:
        # Test with direct PDF analysis (recommended for better OCR)
        print("\n[SEARCH] Testing with direct PDF analysis...")
        horse_performance = parser.parse_ragozin_sheet(pdf_path, use_direct_pdf=True)
        
        # Export results
        parser.export_to_json(horse_performance, "output_alternative_direct.json")
        parser.export_to_csv(horse_performance, "output_alternative_direct.csv")
        
        print("\n[OK] Direct PDF analysis completed successfully!")
        
        # Test with image analysis for comparison
        print("\n[SEARCH] Testing with image analysis for comparison...")
        horse_performance_img = parser.parse_ragozin_sheet(pdf_path, use_direct_pdf=False)
        
        # Export results
        parser.export_to_json(horse_performance_img, "output_alternative_image.json")
        parser.export_to_csv(horse_performance_img, "output_alternative_image.csv")
        
        print("\n[OK] Image analysis completed successfully!")
        print("\n[CHART] Comparison:")
        print(f"   Direct PDF: {len(horse_performance.horse.lines)} races")
        print(f"   Image: {len(horse_performance_img.horse.lines)} races")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 