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
    """Individual race data structure"""
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
        if self.flags is None:
            self.flags = []

@dataclass
class HorseEntry:
    """Horse entry data structure with multiple races"""
    horse_name: str
    sex: str = ""
    age: int = 0
    breeder_owner: str = ""
    foal_date: str = ""
    reg_code: str = ""
    races: int = 0  # Total number of races
    top_fig: str = ""  # Top Ragozin figure
    lines: List[RaceEntry] = None  # List of race lines
    # New fields for enhanced analysis
    horse_analysis: str = ""  # AI-generated analysis of this horse's overall performance
    performance_trend: str = ""  # Analysis of performance trends
    
    def __post_init__(self):
        if self.lines is None:
            self.lines = []

@dataclass
class RaceData:
    """Race data structure"""
    track_name: str
    race_number: int
    date: str
    surface: str
    distance: str
    weather: str = ""
    horses: List[HorseEntry] = None
    
    def __post_init__(self):
        if self.horses is None:
            self.horses = []
    
    @property
    def race_date(self) -> str:
        """Property to maintain compatibility with api.py"""
        return self.date
    
    @property
    def track_surface(self) -> str:
        """Property to maintain compatibility with frontend"""
        return self.surface

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
                print(f"   ragozin_context: {self.ragozin_context}")
                print(f"✅ Loaded Ragozin symbol sheet context ({len(self.ragozin_context)} characters)")
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
                    print(f"✅ Loaded Ragozin symbol sheet context with pdfplumber ({len(self.ragozin_context)} characters)")
                    return
                    
        except Exception as e:
            print(f"pdfplumber failed for symbol sheet: {e}")
        
        print("❌ Failed to load Ragozin symbol sheet context")
    
    def create_analysis_prompt(self) -> str:
        """Create the prompt for analyzing Ragozin sheets with enhanced context"""
        base_prompt = """🎯 AI Prompt for Ragozin Sheet Interpretation:

You are an expert analyst of Ragozin "The Sheets" and your job is to extract and interpret data from an image of a Ragozin Sheet. The sheet contains performance figures for multiple horses in a given race. You must interpret the structure and symbols to output structured, normalized information for each horse.

The sheet layout includes:
- Multiple vertical columns, one per horse
- Each column contains:
  - Horse name, sex, age, pedigree
  - Number of races, top figure
  - A list of past performances including figures, symbols, and track/date notations
  - Optional notes like trouble indicators, pace figures, or summary lines

🔍 For Each Horse Column, extract the following:

📌 Horse Identity:
- Horse name
- Sex (M/F)
- Age
- Breeder or owner (e.g., Amy Z)
- Sire and Dam (e.g., MAXIMUS MISCHIEF out of OVERLY INDULGENT by PLEASANTLY PERFECT)
- Foaling date (e.g., 25JUN)
- Registration number (e.g., KY6062)

🧾 Summary Performance Header:
- Total races listed (e.g., 5 RACES)
- Top Ragozin figure (e.g., 23)

🧠 For Each Line of Form:
Extract:
- Final Ragozin figure (e.g., 21, 18, 26-, 19+, ^21, P23+)
- Symbols/flags:
  - ~ = troubled trip
  - ^ = paired top
  - F = figure likely "faked" or not accurate
  - + or – = slightly slower or faster than base fig
  - P = pace figure only
- Track Code and Race Info:
  - Example: vMSOP31 = turf at MSOP track, race 31
  - Prefix v = turf, Y = synthetic, etc.
  - Month of race (shown vertically on right)
  - Whether it was a dirt/turf race (based on v/Y prefixes)
  - Race type (read from far-right: e.g., MS = Maiden Special, 10AP29 = $10K claiming at Arlington Park on 29th)

🔁 Optional:
For multi-line summaries like "Rui 5X 2/0/1", parse:
- Track (Rui)
- Number of races (5X)
- Record: Wins/Places/Shows (2/0/1)

📊 Output Format Example (per horse):
```json
{
  "horses": [
    {
      "horse_name": "Just an Opinion",
      "sex": "F",
      "age": 3,
      "breeder_owner": null,
      "foal_date": "25JUN",
      "reg_code": "SMA",
      "races": 2,
      "top_fig": 25,
      "horse_analysis": "Comprehensive analysis of this horse's overall performance, strengths, weaknesses, and racing style",
      "performance_trend": "Analysis of performance trends over time (improving, declining, consistent, etc.)",
      "lines": [
        {
          "fig": "20",
          "flags": [],
          "track": "YDM1",
          "month": "JUN",
          "surface": "dirt",
          "race_type": "Claiming 20K",
          "race_date": "06/15/24",
          "notes": null,
          "race_analysis": "Detailed analysis of this specific race performance, including what the Ragozin figure indicates, impact of symbols, and race context"
        },
        {
          "fig": "17-",
          "flags": ["-"],
          "track": "WMSP1",
          "month": "MAY",
          "surface": "dirt",
          "race_type": "MSW",
          "race_date": "05/20/24",
          "notes": null,
          "race_analysis": "Detailed analysis of this specific race performance, including what the Ragozin figure indicates, impact of symbols, and race context"
        }
      ]
    }
  ]
}
```

📌 Critical Notes:
- A lower figure is faster and better
- Horses with improving patterns (e.g., 26 → 21 → 19) may be considered "live"
- Flags like ~, +, -, ^ are crucial to assess performance beyond raw fig
- Pace figures (P~23+) are not final times and should be stored separately from final figs
- Extract ALL horses and ALL their race data from this page with detailed analysis
- Pay special attention to extracting EXACT race dates in the proper format"""

        # Add Ragozin symbol sheet context if available
        if self.ragozin_context:
            context_prompt = f"""

IMPORTANT CONTEXT - RAGOZIN SYMBOL SHEET REFERENCE:
{self.ragozin_context[:3000]}...

Use this context to better understand the symbols, notations, and data structure in the Ragozin sheets. This will help you interpret the figures, symbols, and performance indicators more accurately.

SPECIFIC FORMAT UNDERSTANDING:
- Each page shows the RECENT RACES of each horse
- Each column represents a specific horse with their recent race history
- Horse names are in column headers (e.g., "KANTHAROS", "JUST AN OPINION")
- Sub-headers show race counts and years (e.g., "2 RACES 23", "4 RACES 24")
- Trainer information appears under horse names (e.g., "GRETCHEN-CONVEYANCE KY")
- Performance data flows vertically with months on the left (DEC, NOV, OCT, etc.)
- Ragozin figures are numbers (lower = better performance)
- Track codes + dates (e.g., "CD21" = Churchill Downs on 21st)
- Race dates should be analyzed EXACTLY - look for specific dates in MM/DD, MM/DD/YY, or similar formats
- Race types: "MS" (Maiden), "AW" (Allowance), "aw" (Starter Allowance)
- Performance symbols: Y (rail trip), F (first lasix), w (won race), etc.

"""
            return context_prompt + base_prompt
        
        return base_prompt
    
    def pdf_to_images_pymupdf(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to images using PyMuPDF (no Poppler required)"""
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
                
                # Set zoom factor for better quality
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
                
                # Render page to image
                pix = page.get_pixmap(matrix=mat)
                
                # Save image
                image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                pix.save(image_path)
                image_paths.append(image_path)
                
                print(f"   Converted page {page_num + 1} to image")
                print(f"   Image path: {image_path}")
            
            doc.close()
            print(f"✅ Converted {len(image_paths)} pages to images")
            return image_paths
            
        except Exception as e:
            # Cleanup on error
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise e
    
    def analyze_page(self, image_path: str) -> Dict[str, Any]:
        """Analyze a single page/image using GPT Vision"""
        try:
            # Encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt = self.create_analysis_prompt()
            
            # Prepare request
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                "max_tokens": 8000, 
                "temperature": 0.1
            }
            
            # Debug: Print the URL being used
            print(f"   🔗 Making API request to: {self.base_url}")
            
            # Make API request
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Parse response
            content = response.json()["choices"][0]["message"]["content"]
            
            # Debug: Print the raw response
            print(f"🚀 Raw AI response: {content}")
            
            # Extract JSON
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content.strip()
            
            # Debug: Print the extracted JSON
            print(f"   Extracted JSON: {json_str[:300]}...")
            
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
    
    def parse_ragozin_sheet(self, pdf_path: str) -> RaceData:
        """Parse a Ragozin sheet PDF using GPT Vision"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"🔍 Parsing PDF: {pdf_path}")
        
        try:
            # Convert PDF to images
            image_paths = self.pdf_to_images_pymupdf(pdf_path)
            
            # Analyze each page
            all_horses = []
            horse_dict = {}  # To avoid duplicates
            
            for i, image_path in enumerate(image_paths):
                print(f"   Analyzing page {i + 1}/{len(image_paths)}...")
                
                if i > 1:
                    continue

                result = self.analyze_page(image_path)
                
                if "error" in result:
                    print(f"   ⚠️  Page {i + 1} analysis failed: {result['error']}")
                    continue
                
                # Collect horses and their races
                if "horses" in result and isinstance(result["horses"], list):
                    for horse_data in result["horses"]:
                        horse_name = horse_data.get("horse_name", "")
                        if not horse_name:
                            continue
                            
                        # Check if we already have this horse
                        if horse_name in horse_dict:
                            # Add lines to existing horse
                            existing_horse = horse_dict[horse_name]
                            if "lines" in horse_data and isinstance(horse_data["lines"], list):
                                for line_data in horse_data["lines"]:
                                    race = RaceEntry(
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
                                    existing_horse.lines.append(race)
                        else:
                            # Create new horse
                            horse = HorseEntry(
                                horse_name=horse_name,
                                sex=horse_data.get("sex", ""),
                                age=horse_data.get("age", 0),
                                breeder_owner=horse_data.get("breeder_owner", ""),
                                foal_date=horse_data.get("foal_date", ""),
                                reg_code=horse_data.get("reg_code", ""),
                                races=horse_data.get("races", 0),
                                top_fig=horse_data.get("top_fig", ""),
                                horse_analysis=horse_data.get("horse_analysis", ""),
                                performance_trend=horse_data.get("performance_trend", "")
                            )
                            
                            # Add lines
                            if "lines" in horse_data and isinstance(horse_data["lines"], list):
                                for line_data in horse_data["lines"]:
                                    race = RaceEntry(
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
                            
                            horse_dict[horse_name] = horse
                            all_horses.append(horse)
            
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
            print(f"✅ Parsing completed: {len(all_horses)} horses found with {total_races} total races")
            return race_data
            
        except Exception as e:
            print(f"❌ Parsing failed: {e}")
            raise
    
    def export_to_json(self, race_data: RaceData, output_path: str) -> None:
        """Export race data to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(asdict(race_data), f, indent=2, default=str)
        print(f"💾 Exported to JSON: {output_path}")
    
    def export_to_csv(self, race_data: RaceData, output_path: str) -> None:
        """Export race data to CSV file"""
        import pandas as pd
        
        # Convert to DataFrame - one row per race
        races_data = []
        for horse in race_data.horses:
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
        print(f"💾 Exported to CSV: {output_path} ({len(races_data)} race entries)")

def main():
    """Main function for testing"""
    import os
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please run the API key setup script first:")
        print("   python fix_api_key.py")
        print("   or")
        print("   python set_api_key.py")
        return
    
    # Validate API key format
    if not api_key.startswith("sk-"):
        print("❌ Invalid API key format. OpenAI API keys start with 'sk-'")
        print("Please run the API key setup script:")
        print("   python fix_api_key.py")
        return
    
    print(f"✅ Using API key: {api_key[:10]}...")
    
    # Initialize parser
    try:
        parser = GPTRagozinParserAlternative(api_key=api_key)
    except Exception as e:
        print(f"❌ Failed to initialize parser: {e}")
        return
    
    # Test with sample PDF
    pdf_path = "cd062525.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Test PDF not found: {pdf_path}")
        return
    
    try:
        # Parse PDF
        race_data = parser.parse_ragozin_sheet(pdf_path)
        
        # Export results
        parser.export_to_json(race_data, "output_alternative.json")
        parser.export_to_csv(race_data, "output_alternative.csv")
        
        print("\n✅ Alternative parser test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main() 