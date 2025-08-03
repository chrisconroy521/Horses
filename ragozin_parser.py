import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import numpy as np
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HorseEntry:
    """Data class for individual horse race entries"""
    horse_name: str
    figure: Optional[float]
    track_surface: str  # 'dirt', 'turf', 'poly'
    race_date: str
    race_number: int
    finish_position: Optional[int]
    trouble_indicators: List[str]
    weather_conditions: Optional[str]
    distance: Optional[str]
    odds: Optional[float]
    trainer: Optional[str]
    jockey: Optional[str]
    weight: Optional[float]
    speed_rating: Optional[float]
    pace_rating: Optional[float]
    class_rating: Optional[float]

@dataclass
class RaceData:
    """Data class for complete race information"""
    track_name: str
    race_date: str
    race_number: int
    surface: str
    distance: str
    weather: Optional[str]
    horses: List[HorseEntry]
    race_conditions: Optional[str]

class RagozinParser:
    """
    Parser for Ragozin performance sheets
    Handles PDF extraction and data normalization
    """
    
    def __init__(self):
        self.surface_patterns = {
            'dirt': r'\b(dirt|fast|sloppy|muddy|wet)\b',
            'turf': r'\b(turf|firm|good|yielding|soft)\b',
            'poly': r'\b(poly|synthetic|artificial|all weather)\b'
        }
        
        self.trouble_indicators = [
            'trouble', 'bobbled', 'stumbled', 'checked', 'steadied',
            'wide', 'inside', 'blocked', 'clipped heels', 'bumped',
            'paceless', 'slow pace', 'fast pace', 'rail', 'outside'
        ]
        
        self.weather_conditions = [
            'rain', 'snow', 'fog', 'wind', 'hot', 'cold', 'humid',
            'dry', 'clear', 'overcast', 'sunny', 'cloudy'
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using multiple methods for reliability
        """
        text_content = ""
        
        # Method 1: PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text_content += page.get_text()
            doc.close()
            logger.info(f"Successfully extracted text using PyMuPDF from {pdf_path}")
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")
        
        # Method 2: pdfplumber (if PyMuPDF didn't work well)
        if not text_content.strip():
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text_content += page.extract_text() or ""
                logger.info(f"Successfully extracted text using pdfplumber from {pdf_path}")
            except Exception as e:
                logger.error(f"pdfplumber failed: {e}")
        
        return text_content
    
    def parse_race_header(self, text: str) -> Dict[str, Any]:
        """
        Extract race header information (track, date, race number, surface, distance)
        """
        header_info = {
            'track_name': None,
            'race_date': None,
            'race_number': None,
            'surface': None,
            'distance': None,
            'weather': None
        }
        
        # Track name patterns (common race tracks)
        track_patterns = [
            r'\b(Saratoga|Belmont|Aqueduct|Churchill|Keeneland|Santa Anita|Del Mar|Gulfstream|Tampa|Oaklawn|Monmouth|Arlington|Woodbine|Lone Star|Remington|Sam Houston|Delta Downs|Evangeline|Fair Grounds|Louisiana Downs)\b',
            r'([A-Z][a-z]+)\s+(Race|Racing|Park|Downs|Field|Course)',
        ]
        
        for pattern in track_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                header_info['track_name'] = match.group(0)
                break
        
        # Date patterns
        date_patterns = [
            r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',
            r'\b(\d{1,2}-\d{1,2}-\d{2,4})\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                header_info['race_date'] = match.group(0)
                break
        
        # Race number
        race_num_match = re.search(r'Race\s*#?\s*(\d+)', text, re.IGNORECASE)
        if race_num_match:
            header_info['race_number'] = int(race_num_match.group(1))
        
        # Surface detection
        for surface, pattern in self.surface_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                header_info['surface'] = surface
                break
        
        # Distance patterns
        distance_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(furlongs?|f)\b',
            r'\b(\d+(?:\.\d+)?)\s*(miles?|mi)\b',
            r'\b(\d+)\s*(yards?|yd)\b'
        ]
        
        for pattern in distance_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                header_info['distance'] = match.group(0)
                break
        
        # Weather conditions
        for condition in self.weather_conditions:
            if re.search(rf'\b{condition}\b', text, re.IGNORECASE):
                header_info['weather'] = condition
                break
        
        return header_info
    
    def parse_horse_entries(self, text: str) -> List[HorseEntry]:
        """
        Parse individual horse entries from the text
        """
        horses = []
        
        # Split text into lines for processing
        lines = text.split('\n')
        
        # Look for horse entry patterns
        # This is a simplified pattern - will need refinement based on actual sheet format
        horse_patterns = [
            # Pattern 1: Horse name followed by figure
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(\d+(?:\.\d+)?)',
            # Pattern 2: Figure followed by horse name
            r'(\d+(?:\.\d+)?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            # Pattern 3: Horse name with various data
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(\d+(?:\.\d+)?)\s+(\d+)\s+(\d+(?:\.\d+)?)',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match horse entry patterns
            for pattern in horse_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    try:
                        if len(match) >= 2:
                            # Determine which part is horse name vs figure
                            if match[0].replace('.', '').isdigit():
                                figure = float(match[0])
                                horse_name = match[1]
                            else:
                                horse_name = match[0]
                                figure = float(match[1]) if match[1].replace('.', '').isdigit() else None
                            
                            # Extract additional information if available
                            finish_pos = None
                            odds = None
                            trainer = None
                            jockey = None
                            
                            if len(match) > 2:
                                # Try to extract finish position
                                if match[2].isdigit():
                                    finish_pos = int(match[2])
                                
                                # Try to extract odds
                                if len(match) > 3 and match[3].replace('.', '').isdigit():
                                    odds = float(match[3])
                            
                            # Look for trouble indicators
                            trouble = []
                            for indicator in self.trouble_indicators:
                                if re.search(rf'\b{indicator}\b', line, re.IGNORECASE):
                                    trouble.append(indicator)
                            
                            horse = HorseEntry(
                                horse_name=horse_name,
                                figure=figure,
                                track_surface='dirt',  # Default, will be updated
                                race_date='',  # Will be updated
                                race_number=0,  # Will be updated
                                finish_position=finish_pos,
                                trouble_indicators=trouble,
                                weather_conditions=None,
                                distance=None,
                                odds=odds,
                                trainer=trainer,
                                jockey=jockey,
                                weight=None,
                                speed_rating=None,
                                pace_rating=None,
                                class_rating=None
                            )
                            
                            horses.append(horse)
                            break
                    
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing horse entry: {e}")
                        continue
        
        return horses
    
    def parse_ragozin_sheet(self, pdf_path: str) -> RaceData:
        """
        Main method to parse a complete Ragozin sheet
        """
        logger.info(f"Starting to parse Ragozin sheet: {pdf_path}")
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            raise ValueError(f"Could not extract text from PDF: {pdf_path}")
        
        # Parse race header information
        header_info = self.parse_race_header(text)
        
        # Parse horse entries
        horses = self.parse_horse_entries(text)
        
        # Create RaceData object
        race_data = RaceData(
            track_name=header_info['track_name'] or "Unknown Track",
            race_date=header_info['race_date'] or "Unknown Date",
            race_number=header_info['race_number'] or 0,
            surface=header_info['surface'] or "dirt",
            distance=header_info['distance'] or "Unknown",
            weather=header_info['weather'],
            horses=horses,
            race_conditions=None
        )
        
        # Update horse entries with race-level information
        for horse in horses:
            horse.track_surface = race_data.surface
            horse.race_date = race_data.race_date
            horse.race_number = race_data.race_number
            horse.weather_conditions = race_data.weather
            horse.distance = race_data.distance
        
        logger.info(f"Successfully parsed {len(horses)} horses from race")
        return race_data
    
    def export_to_json(self, race_data: RaceData, output_path: str) -> None:
        """
        Export parsed data to JSON format
        """
        # Convert dataclasses to dictionaries
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: v for k, v in obj.__dict__.items()}
            return obj
        
        # Convert horses to dictionaries
        horses_dict = [dataclass_to_dict(horse) for horse in race_data.horses]
        
        # Create output dictionary
        output_data = {
            'track_name': race_data.track_name,
            'race_date': race_data.race_date,
            'race_number': race_data.race_number,
            'surface': race_data.surface,
            'distance': race_data.distance,
            'weather': race_data.weather,
            'race_conditions': race_data.race_conditions,
            'horses': horses_dict
        }
        
        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Exported race data to: {output_path}")
    
    def export_to_csv(self, race_data: RaceData, output_path: str) -> None:
        """
        Export parsed data to CSV format
        """
        # Convert horses to list of dictionaries
        horses_data = []
        for horse in race_data.horses:
            horse_dict = {
                'horse_name': horse.horse_name,
                'figure': horse.figure,
                'track_surface': horse.track_surface,
                'race_date': horse.race_date,
                'race_number': horse.race_number,
                'finish_position': horse.finish_position,
                'trouble_indicators': '; '.join(horse.trouble_indicators) if horse.trouble_indicators else None,
                'weather_conditions': horse.weather_conditions,
                'distance': horse.distance,
                'odds': horse.odds,
                'trainer': horse.trainer,
                'jockey': horse.jockey,
                'weight': horse.weight,
                'speed_rating': horse.speed_rating,
                'pace_rating': horse.pace_rating,
                'class_rating': horse.class_rating
            }
            horses_data.append(horse_dict)
        
        # Create DataFrame and export
        df = pd.DataFrame(horses_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported race data to: {output_path}")

def main():
    """
    Example usage of the Ragozin parser
    """
    parser = RagozinParser()
    
    # Example usage (replace with actual PDF path)
    pdf_path = "example_ragozin_sheet.pdf"
    
    try:
        # Parse the sheet
        race_data = parser.parse_ragozin_sheet(pdf_path)
        
        # Export to different formats
        parser.export_to_json(race_data, "race_data.json")
        parser.export_to_csv(race_data, "race_data.csv")
        
        print(f"Successfully parsed race with {len(race_data.horses)} horses")
        print(f"Track: {race_data.track_name}")
        print(f"Date: {race_data.race_date}")
        print(f"Surface: {race_data.surface}")
        
    except FileNotFoundError:
        print(f"PDF file not found: {pdf_path}")
    except Exception as e:
        print(f"Error parsing PDF: {e}")

if __name__ == "__main__":
    main()