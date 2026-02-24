import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import numpy as np
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import date
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
    age: int = 0
    past_performances: List[Dict] = field(default_factory=list)

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


# ---------------------------------------------------------------------------
# Ragozin figure-line regex
# Matches optional prefix symbols, a number, and optional suffix (+, -, ")
# Examples: "20", "17-", ".14\"", "F18\"", "^=18", "gF*18-", "P~=14\""
# ---------------------------------------------------------------------------
_FIGURE_RE = re.compile(
    r'^'
    r'([^0-9]*?)'           # group 1: prefix symbols (., ^, ~, F, P~=, g, r, etc.)
    r'(\d+(?:\.\d+)?)'      # group 2: the numeric figure
    r'\s*([+\-"]*)'         # group 3: suffix (+, -, ", or combinations)
    r'\s*$'
)

# Year-summary line:  "N  RACES  YY" or "N  RACE  YY"
_YEAR_SUMMARY_RE = re.compile(r'^\s*(\d+)\s+RACES?\s+(\d{2})\s*$')

# Page marker: "CD  pN" where CD is the track code
_PAGE_MARKER_RE = re.compile(r'^([A-Z]{2,4})\s+p(\d+)\s*$')

# Horse name + sex + foal-year line: "HORSE NAME      F  23 "
# Name is all-caps possibly with spaces, hyphens, apostrophes (straight or curly),
# periods, -GB etc.  Sex is M or F, last field is 2-digit foal year (or age if < 10).
_HORSE_LINE_RE = re.compile(
    r"^([A-Z][A-Z\s'\u2018\u2019\.\-]+?)\s{2,}([FM])\s+(\d{1,2})\s*$"
)


def _foal_year_to_age(raw_val: int) -> int:
    """Convert Ragozin foal-year notation to racing age.

    Values >= 10 are 2-digit foal years (e.g. 23 = born 2023).
    Values < 10 are assumed to already be the age.
    """
    if raw_val >= 10:
        return date.today().year - (2000 + raw_val)
    return raw_val

# Post + Race line: "POST  Race N"
# Post can be digits + optional letter suffix (1T, 5, 1116, etc.)
_RACE_LINE_RE = re.compile(r'^\s*(\S+)\s+Race\s+(\d+)\s*$')

# Conditions line: "F/M  4YO  25JUN6062 SMA" or "MALE  3YO  25JUN6062 JBr"
_CONDITIONS_RE = re.compile(
    r'^\s*(F/M|MALE|F&M|M&F)\s+(\d+)YO\s+(\S+)\s+(\S+)\s*$'
)

# Sire line: all-caps, may include apostrophes (straight or curly), hyphens, -GB etc.
_SIRE_RE = re.compile(r"^[A-Z][A-Z\s'\u2018\u2019\-\.]+(?:-[A-Z]{2,3})?\s*$")

# Dam line: DAM-DAMSIRE   STATE+YEAR
_DAM_RE = re.compile(r'^[A-Z].*\s{2,}[A-Z]{2}\d{4}\s*$')

# Detect 2+ consecutive uppercase letters (track/race codes in data lines)
_HAS_TRACK_CODE = re.compile(r'[A-Z]{2,}')


def _is_figure_line(line: str) -> bool:
    """True if line is a Ragozin figure line (not a data/header/noise line)."""
    m = _FIGURE_RE.match(line)
    if not m:
        return False
    prefix = m.group(1)
    fig = float(m.group(2))
    # Reject if prefix contains consecutive uppercase (track codes / horse names)
    if _HAS_TRACK_CODE.search(prefix):
        return False
    # Reject impossible figures (>60)
    if fig > 60:
        return False
    return True


def _parse_figure_line(line: str) -> Optional[Dict]:
    """Extract figure value, prefix symbols, suffix from a figure line."""
    m = _FIGURE_RE.match(line)
    if not m:
        return None
    prefix = m.group(1).strip()
    suffix = m.group(3).strip()
    flags = _symbols_to_flags(prefix, suffix)
    return {
        'raw_text': line,
        'parsed_figure': float(m.group(2)),
        'prefix': prefix,
        'suffix': suffix,
        'flags': flags,
        'surface': 'DIRT',
        'track': '',
        'data_text': '',
    }


def _parse_data_line(line: str) -> Dict:
    """Extract surface and track info from a race data line."""
    m = _HAS_TRACK_CODE.search(line)
    if not m:
        return {'surface': 'DIRT', 'track': '', 'raw': line}
    prefix = line[:m.start()]
    track_code = m.group()
    # Infer surface from prefix chars AND track code
    lc_prefix = prefix.lower()
    tc_upper = track_code.upper()
    if 'AW' in tc_upper or 'aw' in lc_prefix:
        surface = 'POLY'
    elif 't' in lc_prefix:
        surface = 'TURF'
    elif 's' in lc_prefix:
        surface = 'SLOP'
    else:
        surface = 'DIRT'
    return {'surface': surface, 'track': track_code, 'raw': line}


def _symbols_to_flags(prefix: str, suffix: str) -> List[str]:
    """Convert Ragozin prefix/suffix symbols to semantic flag names."""
    flags: List[str] = []
    for c in prefix:
        if c == 'P':
            flags.append('PACE')
        elif c in ('\u2727', '*'):
            flags.append('TOP')
        elif c == '^':
            flags.append('BOUNCE')
        elif c == 'g':
            flags.append('GROUND_SAVE')
        elif c == 'F':
            flags.append('FAST')
        elif c == 'r':
            flags.append('WIDE')
    for c in suffix:
        if c == '+':
            flags.append('PLUS')
        elif c == '-':
            flags.append('MINUS')
        elif c == '"':
            flags.append('KEY')
    return flags


class RagozinParser:
    """
    Parser for Ragozin performance sheets.
    Splits the PDF text on page markers (e.g. "CD  p1") and extracts
    one horse per page with correct race_number from the "Race N" header.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # PDF text extraction (unchanged)
    # ------------------------------------------------------------------

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple methods for reliability."""
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

        # Method 2: pdfplumber fallback
        if not text_content.strip():
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text_content += page.extract_text() or ""
                logger.info(f"Successfully extracted text using pdfplumber from {pdf_path}")
            except Exception as e:
                logger.error(f"pdfplumber failed: {e}")

        return text_content

    # ------------------------------------------------------------------
    # Page splitting
    # ------------------------------------------------------------------

    def _split_into_pages(self, text: str) -> List[Tuple[str, int, str]]:
        """Split full text into (track_code, page_number, section_text) tuples."""
        markers = list(re.finditer(r'^([A-Z]{2,4})\s+p(\d+)\s*$', text, re.MULTILINE))
        if not markers:
            return []

        sections: List[Tuple[str, int, str]] = []
        for i, m in enumerate(markers):
            start = m.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            track_code = m.group(1)
            page_num = int(m.group(2))
            section_text = text[start:end]
            sections.append((track_code, page_num, section_text))

        return sections

    # ------------------------------------------------------------------
    # Per-page parsing
    # ------------------------------------------------------------------

    def _parse_page(self, track_code: str, page_num: int, section: str) -> Optional[HorseEntry]:
        """Parse a single page section into a HorseEntry."""
        lines = [l.strip() for l in section.split('\n') if l.strip()]

        # Skip noise lines (Ragozin header, copyright, timeline labels)
        cleaned: List[str] = []
        for line in lines:
            if line.startswith('Ragozin --') or line.startswith('TM'):
                continue
            if line.startswith('These Sheets are') or line.startswith('copied or'):
                continue
            if line.startswith('The Sheets:'):
                continue
            # Timeline labels: two-letter month abbreviations or bare "6062" etc.
            if re.match(r'^(DE|NO|OC|SE|AU|JL|JU|MA|AP|FE|JA|6062|N|C|V|T|P|G|Y|R|B)$', line):
                continue
            # "top:" summary lines
            if line.startswith('top:') or re.match(r'^spr\s+dst\s+turf$', line):
                continue
            if re.match(r'^\d+yo>\s+', line):
                continue
            # Skip year-marker noise (e.g. "0^6062", "0 ^ 6 0 6 2")
            if re.match(r'^0\s*\^?\s*6\s*0\s*6\s*2\s*$', line):
                continue
            cleaned.append(line)

        # Fix concatenated dam+horse lines:
        # "SENORITA CORREDORA-EL CORREDOR   KY6062CAROLINA CANDY      F  21"
        # "BECKY'S BLUEGRASS-MACHO UNO   KY6062 BELLA BLUEGRASS      F  22"
        # Split at STATE+YEAR boundary when followed by a horse name pattern.
        expanded: List[str] = []
        for line in cleaned:
            split_m = re.search(r'([A-Z]{2}\d{4})\s*([A-Z])', line)
            if split_m:
                before = line[:split_m.end(1)]
                after = line[split_m.end(1):].strip()
                if _HORSE_LINE_RE.match(after):
                    expanded.append(before)
                    expanded.append(after)
                    continue
            expanded.append(line)
        cleaned = expanded

        if len(cleaned) < 4:
            logger.debug(f"Page {page_num}: too few lines after cleanup ({len(cleaned)})")
            return None

        # Expected order after cleanup:
        #   [0] Sire
        #   [1] Dam-Damsire  STATE+YEAR
        #   [2] HORSE NAME      Sex  Age
        #   [3] POST  Race N
        #   [4] Conditions line
        #   [5+] Race lines

        horse_name = None
        sex = None
        age = None
        race_number = 0
        post = ''
        sire = ''
        dam_line = ''
        conditions = ''
        trainer_code = ''
        race_date_raw = ''

        # Scan first 8 lines for the structured header
        for i, line in enumerate(cleaned[:8]):
            # Try horse name line
            hm = _HORSE_LINE_RE.match(line)
            if hm:
                horse_name = hm.group(1).strip()
                sex = hm.group(2)
                age = _foal_year_to_age(int(hm.group(3)))
                # Sire is the line before dam, dam is the line before horse
                if i >= 2:
                    sire = cleaned[i - 2] if _SIRE_RE.match(cleaned[i - 2]) else ''
                    dam_line = cleaned[i - 1]
                elif i >= 1:
                    dam_line = cleaned[i - 1]
                continue

            # Try Race line
            rm = _RACE_LINE_RE.match(line)
            if rm and horse_name:  # only match after we've found the horse name
                post = rm.group(1)
                race_number = int(rm.group(2))
                continue

            # Try conditions line
            cm = _CONDITIONS_RE.match(line)
            if cm and horse_name:
                race_date_raw = cm.group(3)
                trainer_code = cm.group(4)
                conditions = line
                continue

        if not horse_name:
            logger.debug(f"Page {page_num}: could not find horse name")
            return None

        # Parse race date from raw (e.g. "25JUN6062" -> "06/25")
        race_date = self._parse_race_date(race_date_raw)

        # --- Performance-line pairing (figure + data) ---
        # Find where performance zone starts (after the header lines)
        header_end = 0
        for i, line in enumerate(cleaned[:10]):
            if _CONDITIONS_RE.match(line) or (_RACE_LINE_RE.match(line) and horse_name):
                header_end = max(header_end, i + 1)

        past_performances: List[Dict] = []
        pending_figure: Optional[Dict] = None

        for line in cleaned[header_end:]:
            # Stop at chart footer
            if line.startswith('Ragozin --') or line == 'TM':
                break
            # Skip year-summary and year-marker noise
            if _YEAR_SUMMARY_RE.match(line):
                continue
            if re.match(r'^0\s*\^?\s*6\s*0\s*6\s*2', line):
                continue

            # Try as figure line
            if _is_figure_line(line):
                if pending_figure is not None:
                    past_performances.append(pending_figure)
                pending_figure = _parse_figure_line(line)
                continue

            # Try as data line (pair with pending figure)
            if pending_figure is not None and _HAS_TRACK_CODE.search(line):
                data = _parse_data_line(line)
                pending_figure['surface'] = data['surface']
                pending_figure['track'] = data.get('track', '')
                pending_figure['data_text'] = line
                past_performances.append(pending_figure)
                pending_figure = None
                continue

            # Otherwise: annotation (GELDED, etc.) or noise — skip

        # Trailing unpaired figure
        if pending_figure is not None:
            past_performances.append(pending_figure)

        # Best (lowest) figure from past performances
        fig_vals = [pp['parsed_figure'] for pp in past_performances
                    if pp.get('parsed_figure') is not None]
        top_figure = min(fig_vals) if fig_vals else None

        entry = HorseEntry(
            horse_name=horse_name,
            figure=top_figure,
            track_surface='dirt',
            race_date=race_date,
            race_number=race_number,
            finish_position=None,
            trouble_indicators=[],
            weather_conditions=None,
            distance=None,
            odds=None,
            trainer=trainer_code if trainer_code else None,
            jockey=None,
            weight=None,
            speed_rating=None,
            pace_rating=None,
            class_rating=None,
            age=age if age is not None else 0,
            past_performances=past_performances,
        )
        return entry

    @staticmethod
    def _parse_race_date(raw: str) -> str:
        """Convert '25JUN6062' -> '06/25/2025' (month/day/year)."""
        if not raw:
            return ''
        m = re.match(r'(\d{1,2})([A-Z]{3})(\d{4})', raw)
        if not m:
            return raw
        day = m.group(1)
        month_abbr = m.group(2)
        year_raw = m.group(3)
        months = {
            'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12',
        }
        month_num = months.get(month_abbr, '00')
        # Ragozin sheets use a proprietary font that renders years oddly.
        # "6062" is the rendering of "2025" in Ragozin font encoding.
        # Map known values; fall back to raw if unknown.
        ragozin_year_map = {'6062': '2025'}
        year = ragozin_year_map.get(year_raw, year_raw)
        return f"{month_num}/{day}/{year}"

    # ------------------------------------------------------------------
    # Main parse entry point
    # ------------------------------------------------------------------

    def parse_ragozin_sheet(self, pdf_path: str) -> RaceData:
        """Parse a complete Ragozin sheet PDF into structured data."""
        logger.info(f"Starting to parse Ragozin sheet: {pdf_path}")

        text = self.extract_text_from_pdf(pdf_path)
        if not text.strip():
            raise ValueError(f"Could not extract text from PDF: {pdf_path}")

        sections = self._split_into_pages(text)
        if not sections:
            logger.warning("No page markers found, falling back to legacy parsing")
            return self._legacy_parse(text)

        track_code = sections[0][0] if sections else 'UNK'

        horses: List[HorseEntry] = []
        for tc, page_num, section_text in sections:
            entry = self._parse_page(tc, page_num, section_text)
            if entry:
                horses.append(entry)

        # Derive overall session metadata
        race_numbers = sorted(set(h.race_number for h in horses if h.race_number > 0))
        race_dates = [h.race_date for h in horses if h.race_date]
        overall_date = race_dates[0] if race_dates else 'Unknown Date'

        # Track code mapping
        track_names = {
            'CD': 'Churchill Downs', 'SA': 'Santa Anita', 'GP': 'Gulfstream Park',
            'AQ': 'Aqueduct', 'BEL': 'Belmont', 'SAR': 'Saratoga',
            'KEE': 'Keeneland', 'DM': 'Del Mar', 'OP': 'Oaklawn Park',
            'TAM': 'Tampa Bay Downs', 'FG': 'Fair Grounds',
        }
        track_name = track_names.get(track_code, track_code)

        race_data = RaceData(
            track_name=track_name,
            race_date=overall_date,
            race_number=0,  # multi-race session
            surface='dirt',
            distance='Unknown',
            weather=None,
            horses=horses,
            race_conditions=None,
        )

        logger.info(
            f"Parsed {len(horses)} horses across races "
            f"{race_numbers} from {len(sections)} pages"
        )
        return race_data

    # ------------------------------------------------------------------
    # Legacy fallback (old blob-regex approach, kept as safety net)
    # ------------------------------------------------------------------

    def _legacy_parse(self, text: str) -> RaceData:
        """Fallback parser for non-Ragozin PDFs that lack page markers."""
        logger.warning("Using legacy blob parser (no page markers found)")
        horses: List[HorseEntry] = []
        # Minimal: just return an empty RaceData so callers don't crash
        return RaceData(
            track_name='Unknown Track',
            race_date='Unknown Date',
            race_number=0,
            surface='dirt',
            distance='Unknown',
            weather=None,
            horses=horses,
            race_conditions=None,
        )

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def export_to_json(self, race_data: RaceData, output_path: str) -> None:
        """Export parsed data to JSON format compatible with frontend."""
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
                'sex': horse.track_surface,  # placeholder
                'age': horse.age,
                'breeder_owner': 'Unknown',
                'foal_date': 'Unknown',
                'reg_code': 'Unknown',
                'races': 1,
                'top_fig': str(horse.figure) if horse.figure is not None else '',
                'horse_analysis': f"Ragozin Figure: {horse.figure}, Surface: {horse.track_surface}, Distance: {horse.distance}",
                'performance_trend': 'No trend data available',
                'lines': [race_line],
            }
            horses_dict.append(horse_dict)

        output_data = {
            'track_name': race_data.track_name,
            'race_date': race_data.race_date,
            'race_number': race_data.race_number,
            'surface': race_data.surface,
            'distance': race_data.distance,
            'weather': race_data.weather,
            'race_conditions': race_data.race_conditions,
            'horses': horses_dict,
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Exported race data to: {output_path}")

    def export_to_csv(self, race_data: RaceData, output_path: str) -> None:
        """Export parsed data to CSV format."""
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
                'class_rating': horse.class_rating,
            }
            horses_data.append(horse_dict)

        df = pd.DataFrame(horses_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported race data to: {output_path}")


def main():
    """Example usage of the Ragozin parser."""
    parser = RagozinParser()
    pdf_path = "cd062525.pdf"

    try:
        race_data = parser.parse_ragozin_sheet(pdf_path)
        parser.export_to_json(race_data, "race_data.json")
        parser.export_to_csv(race_data, "race_data.csv")

        print(f"Successfully parsed {len(race_data.horses)} horses")
        print(f"Track: {race_data.track_name}")
        print(f"Date: {race_data.race_date}")

        # Show per-race breakdown
        from collections import Counter
        race_counts = Counter(h.race_number for h in race_data.horses)
        for rn in sorted(race_counts):
            print(f"  Race {rn}: {race_counts[rn]} horses")

    except FileNotFoundError:
        print(f"PDF file not found: {pdf_path}")
    except Exception as e:
        print(f"Error parsing PDF: {e}")

if __name__ == "__main__":
    main()
