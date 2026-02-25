"""Equibase Results Chart PDF parser.

Extracts race results from Equibase-style chart PDFs including starters,
finish positions, odds, and payoff data.

Handles common variations:
- Program numbers: numeric (1), alpha-coupled (1A, 1X), AE, MTO
- Dead heats and coupled entries (1/1A)
- Scratched entries
- Payoff blocks: $2 Mutuel Prices, $2 Exacta, $1 Trifecta, Daily Double, etc.
- Surface/distance lines above or below race header

Usage:
    parser = ResultsChartParser()
    result = parser.parse("path/to/chart.pdf")
    rows = chart_to_rows(result)
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChartEntry:
    """Single starter in a race."""
    program: str = ""         # raw program number: "1", "1A", "AE", "MTO"
    post: int = 0             # numeric post position (0 if unparseable)
    horse_name: str = ""
    finish_pos: int = 0
    odds: Optional[float] = None
    win_payoff: Optional[float] = None
    place_payoff: Optional[float] = None
    show_payoff: Optional[float] = None
    beaten_lengths: Optional[float] = None
    scratched: bool = False
    coupled_with: str = ""    # e.g. "1" if this is entry "1A"
    dead_heat: bool = False


@dataclass
class ChartRace:
    """One race on the card."""
    race_number: int = 0
    surface: str = ""
    distance: str = ""
    entries: List[ChartEntry] = field(default_factory=list)


@dataclass
class ChartCard:
    """Full card of races."""
    track: str = ""
    race_date: str = ""
    races: List[ChartRace] = field(default_factory=list)


@dataclass
class ParseResult:
    """Result of parsing including confidence, missing fields, and raw text."""
    card: ChartCard = field(default_factory=ChartCard)
    confidence: float = 0.0
    raw_text: str = ""
    missing_fields: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Race header: "FIRST RACE" through "FOURTEENTH RACE" or "Race N"
_ORDINALS = (
    "FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|"
    "TENTH|ELEVENTH|TWELFTH|THIRTEENTH|FOURTEENTH"
)
_RACE_HEADER_ORDINAL = re.compile(
    rf"({_ORDINALS})\s+RACE", re.IGNORECASE
)
_RACE_HEADER_NUMERIC = re.compile(r"Race\s+(\d{1,2})", re.IGNORECASE)

_ORDINAL_MAP = {
    "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5,
    "SIXTH": 6, "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9, "TENTH": 10,
    "ELEVENTH": 11, "TWELFTH": 12, "THIRTEENTH": 13, "FOURTEENTH": 14,
}

# Program number: numeric, alpha-coupled (1A, 1X), or special (AE, MTO)
_PGM_RE = re.compile(r"^(\d{1,2}[A-Za-z]?|AE|MTO)$")

# Starters line: pgm  HORSE NAME  ...  finish  odds
# Supports alpha program numbers like "1A"
_STARTER_RE = re.compile(
    r"^\s*(\d{1,2}[A-Za-z]?)"             # program number (1, 1A, etc.)
    r"\s+((?:[A-Z][A-Za-z'.\-]+\s*)+?)"   # horse name (one or more words)
    r"\s+.*?"                               # jockey/weight/other columns
    r"(\d{1,2})\s*(?:st|nd|rd|th)?"        # finish position
    r"\s+(\*?[\d]+\.[\d]+)"                # odds (optional * for favorite)
    r"\s*$",
    re.MULTILINE,
)

# Alternate starter format with clear column positions
_STARTER_ALT_RE = re.compile(
    r"^\s*(\d{1,2}[A-Za-z]?)"             # program number
    r"\s+([A-Z][A-Za-z'\.\-\s]{2,30}?)"   # horse name
    r"\s+[A-Z][a-z].*?"                    # jockey (mixed case)
    r"\b(\d{1,2})\s*(?:st|nd|rd|th|nose|head|nk|hd|½|¾)?"  # finish
    r"[^\d]*?"
    r"(\*?[\d]+\.[\d]+)"                   # odds
    r"\s*$",
    re.MULTILINE,
)

# Scratched entry inline: "pgm  HORSE NAME  ...  Scratched|(S)|SCR"
_SCRATCH_INLINE_RE = re.compile(
    r"^\s*(\d{1,2}[A-Za-z]?)"             # program number
    r"\s+([A-Z][A-Za-z'\.\-\s]{2,30}?)"   # horse name
    r"\s+.*?"
    r"(?:Scratched|\(S\)|SCR)",            # scratch marker
    re.MULTILINE | re.IGNORECASE,
)

# Scratched list: "Scratched: 3 - LUCKY PENNY (reason), 5 - BLUE HORIZON (reason)"
_SCRATCH_LIST_RE = re.compile(
    r"(\d{1,2}[A-Za-z]?)\s*[-–—]\s*([A-Z][A-Z'\.\-\s]+?)(?:\s*\([^)]*\))?\s*(?:,|$)",
)

# Dead heat marker
_DEAD_HEAT_RE = re.compile(r"Dead\s+Heat", re.IGNORECASE)

# Payoff header: "$2 Mutuel Prices" (the WPS payoff section we care about)
_PAYOFF_HEADER_RE = re.compile(
    r"\$2\s+Mutuel\s+Prices", re.IGNORECASE
)

# Exotic payoff headers to skip past (stop WPS payoff extraction here)
_EXOTIC_HEADER_RE = re.compile(
    r"\$[12]\s+(?:Exacta|Trifecta|Superfecta|Pick|Quinella|Daily\s+Double|Consolation)",
    re.IGNORECASE,
)

# Payoff line: pgm - HORSE NAME   win  place  show
_PAYOFF_RE = re.compile(
    r"(\d{1,2}[A-Za-z]?)\s*[-–—]\s*"    # program number (supports 1A)
    r"([A-Z][A-Z\s'.\-]+?)"              # horse name (UPPER)
    r"\s{2,}"                             # gap
    r"([\d]+\.[\d]{2})"                   # win payoff
    r"(?:\s+([\d]+\.[\d]{2}))?"           # place payoff (optional)
    r"(?:\s+([\d]+\.[\d]{2}))?"           # show payoff (optional)
)

# Surface/distance patterns
_SURFACE_RE = re.compile(
    r"\b(Dirt|Turf|All Weather|Synthetic|Polytrack|Inner Turf)\b",
    re.IGNORECASE,
)
_DISTANCE_RE = re.compile(
    r"\b(\d+(?:\s+\d+/\d+)?)\s*(Furlongs?|Miles?|Yards?)\b",
    re.IGNORECASE,
)

# Track names (common US tracks)
_TRACK_NAMES = [
    "Gulfstream Park", "Churchill Downs", "Santa Anita Park", "Santa Anita",
    "Belmont Park", "Saratoga", "Keeneland", "Del Mar",
    "Oaklawn Park", "Tampa Bay Downs", "Fair Grounds",
    "Aqueduct", "Pimlico", "Monmouth Park", "Parx Racing",
    "Laurel Park", "Golden Gate Fields", "Woodbine", "Turfway Park",
    "Gulfstream Park West", "Los Alamitos", "Hawthorne", "Arlington Park",
    "Penn National", "Charles Town", "Remington Park", "Lone Star Park",
    "Finger Lakes", "Suffolk Downs", "Mountaineer", "Prairie Meadows",
    "Sam Houston", "Sunland Park", "Turf Paradise", "Canterbury Park",
    "Indiana Grand", "Horseshoe Indianapolis", "Ellis Park",
]
_TRACK_PATTERN = re.compile(
    "|".join(re.escape(t) for t in _TRACK_NAMES), re.IGNORECASE
)

# Date patterns
_DATE_LONG = re.compile(
    r"(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})"
)
_DATE_SLASH = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")

_MONTH_MAP = {
    "January": "01", "February": "02", "March": "03", "April": "04",
    "May": "05", "June": "06", "July": "07", "August": "08",
    "September": "09", "October": "10", "November": "11", "December": "12",
}


def _pgm_to_post(pgm: str) -> int:
    """Extract numeric post from program number. '1A' -> 1, 'AE' -> 0."""
    m = re.match(r"(\d+)", pgm)
    return int(m.group(1)) if m else 0


def _pgm_coupled_with(pgm: str) -> str:
    """If pgm is coupled (e.g. '1A'), return the base ('1'). Else ''."""
    m = re.match(r"^(\d+)[A-Za-z]$", pgm)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ResultsChartParser:
    """Parse Equibase results chart PDFs."""

    def parse(self, pdf_path: str) -> ParseResult:
        """Main entry point: parse PDF and return structured result."""
        raw_text = self._extract_text(pdf_path)
        if not raw_text:
            return ParseResult(raw_text="")

        track, race_date = self._extract_track_date(raw_text)
        race_blocks = self._split_race_blocks(raw_text)

        races = []
        for race_num, block_text in race_blocks:
            race = self._parse_race_block(race_num, block_text)
            if race and race.entries:
                races.append(race)

        card = ChartCard(track=track, race_date=race_date, races=races)
        missing = self._detect_missing_fields(card)
        confidence = self._calc_confidence(card)

        return ParseResult(
            card=card, confidence=confidence,
            raw_text=raw_text, missing_fields=missing,
        )

    def _extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF, page by page."""
        try:
            doc = fitz.open(pdf_path)
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n".join(pages)
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""

    def _extract_track_date(self, text: str) -> tuple:
        """Extract track name and date from first portion of text."""
        header = text[:3000]

        track = ""
        m = _TRACK_PATTERN.search(header)
        if m:
            track = m.group(0).strip()

        race_date = ""
        m = _DATE_LONG.search(header)
        if m:
            month = _MONTH_MAP.get(m.group(1), "01")
            day = m.group(2).zfill(2)
            year = m.group(3)
            race_date = f"{month}/{day}/{year}"
        else:
            m = _DATE_SLASH.search(header)
            if m:
                race_date = f"{m.group(1).zfill(2)}/{m.group(2).zfill(2)}/{m.group(3)}"

        return track, race_date

    def _split_race_blocks(self, text: str) -> list:
        """Find race header positions, return list of (race_num, block_text)."""
        positions = []

        for m in _RACE_HEADER_ORDINAL.finditer(text):
            ordinal = m.group(1).upper()
            race_num = _ORDINAL_MAP.get(ordinal, 0)
            if race_num:
                positions.append((race_num, m.start()))

        for m in _RACE_HEADER_NUMERIC.finditer(text):
            race_num = int(m.group(1))
            existing_nums = {p[0] for p in positions}
            if race_num not in existing_nums:
                positions.append((race_num, m.start()))

        positions.sort(key=lambda x: x[1])

        blocks = []
        for i, (race_num, start) in enumerate(positions):
            end = positions[i + 1][1] if i + 1 < len(positions) else len(text)
            blocks.append((race_num, text[start:end]))

        return blocks

    def _parse_race_block(self, race_num: int, text: str) -> Optional[ChartRace]:
        """Extract starters, scratches, and payoffs from a race block."""
        race = ChartRace(race_number=race_num)

        # Extract surface — search wider area (above/below header)
        sm = _SURFACE_RE.search(text[:800])
        if sm:
            raw = sm.group(1).lower()
            if "turf" in raw:
                race.surface = "Turf"
            elif raw == "dirt":
                race.surface = "Dirt"
            elif raw in ("all weather", "synthetic", "polytrack"):
                race.surface = "Poly"
            else:
                race.surface = sm.group(1).title()

        # Extract distance — search wider area
        dm = _DISTANCE_RE.search(text[:800])
        if dm:
            race.distance = f"{dm.group(1)} {dm.group(2)}"

        # Check for dead heat anywhere in block
        has_dead_heat = bool(_DEAD_HEAT_RE.search(text))

        # Extract starters
        entries_by_pgm = {}

        for m in _STARTER_RE.finditer(text):
            pgm = m.group(1).strip()
            name = m.group(2).strip()
            finish = int(m.group(3))
            odds_str = m.group(4).replace("*", "")
            try:
                odds = float(odds_str)
            except ValueError:
                odds = None

            post = _pgm_to_post(pgm)
            coupled = _pgm_coupled_with(pgm)

            if pgm not in entries_by_pgm or finish < entries_by_pgm[pgm].finish_pos:
                entries_by_pgm[pgm] = ChartEntry(
                    program=pgm, post=post, horse_name=name,
                    finish_pos=finish, odds=odds,
                    coupled_with=coupled,
                )

        # Try alternate regex if primary found nothing
        if not entries_by_pgm:
            for m in _STARTER_ALT_RE.finditer(text):
                pgm = m.group(1).strip()
                name = m.group(2).strip()
                finish = int(m.group(3))
                odds_str = m.group(4).replace("*", "")
                try:
                    odds = float(odds_str)
                except ValueError:
                    odds = None

                post = _pgm_to_post(pgm)
                coupled = _pgm_coupled_with(pgm)

                if pgm not in entries_by_pgm:
                    entries_by_pgm[pgm] = ChartEntry(
                        program=pgm, post=post, horse_name=name,
                        finish_pos=finish, odds=odds,
                        coupled_with=coupled,
                    )

        # Detect scratched entries — inline format ("pgm NAME ... Scratched")
        for m in _SCRATCH_INLINE_RE.finditer(text):
            pgm = m.group(1).strip()
            name = m.group(2).strip()
            post = _pgm_to_post(pgm)
            coupled = _pgm_coupled_with(pgm)
            if pgm not in entries_by_pgm:
                entries_by_pgm[pgm] = ChartEntry(
                    program=pgm, post=post, horse_name=name,
                    scratched=True, coupled_with=coupled,
                )

        # Detect scratched entries — list format ("Scratched: 3 - NAME, 5 - NAME")
        scratch_list_m = re.search(r"Scratched\s*:", text, re.IGNORECASE)
        if scratch_list_m:
            scratch_line = text[scratch_list_m.end():scratch_list_m.end() + 500]
            # Stop at newline that doesn't continue the list
            newline_stop = re.search(r"\n\s*\n", scratch_line)
            if newline_stop:
                scratch_line = scratch_line[:newline_stop.start()]
            for m in _SCRATCH_LIST_RE.finditer(scratch_line):
                pgm = m.group(1).strip()
                name = m.group(2).strip()
                post = _pgm_to_post(pgm)
                coupled = _pgm_coupled_with(pgm)
                if pgm not in entries_by_pgm:
                    entries_by_pgm[pgm] = ChartEntry(
                        program=pgm, post=post, horse_name=name,
                        scratched=True, coupled_with=coupled,
                    )

        # Mark dead heats: if two entries share the same finish position
        if has_dead_heat:
            finish_counts = {}
            for e in entries_by_pgm.values():
                if e.finish_pos > 0 and not e.scratched:
                    finish_counts.setdefault(e.finish_pos, []).append(e)
            for pos, entries_list in finish_counts.items():
                if len(entries_list) > 1:
                    for e in entries_list:
                        e.dead_heat = True

        # Extract WPS payoffs (stop at exotic headers)
        payoff_start = _PAYOFF_HEADER_RE.search(text)
        if payoff_start:
            payoff_text = text[payoff_start.start():]
            # Truncate at first exotic header to avoid mismatching
            exotic_m = _EXOTIC_HEADER_RE.search(payoff_text[20:])  # skip past "$2 Mutuel" itself
            if exotic_m:
                payoff_text = payoff_text[:20 + exotic_m.start()]

            for m in _PAYOFF_RE.finditer(payoff_text):
                pgm = m.group(1).strip()
                win = float(m.group(3))
                place = float(m.group(4)) if m.group(4) else None
                show = float(m.group(5)) if m.group(5) else None

                if pgm in entries_by_pgm:
                    entries_by_pgm[pgm].win_payoff = win
                    entries_by_pgm[pgm].place_payoff = place
                    entries_by_pgm[pgm].show_payoff = show

        race.entries = sorted(entries_by_pgm.values(), key=lambda e: (e.post, e.program))
        return race

    def _detect_missing_fields(self, card: ChartCard) -> list:
        """Return list of field-level issues detected in the parse."""
        missing = []
        if not card.track:
            missing.append("track: not detected")
        if not card.race_date:
            missing.append("race_date: not detected")
        if not card.races:
            missing.append("races: none found")
            return missing

        all_entries = [e for r in card.races for e in r.entries if not e.scratched]
        if not all_entries:
            missing.append("entries: none found (all scratched?)")
            return missing

        no_finish = [e for e in all_entries if e.finish_pos <= 0]
        if no_finish:
            missing.append(f"finish_pos: missing on {len(no_finish)}/{len(all_entries)} entries")

        no_odds = [e for e in all_entries if e.odds is None]
        if no_odds:
            missing.append(f"odds: missing on {len(no_odds)}/{len(all_entries)} entries")

        # Check payoffs on top-3 finishers
        top3 = [e for e in all_entries if 1 <= e.finish_pos <= 3]
        no_payoff = [e for e in top3
                     if e.win_payoff is None and e.place_payoff is None and e.show_payoff is None]
        if no_payoff:
            missing.append(f"payoffs: missing on {len(no_payoff)}/{len(top3)} top-3 finishers")

        for race in card.races:
            if not race.surface:
                missing.append(f"surface: missing on race {race.race_number}")
            if not race.distance:
                missing.append(f"distance: missing on race {race.race_number}")

        return missing

    def _calc_confidence(self, card: ChartCard) -> float:
        """Weighted confidence score from field completeness (0.0–1.0)."""
        if not card.races:
            return 0.0

        # Race count component (weight 0.20): perfect at 5+
        race_count = len(card.races)
        race_score = min(race_count / 5.0, 1.0)

        # Exclude scratches from entry analysis
        all_entries = [e for r in card.races for e in r.entries if not e.scratched]
        if not all_entries:
            return race_score * 0.20

        # Entry density component (weight 0.15): perfect at 6+ per race
        avg_entries = len(all_entries) / max(race_count, 1)
        density_score = min(avg_entries / 6.0, 1.0)

        # Finish rate component (weight 0.30): % of entries with finish_pos
        with_finish = sum(1 for e in all_entries if e.finish_pos > 0)
        finish_score = with_finish / len(all_entries)

        # Odds rate component (weight 0.20): % of entries with odds
        with_odds = sum(1 for e in all_entries if e.odds is not None)
        odds_score = with_odds / len(all_entries)

        # Payoff rate component (weight 0.15): top-3 finishers have payoffs
        top3_count = 0
        top3_with_payoff = 0
        for race in card.races:
            for e in race.entries:
                if e.scratched:
                    continue
                if 1 <= e.finish_pos <= 3:
                    top3_count += 1
                    if e.win_payoff is not None or e.place_payoff is not None or e.show_payoff is not None:
                        top3_with_payoff += 1
        payoff_score = (top3_with_payoff / top3_count) if top3_count > 0 else 0.0

        confidence = (
            race_score * 0.20
            + density_score * 0.15
            + finish_score * 0.30
            + odds_score * 0.20
            + payoff_score * 0.15
        )
        return round(confidence, 3)


# ---------------------------------------------------------------------------
# Converter: ParseResult -> flat row dicts (matching CSV schema)
# ---------------------------------------------------------------------------

def chart_to_rows(result: ParseResult) -> list:
    """Convert ParseResult to list of row dicts matching results CSV schema.

    Scratched entries are included with finish_pos=0 so the UI can display
    them, but they will be skipped during DB ingestion.
    """
    rows = []
    card = result.card
    for race in card.races:
        for entry in race.entries:
            rows.append({
                "race_number": race.race_number,
                "program": entry.program,
                "post": entry.post,
                "horse_name": entry.horse_name,
                "finish_pos": entry.finish_pos,
                "odds": entry.odds,
                "win_payoff": entry.win_payoff,
                "place_payoff": entry.place_payoff,
                "show_payoff": entry.show_payoff,
                "surface": race.surface,
                "distance": race.distance,
                "beaten_lengths": entry.beaten_lengths,
                "scratched": entry.scratched,
                "coupled_with": entry.coupled_with,
                "dead_heat": entry.dead_heat,
            })
    return rows
