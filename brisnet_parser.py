"""BRISNET Ultimate PP's w/ QuickPlay Comments parser."""
from __future__ import annotations

import fitz
import re
import json
import argparse
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Horse block anchor: bare post number line followed by Name (Style N)
# Name must NOT cross newlines (use [ ] instead of \s to avoid matching \n)
_HORSE_BLOCK_RE = re.compile(
    r'^(\d{1,2})\n'
    r"([\w '.,-]+?)\s*"
    r'\(([ESP/]+)\s+(\d+)\)',
    re.MULTILINE,
)

_PRIME_POWER_RE = re.compile(r'Prime Power:\s*([\d.]+)\s*\((\w+)\)')
_LIFE_RE = re.compile(
    r'Life:\s*\n\s*(\d+)\s*\n\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*\n\s*\$([\d,]+)\s*(\d*)'
)
_SURFACE_REC_RE = re.compile(
    r'(Fst|Off|Dis|Trf|AW)\s*(?:\([^)]*\))?\s*\n\s*(\d+)\s*\n\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*\n\s*\$([\d,]+)\s*\n?\s*(\d*)'
)
_OWNER_RE = re.compile(r'^Own:\s*(.+)', re.MULTILINE)
_TRAINER_RE = re.compile(r'Trnr:\s*\n\s*(.+?)\s*\((.+?)\)', re.DOTALL)
_JOCKEY_RE = re.compile(
    r'^([A-Z][A-Z,.\s]+?)\s+\((\d+\s+[\d\-]+\s+\d+%)\)',
    re.MULTILINE,
)
_SIRE_RE = re.compile(r'^Sire\s*:\s*(.+)', re.MULTILINE)
_DAM_RE = re.compile(r'^Dam:\s*(.+)', re.MULTILINE)
_BREEDER_RE = re.compile(r'^Brdr:\s*(.+)', re.MULTILINE)
_BREED_RE = re.compile(r'^\$[\d,]+\s+(.+)', re.MULTILINE)
_WEIGHT_RE = re.compile(r'^L\s+(\d+)', re.MULTILINE)

# Running-line date-track anchor: 04Dec25GP¨ or 31Jan26GP¨
_RL_DATE_RE = re.compile(r'^(\d{2})([A-Z][a-z]{2})(\d{2})(\w{2,3})', re.MULTILINE)

# Speed figure line within running line: SPD PP ST (e.g. "72 7 1" or "76 8 6ªƒ")
_SPD_LINE_RE = re.compile(r'^(\d{2,3})\s+(\d{1,2})\s+(\d)', re.MULTILINE)

# Workout pattern: date track dist surface time rank (multiple per line)
_WORKOUT_RE = re.compile(
    r"(\d{1,2}[A-Z][a-z]{2})(?:'(\d{2}))?\s+"
    r"(\w{2,3})\s+"
    r"(\d+f|[^:]+?f)\s+"
    r"(ft|fm|gd|sy|my|yl|wd)\s+"
    r"([\d:]+\S*)\s+"
    r"(\w+)\s+"
    r"(\d+/\d+)"
)

# Race conditions line (starts with ™ or a class abbreviation like MC, Clm, OC, Alw, Stk)
_CONDITIONS_LINE_RE = re.compile(
    r'^(™(?:MC|Clm|OC|Alw|Stk|AOC|WCL|MSW|Md|Hcp).+)$', re.MULTILINE
)
_CONDITIONS_LINE_NOTM_RE = re.compile(
    r'^((?:MC|Clm|OC|Alw|Stk|AOC|WCL|MSW|Md|Hcp)\s.+)$', re.MULTILINE
)

# Race number
_RACE_NUM_RE = re.compile(r'^Race\s+(\d+)\s*$', re.MULTILINE)

# QuickPlay markers
_QP_POSITIVE = '\u00f1'  # ñ
_QP_NEGATIVE = '\u00d7'  # ×


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BrisnetHorse:
    post: int = 0
    name: str = ''
    runstyle: str = ''
    runstyle_rating: int = 0
    prime_power: Optional[float] = None
    prime_power_rank: str = ''
    class_rating: Optional[float] = None
    last_speed: Optional[int] = None
    best_dist_speed: Optional[int] = None
    owner: str = ''
    odds: str = ''
    jockey: str = ''
    jockey_stats: str = ''
    trainer: str = ''
    trainer_stats: str = ''
    breed_info: str = ''
    sire: str = ''
    dam: str = ''
    breeder: str = ''
    weight: Optional[int] = None
    life_starts: int = 0
    life_record: str = ''
    life_earnings: str = ''
    life_speed: Optional[int] = None
    surface_records: Dict[str, Dict] = field(default_factory=dict)
    quickplay_positive: List[str] = field(default_factory=list)
    quickplay_negative: List[str] = field(default_factory=list)
    running_lines: List[Dict] = field(default_factory=list)
    workouts: List[Dict] = field(default_factory=list)
    raw_block: str = ''


@dataclass
class BrisnetRace:
    track: str = ''
    date: str = ''
    race_number: int = 0
    conditions_raw: str = ''
    surface: str = ''
    distance: str = ''
    race_class: str = ''
    purse: str = ''
    restrictions: str = ''
    pars: Dict[str, Any] = field(default_factory=dict)
    horses: List[BrisnetHorse] = field(default_factory=list)


@dataclass
class BrisnetCard:
    track: str = ''
    date: str = ''
    races: List[BrisnetRace] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class BrisnetParser:
    """Parse BRISNET Ultimate PP's w/ QuickPlay Comments PDFs."""

    def parse(self, pdf_path: str) -> BrisnetCard:
        """Main entry point. Returns a BrisnetCard with all races and horses."""
        logger.info(f"Parsing BRISNET PDF: {pdf_path}")
        pages = self._extract_text_pages(pdf_path)
        race_pages = self._group_pages_by_race(pages)

        # Extract track and date from first page
        track = 'Unknown'
        date = ''
        if pages:
            m = re.search(r"QuickPlay Comments\s+(.+?)$", pages[0], re.MULTILINE)
            if m:
                track = m.group(1).strip()
            m2 = re.search(
                r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(.+?\d{4})',
                pages[0],
            )
            if m2:
                date = m2.group(2).strip()

        races: List[BrisnetRace] = []
        for rn in sorted(race_pages.keys()):
            pg_texts = race_pages[rn]
            race = self._parse_race(rn, pg_texts, track, date)
            races.append(race)

        card = BrisnetCard(track=track, date=date, races=races)
        total_horses = sum(len(r.horses) for r in races)
        logger.info(f"Parsed {len(races)} races, {total_horses} horses from {len(pages)} pages")
        return card

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_text_pages(self, pdf_path: str) -> List[str]:
        doc = fitz.open(pdf_path)
        pages = [doc[i].get_text() for i in range(len(doc))]
        doc.close()
        return pages

    def _group_pages_by_race(self, pages: List[str]) -> Dict[int, List[str]]:
        groups: Dict[int, List[str]] = {}
        for pg_text in pages:
            m = _RACE_NUM_RE.search(pg_text[:600])
            if m:
                rn = int(m.group(1))
                groups.setdefault(rn, []).append(pg_text)
        return groups

    # ------------------------------------------------------------------
    # Race parsing
    # ------------------------------------------------------------------

    def _parse_race(
        self, race_number: int, page_texts: List[str], track: str, date: str
    ) -> BrisnetRace:
        full_text = '\n'.join(page_texts)
        first_page = page_texts[0]

        race = BrisnetRace(
            track=track, date=date, race_number=race_number,
        )

        # Parse conditions from first page (try ™-prefixed first, then without)
        cm = _CONDITIONS_LINE_RE.search(first_page)
        if not cm:
            cm = _CONDITIONS_LINE_NOTM_RE.search(first_page)
        if cm:
            race.conditions_raw = cm.group(1)
            self._parse_conditions(race, race.conditions_raw, first_page)

        # Parse PARS
        pars_m = re.search(
            r'PARS:\s*\n\s*([\d]+)\s+([\d]+)/\s*([\d]+)\s+([\d]+)',
            first_page,
        )
        if pars_m:
            race.pars = {
                'E1': int(pars_m.group(1)),
                'E2': int(pars_m.group(2)),
                'LATE': int(pars_m.group(3)),
                'SPEED': int(pars_m.group(4)),
            }

        # Split and parse horse blocks
        blocks = self._split_horse_blocks(full_text)
        for post, name, style, rating, block_text in blocks:
            horse = self._parse_horse_block(block_text, post, name, style, rating)
            race.horses.append(horse)

        return race

    def _parse_conditions(self, race: BrisnetRace, cond_line: str, full: str):
        """Extract surface, distance, class, purse from conditions."""
        # Surface: (T) = Turf, (AW) = All-Weather, else Dirt
        if '(T)' in cond_line or '(Turf)' in full[:2000]:
            race.surface = 'TURF'
        elif '(AW)' in cond_line or '(All-Weather)' in full[:3000]:
            race.surface = 'AW'
        else:
            race.surface = 'DIRT'

        # Distance — handle Ì modifier (maps to ½), ˆ (1/16), and formats like 1m70yds
        dist_m = re.search(r'([\d]+[ˆ½¼¾Ì]*)\s*(Mile|Furlongs?|f)\b', cond_line)
        if not dist_m:
            # Try "Nm70yds" or "NmNNyds" format (e.g. 1m70yds = about 1 mile)
            dist_m2 = re.search(r'(\d+m\d+yds)', cond_line)
            if dist_m2:
                race.distance = dist_m2.group(1)
            else:
                # Try in full text
                dist_m3 = re.search(r'([\d]+[ˆ½¼¾Ì]*)\s*(Mile|Furlongs?)', full[:3000])
                if dist_m3:
                    race.distance = f"{dist_m3.group(1)} {dist_m3.group(2)}"
        else:
            race.distance = f"{dist_m.group(1)} {dist_m.group(2)}"

        # Class (first token, strip optional ™ prefix)
        cls_m = re.match(r'™?(\S+)', cond_line)
        if cls_m:
            race.race_class = cls_m.group(1)

        # Purse
        purse_m = re.search(r'Purse\s+\$([\d,]+)', full[:3000])
        if purse_m:
            race.purse = purse_m.group(1)

        # Restrictions (age/sex from conditions line)
        rest_m = re.search(r'(\d+yo\s+\S+|\d+&up)', cond_line, re.IGNORECASE)
        if rest_m:
            race.restrictions = rest_m.group(1)

    # ------------------------------------------------------------------
    # Horse block splitting
    # ------------------------------------------------------------------

    def _split_horse_blocks(self, text: str) -> List[Tuple[int, str, str, int, str]]:
        """Find horse blocks in concatenated race text.
        Returns [(post, name, style, style_rating, block_text), ...]
        """
        matches = list(_HORSE_BLOCK_RE.finditer(text))
        if not matches:
            return []

        # Filter: valid post numbers (1-30), name must contain a letter,
        # and the match must be followed by "Own:" within 200 chars
        valid = []
        for m in matches:
            post = int(m.group(1))
            name = m.group(2).strip()
            # Post numbers above 30 are PARS/stat noise
            if post < 1 or post > 30:
                continue
            # Name must have at least 2 alpha chars (filters out stray numbers)
            if sum(1 for c in name if c.isalpha()) < 2:
                continue
            # "Own:" must appear shortly after the match (real horse blocks)
            after = text[m.end():m.end() + 200]
            if 'Own:' not in after:
                continue
            valid.append(m)

        blocks: List[Tuple[int, str, str, int, str]] = []
        for i, m in enumerate(valid):
            post = int(m.group(1))
            name = m.group(2).strip()
            style = m.group(3).strip()
            rating = int(m.group(4))
            start = m.start()
            end = valid[i + 1].start() if i + 1 < len(valid) else len(text)
            block_text = text[start:end]
            blocks.append((post, name, style, rating, block_text))

        return blocks

    # ------------------------------------------------------------------
    # Horse block parsing
    # ------------------------------------------------------------------

    def _parse_horse_block(
        self, text: str, post: int, name: str, style: str, rating: int,
    ) -> BrisnetHorse:
        horse = BrisnetHorse(
            post=post, name=name, runstyle=style,
            runstyle_rating=rating, raw_block=text,
        )

        # Owner
        om = _OWNER_RE.search(text)
        if om:
            horse.owner = om.group(1).strip()

        # Odds (line after Own:)
        odds_m = re.search(r'^Own:.+\n(\d+/\d+|\d+\.\d+|\*[\d.]+)', text, re.MULTILINE)
        if odds_m:
            horse.odds = odds_m.group(1)

        # Jockey
        jm = _JOCKEY_RE.search(text)
        if jm:
            horse.jockey = jm.group(1).strip()
            horse.jockey_stats = jm.group(2).strip()

        # Trainer
        tm = _TRAINER_RE.search(text)
        if tm:
            horse.trainer = tm.group(1).strip()
            horse.trainer_stats = tm.group(2).strip()

        # Breed info
        bm = _BREED_RE.search(text)
        if bm:
            horse.breed_info = bm.group(1).strip()

        # Sire, Dam, Breeder
        sm = _SIRE_RE.search(text)
        if sm:
            horse.sire = sm.group(1).strip()
        dm = _DAM_RE.search(text)
        if dm:
            horse.dam = dm.group(1).strip()
        brm = _BREEDER_RE.search(text)
        if brm:
            horse.breeder = brm.group(1).strip()

        # Prime Power
        pm = _PRIME_POWER_RE.search(text)
        if pm:
            horse.prime_power = float(pm.group(1))
            horse.prime_power_rank = pm.group(2)

        # Weight
        wm = _WEIGHT_RE.search(text)
        if wm:
            horse.weight = int(wm.group(1))

        # Life record
        lm = _LIFE_RE.search(text)
        if lm:
            horse.life_starts = int(lm.group(1))
            horse.life_record = f"{lm.group(2)}-{lm.group(3)}-{lm.group(4)}"
            horse.life_earnings = lm.group(5)
            if lm.group(6):
                horse.life_speed = int(lm.group(6))

        # Surface records
        for sm_match in _SURFACE_REC_RE.finditer(text):
            label = sm_match.group(1)
            starts = int(sm_match.group(2))
            w, p, s = int(sm_match.group(3)), int(sm_match.group(4)), int(sm_match.group(5))
            earnings = sm_match.group(6)
            spd = int(sm_match.group(7)) if sm_match.group(7) else None
            horse.surface_records[label] = {
                'starts': starts, 'w': w, 'p': p, 's': s,
                'earnings': earnings, 'speed': spd,
            }

        # QuickPlay comments
        self._parse_quickplay(horse, text)

        # Running lines
        self._parse_running_lines(horse, text)

        # Workouts
        self._parse_workouts(horse, text)

        # Derive last_speed from first running line (most recent)
        if horse.running_lines and horse.running_lines[0].get('speed'):
            horse.last_speed = horse.running_lines[0]['speed']
        elif horse.life_speed:
            horse.last_speed = horse.life_speed

        return horse

    def _parse_quickplay(self, horse: BrisnetHorse, text: str):
        """Extract QuickPlay ñ (positive) and × (negative) comments."""
        # Find lines containing ñ or ×
        for line in text.split('\n'):
            if _QP_POSITIVE not in line and _QP_NEGATIVE not in line:
                continue
            # Split on markers, tag each piece
            parts = re.split(r'([ñ×])', line)
            current_marker = None
            for part in parts:
                if part == _QP_POSITIVE:
                    current_marker = 'pos'
                elif part == _QP_NEGATIVE:
                    current_marker = 'neg'
                elif current_marker and part.strip():
                    comment = part.strip()
                    if current_marker == 'pos':
                        horse.quickplay_positive.append(comment)
                    else:
                        horse.quickplay_negative.append(comment)

    def _parse_running_lines(self, horse: BrisnetHorse, text: str):
        """Extract running line entries from the horse block."""
        # Find running line header
        header_m = re.search(r'DATE\s+TRK\s+DIST', text)
        if not header_m:
            return

        rl_zone = text[header_m.start():]

        # Find all date-track anchors
        anchors = list(_RL_DATE_RE.finditer(rl_zone))
        if not anchors:
            return

        for i, anchor in enumerate(anchors):
            start = anchor.start()
            end = anchors[i + 1].start() if i + 1 < len(anchors) else len(rl_zone)
            block = rl_zone[start:end]

            # Stop if we hit workout zone (contains patterns like "08Feb GP  5f  ft")
            if re.search(r'\d{1,2}[A-Z][a-z]{2}\s+\w{2,3}\s+\d+f\s+(?:ft|fm|gd|sy)', block):
                # Check if this is a workout line rather than running line
                if not re.match(r'\d{2}[A-Z][a-z]{2}\d{2}\w{2,3}', block):
                    break

            day = anchor.group(1)
            month = anchor.group(2)
            year = anchor.group(3)
            track = anchor.group(4)

            entry: Dict[str, Any] = {
                'date': f"{day}{month}{year}",
                'track': track,
                'raw_text': block.strip()[:200],
            }

            # Extract surface from distance line (fm=turf, ft=dirt, gd=good, sy=sloppy)
            surf_m = re.search(r'\b(fm|ft|gd|sy|my|yl|sf)\b', block)
            if surf_m:
                surf_code = surf_m.group(1)
                entry['surface'] = {
                    'fm': 'TURF', 'ft': 'DIRT', 'gd': 'DIRT', 'sy': 'SLOP',
                    'my': 'SLOP', 'yl': 'TURF', 'sf': 'TURF',
                }.get(surf_code, 'DIRT')
            else:
                entry['surface'] = 'DIRT'

            # Extract speed figure: look for SPD PP ST pattern
            spd_m = _SPD_LINE_RE.search(block)
            if spd_m:
                spd_val = int(spd_m.group(1))
                if 30 <= spd_val <= 130:
                    entry['speed'] = spd_val

            # Extract race type (™ prefix)
            rt_m = re.search(r'™(\S+)', block)
            if rt_m:
                entry['race_type'] = rt_m.group(1)

            # Extract comment (last line before next entry)
            lines = block.strip().split('\n')
            if len(lines) >= 2:
                last_line = lines[-1].strip()
                # Comment lines don't start with numbers or special markers
                if last_line and not re.match(r'^[\d¨©ª«]', last_line):
                    entry['comment'] = last_line

            horse.running_lines.append(entry)

    def _parse_workouts(self, horse: BrisnetHorse, text: str):
        """Extract workout entries."""
        for wm in _WORKOUT_RE.finditer(text):
            wo: Dict[str, str] = {
                'date': wm.group(1) + (f"'{wm.group(2)}" if wm.group(2) else ''),
                'track': wm.group(3),
                'distance': wm.group(4),
                'surface': wm.group(5),
                'time': wm.group(6),
                'rank_letter': wm.group(7),
                'rank': wm.group(8),
            }
            horse.workouts.append(wo)


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def to_races_json(card: BrisnetCard) -> Dict[str, Any]:
    """Produce the user's desired BRISNET JSON schema."""
    races_out: List[Dict] = []
    for race in card.races:
        horses_out: List[Dict] = []
        for h in race.horses:
            notes_flags = h.quickplay_positive + [f"NEGATIVE: {n}" for n in h.quickplay_negative]
            horses_out.append({
                'post': h.post,
                'name': h.name,
                'runstyle': h.runstyle,
                'prime_power': h.prime_power,
                'class_rating': h.class_rating,
                'last_speed': h.last_speed,
                'best_dist_speed': h.best_dist_speed,
                'trainer': h.trainer,
                'jockey': h.jockey,
                'med_days_since': None,
                'workouts': [
                    f"{w['date']} {w['track']} {w['distance']} {w['surface']} {w['time']} {w['rank_letter']} {w['rank']}"
                    for w in h.workouts
                ],
                'running_lines': [rl.get('raw_text', '') for rl in h.running_lines],
                'notes_flags': notes_flags,
            })

        races_out.append({
            'track': race.track,
            'date': race.date,
            'race_number': race.race_number,
            'conditions': {
                'surface': race.surface,
                'distance': race.distance,
                'class': race.race_class,
                'purse': race.purse,
                'restrictions': race.restrictions,
            },
            'bias': {
                'runstyle_percent': race.pars,
                'post_bias_notes': '',
            },
            'horses': horses_out,
        })

    return {'track': card.track, 'date': card.date, 'races': races_out}


def to_race_chunks(card: BrisnetCard, output_dir: str):
    """Write one race_N.txt per race with META header for Claude prompting."""
    os.makedirs(output_dir, exist_ok=True)
    for race in card.races:
        lines = [
            f"META:",
            f"track={race.track}",
            f"date={race.date}",
            f"race={race.race_number}",
            f"surface={race.surface}",
            f"distance={race.distance}",
            f"class={race.race_class}",
            f"purse={race.purse}",
            f"restrictions={race.restrictions}",
            f"pars={race.pars}",
            '',
        ]
        for h in race.horses:
            lines.append(f"--- HORSE {h.post} {h.name} ---")
            lines.append(f"Style: {h.runstyle} ({h.runstyle_rating})")
            lines.append(f"Prime Power: {h.prime_power} ({h.prime_power_rank})")
            lines.append(f"Jockey: {h.jockey} | Trainer: {h.trainer}")
            lines.append(f"Odds: {h.odds} | Weight: {h.weight}")
            lines.append(f"Life: {h.life_starts} starts {h.life_record} ${h.life_earnings} speed={h.life_speed}")
            if h.quickplay_positive:
                lines.append(f"POSITIVE: {' | '.join(h.quickplay_positive)}")
            if h.quickplay_negative:
                lines.append(f"NEGATIVE: {' | '.join(h.quickplay_negative)}")
            if h.running_lines:
                lines.append(f"Running lines ({len(h.running_lines)}):")
                for rl in h.running_lines[:6]:
                    spd = rl.get('speed', '?')
                    lines.append(f"  {rl['date']} {rl['track']} {rl.get('surface','?')} spd={spd} {rl.get('race_type','')}")
            if h.workouts:
                wo_strs = [f"{w['date']} {w['track']} {w['distance']} {w['time']}" for w in h.workouts[:4]]
                lines.append(f"Workouts: {' | '.join(wo_strs)}")
            lines.append('')

        path = os.path.join(output_dir, f"race_{race.race_number}.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def to_pipeline_json(card: BrisnetCard) -> Dict[str, Any]:
    """Convert to the existing pipeline schema for api.py / engine compatibility.

    Includes full enrichment fields (trainer, jockey, workouts, breed info,
    life/surface records, class/runstyle ratings) and per-race ``races_detail``.
    """
    all_horses: List[Dict] = []
    races_detail: List[Dict] = []

    for race in card.races:
        # Build per-race detail entry
        races_detail.append({
            'race_number': race.race_number,
            'conditions_raw': race.conditions_raw,
            'surface': race.surface,
            'distance': race.distance,
            'race_class': race.race_class,
            'purse': race.purse,
            'restrictions': race.restrictions,
            'pars': race.pars,
        })

        for h in race.horses:
            lines: List[Dict] = []
            for rl in h.running_lines:
                lines.append({
                    'raw_text': rl.get('raw_text', ''),
                    'fig': str(rl.get('speed', '')),
                    'parsed_figure': rl.get('speed'),
                    'flags': h.quickplay_positive[:],
                    'surface': rl.get('surface', race.surface),
                    'surface_type': rl.get('surface', race.surface),
                    'track': rl.get('track', race.track),
                    'data_text': rl.get('comment', ''),
                    'race_type': rl.get('race_type', ''),
                    'race_date': rl.get('date', ''),
                    'notes': rl.get('comment', ''),
                    'race_analysis': '',
                })
            if not lines and h.life_speed:
                # Fallback: single line from life record
                lines = [{
                    'fig': str(h.life_speed),
                    'parsed_figure': h.life_speed,
                    'flags': [],
                    'surface': race.surface,
                    'track': race.track,
                    'race_type': race.race_class,
                    'race_date': race.date,
                    'notes': '',
                    'race_analysis': '',
                    'raw_text': f"Life speed: {h.life_speed}",
                }]

            horse_dict = {
                'horse_name': h.name,
                'race_number': race.race_number,
                'sex': h.breed_info,
                'age': 0,
                'breeder_owner': h.owner,
                'foal_date': '',
                'reg_code': '',
                'races': len(lines),
                'top_fig': str(h.last_speed or h.life_speed or ''),
                'horse_analysis': f"Prime Power: {h.prime_power}",
                'performance_trend': f"{len(lines)} past performances",
                'lines': lines,
                'post': h.post,
                'runstyle': h.runstyle,
                'runstyle_rating': h.runstyle_rating,
                'prime_power': h.prime_power,
                'prime_power_rank': h.prime_power_rank,
                'quickplay_positive': h.quickplay_positive,
                'quickplay_negative': h.quickplay_negative,
                'figure_source': 'brisnet',
                # Enrichment fields
                'trainer': h.trainer,
                'trainer_stats': h.trainer_stats,
                'jockey': h.jockey,
                'jockey_stats': h.jockey_stats,
                'workouts': h.workouts,
                'owner': h.owner,
                'breed_info': h.breed_info,
                'sire': h.sire,
                'dam': h.dam,
                'breeder': h.breeder,
                'weight': h.weight,
                'odds': h.odds,
                'life_starts': h.life_starts,
                'life_record': h.life_record,
                'life_earnings': h.life_earnings,
                'life_speed': h.life_speed,
                'surface_records': h.surface_records,
                'class_rating': h.class_rating,
            }
            all_horses.append(horse_dict)

    return {
        'track_name': card.track,
        'race_date': card.date,
        'race_number': 0,
        'surface': '',
        'distance': '',
        'weather': None,
        'race_conditions': None,
        'horses': all_horses,
        'races_detail': races_detail,
        'pdf_type': 'brisnet',
        'figure_source': 'brisnet',
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Parse BRISNET Ultimate PPs PDF')
    ap.add_argument('pdf_path', help='Path to BRISNET PDF')
    ap.add_argument('--output-dir', '-o', default='brisnet_output',
                    help='Output directory (default: brisnet_output)')
    ap.add_argument('--format', choices=['json', 'both'], default='both',
                    help='json only, or json + race_N.txt chunks')
    args = ap.parse_args()

    bp = BrisnetParser()
    card = bp.parse(args.pdf_path)

    os.makedirs(args.output_dir, exist_ok=True)

    # Write races.json
    races_json = to_races_json(card)
    json_path = os.path.join(args.output_dir, 'races.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(races_json, f, indent=2, ensure_ascii=False)
    print(f"Wrote {json_path}")

    # Write pipeline-compatible JSON
    pipeline_json = to_pipeline_json(card)
    pipe_path = os.path.join(args.output_dir, 'pipeline.json')
    with open(pipe_path, 'w', encoding='utf-8') as f:
        json.dump(pipeline_json, f, indent=2, ensure_ascii=False)
    print(f"Wrote {pipe_path}")

    # Write race chunks
    if args.format == 'both':
        to_race_chunks(card, args.output_dir)
        print(f"Wrote race_*.txt chunks to {args.output_dir}/")

    # Summary
    print(f"\nParsed {len(card.races)} races, "
          f"{sum(len(r.horses) for r in card.races)} horses")
    for r in card.races:
        speeds = [h.last_speed for h in r.horses if h.last_speed]
        avg_spd = sum(speeds) / len(speeds) if speeds else 0
        print(f"  Race {r.race_number}: {len(r.horses)} horses, "
              f"surface={r.surface}, dist={r.distance}, "
              f"avg_speed={avg_spd:.0f}")


if __name__ == '__main__':
    main()
