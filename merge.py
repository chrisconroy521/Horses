"""Deterministic merge of traditional + GPT parsed Ragozin data."""
from __future__ import annotations

import re
from typing import Dict, List, Any

# --- Name normalization ---

def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


# --- Field precedence sets ---

# Numeric / Sheets fields -> prefer traditional
TRADITIONAL_FIELDS = frozenset({
    'fig', 'parsed_figure', 'figure', 'figure_raw',
    'flags', 'pre_symbols', 'post_symbols',
    'surface', 'surface_type', 'track_surface',
    'track', 'track_code',
    'race_date', 'month', 'month_label', 'date_code',
    'race_number', 'race_year', 'race_index',
    'distance', 'distance_bracket',
    'race_class_code', 'top_fig',
    'finish_position', 'odds', 'weight',
    'speed_rating', 'pace_rating', 'class_rating',
    'weather_conditions', 'trouble_indicators',
    'races',
})

# Free-text / narrative fields -> prefer GPT
GPT_FIELDS = frozenset({
    'notes', 'race_analysis', 'horse_analysis', 'performance_trend',
    'ai_analysis', 'race_type',
    'sire', 'dam', 'damsire', 'state_bred', 'sex',
    'breeder_owner', 'foaling_year', 'sheet_page_number',
})

_EMPTY = (None, '', 0, [], {}, 'Unknown')


def _is_empty(val: Any) -> bool:
    return val in _EMPTY


# --- Single-line merge ---

def _merge_fields(trad: Dict, gpt: Dict) -> Dict:
    """Merge two flat dicts using field-precedence rules."""
    merged: Dict[str, Any] = {}
    all_keys = set(trad.keys()) | set(gpt.keys())

    for key in all_keys:
        if key == 'source':
            continue
        trad_val = trad.get(key)
        gpt_val = gpt.get(key)

        if key in TRADITIONAL_FIELDS:
            merged[key] = trad_val if not _is_empty(trad_val) else gpt_val
        elif key in GPT_FIELDS:
            merged[key] = gpt_val if not _is_empty(gpt_val) else trad_val
        else:
            # Unknown field: prefer non-empty, GPT wins ties
            merged[key] = gpt_val if not _is_empty(gpt_val) else trad_val

    return merged


# --- Race-line merge ---

def _line_key(line: Dict) -> str:
    """Key a race line by (race_date, normalized track)."""
    return f"{line.get('race_date', '')}|{normalize_name(line.get('track', ''))}"


def _merge_race_lines(trad_lines: List[Dict], gpt_lines: List[Dict]) -> List[Dict]:
    """Merge two lists of race-line dicts."""
    gpt_by_key: Dict[str, Dict] = {}
    for line in gpt_lines:
        gpt_by_key[_line_key(line)] = line

    merged_lines: List[Dict] = []
    seen_keys: set = set()

    for trad_line in trad_lines:
        key = _line_key(trad_line)
        gpt_line = gpt_by_key.get(key)
        if gpt_line:
            seen_keys.add(key)
            merged = _merge_fields(trad_line, gpt_line)
            merged['source'] = 'both'
        else:
            merged = dict(trad_line)
            merged['source'] = 'traditional'
        merged_lines.append(merged)

    for key, gpt_line in gpt_by_key.items():
        if key not in seen_keys:
            line = dict(gpt_line)
            line['source'] = 'gpt'
            merged_lines.append(line)

    return merged_lines


# --- Horse-level merge ---

def _merge_horse_dicts(trad: Dict, gpt: Dict) -> Dict:
    """Merge a single horse dict from both parsers."""
    # Separate lines for special handling
    trad_lines = trad.get('lines', [])
    gpt_lines = gpt.get('lines', [])

    trad_flat = {k: v for k, v in trad.items() if k != 'lines'}
    gpt_flat = {k: v for k, v in gpt.items() if k != 'lines'}

    merged = _merge_fields(trad_flat, gpt_flat)
    merged['lines'] = _merge_race_lines(trad_lines, gpt_lines)
    merged['source'] = 'both'
    return merged


# --- Session-level merge (public API) ---

def merge_parsed_sessions(
    trad_json: Dict[str, Any],
    gpt_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two parsed sessions (traditional + GPT) into one combined dict.

    Both inputs should be in the JSON-output schema:
      { "horses": [ { "horse_name": ..., "lines": [...], ... }, ... ], ... }

    GPT format may also be { "horse": { ... }, "parsed_at": ..., "source_file": ... }
    which gets normalized to the horses-list form.

    Returns a merged dict with per-horse ``source`` labels and a top-level
    ``parse_source`` = "both".
    """
    # Normalize GPT wrapper format -> horses list
    trad_horses = trad_json.get('horses', [])
    gpt_horses = gpt_json.get('horses', [])
    if not gpt_horses and 'horse' in gpt_json:
        gpt_horses = [gpt_json['horse']]

    # Index by normalized name
    trad_by_name: Dict[str, Dict] = {}
    for h in trad_horses:
        key = normalize_name(h.get('horse_name', ''))
        if key:
            trad_by_name[key] = h

    gpt_by_name: Dict[str, Dict] = {}
    for h in gpt_horses:
        key = normalize_name(h.get('horse_name', ''))
        if key:
            gpt_by_name[key] = h

    all_names = set(trad_by_name.keys()) | set(gpt_by_name.keys())

    merged_horses: List[Dict] = []
    for name in sorted(all_names):
        trad_h = trad_by_name.get(name)
        gpt_h = gpt_by_name.get(name)

        if trad_h and gpt_h:
            merged_horses.append(_merge_horse_dicts(trad_h, gpt_h))
        elif trad_h:
            horse = dict(trad_h)
            horse['source'] = 'traditional'
            merged_horses.append(horse)
        else:
            horse = dict(gpt_h)
            horse['source'] = 'gpt'
            merged_horses.append(horse)

    # Top-level metadata: prefer traditional for numeric, GPT for text
    merged: Dict[str, Any] = {}
    for key in set(trad_json.keys()) | set(gpt_json.keys()):
        if key in ('horses', 'horse'):
            continue
        trad_val = trad_json.get(key)
        gpt_val = gpt_json.get(key)
        if key in TRADITIONAL_FIELDS:
            merged[key] = trad_val if not _is_empty(trad_val) else gpt_val
        elif key in GPT_FIELDS:
            merged[key] = gpt_val if not _is_empty(gpt_val) else trad_val
        else:
            merged[key] = trad_val if not _is_empty(trad_val) else gpt_val

    merged['horses'] = merged_horses
    merged['parse_source'] = 'both'

    return merged
