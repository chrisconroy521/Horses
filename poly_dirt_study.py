"""Paired-Poly-to-Dirt study: 3yo focus, all ages reported.

Scenario:
  - Horse has consecutive POLY starts that are paired (within 0-1 pts).
  - Next start is on DIRT.
  - Measure: did the horse "go forward" on dirt?

Data sources:
  PRIMARY  = Ragozin Sheets  (figures, surface from track codes)
  ENRICHMENT = BRISNET PPs   (runstyle, race dates, age, sire)

Surface derivation: track code starting with / containing "AW" -> POLY.
Age derivation:
  1) BRISNET race restrictions ("3yo Fillies" -> 3)
  2) Ragozin foal year (>=10 -> current_year - (2000 + val))
  3) 0 = unknown
Deduplication: unique (horse_name, poly_0_fig, poly_1_fig, dirt_fig).
"""
import json
import glob
import re
import os
from datetime import date, datetime
from collections import defaultdict
from typing import List, Dict, Optional, Any

CURRENT_YEAR = date.today().year

# Month abbreviations used in BRISNET date format (e.g., "04Dec25")
_MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jly": 7, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


# ======================================================================
# Surface + age helpers
# ======================================================================

def derive_surface(track_code: str, raw_surface: str) -> str:
    tc = (track_code or "").upper()
    if tc.startswith("AW") or "AW" in tc:
        return "POLY"
    raw = (raw_surface or "DIRT").upper()
    if raw in ("AW", "SYNTH", "POLY"):
        return "POLY"
    return raw


def derive_age_from_foal_year(raw_age: int) -> int:
    if raw_age >= 10:
        return CURRENT_YEAR - (2000 + raw_age)
    return raw_age


def parse_brisnet_date(date_str: str) -> Optional[date]:
    """Parse BRISNET date like '04Dec25' or '31Jan26' -> date object."""
    m = re.match(r"(\d{2})([A-Z][a-z]{2})(\d{2})", date_str or "")
    if not m:
        return None
    day = int(m.group(1))
    month = _MONTH_MAP.get(m.group(2))
    year = 2000 + int(m.group(3))
    if not month:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def normalize_name(name: str) -> str:
    """Normalize horse name for matching: uppercase, strip punctuation."""
    n = (name or "").upper().strip()
    n = re.sub(r"['\-.,()]", "", n)
    n = re.sub(r"\s+", " ", n)
    return n


# ======================================================================
# BRISNET enrichment adapter
# ======================================================================

def load_brisnet_enrichment(
    pipeline_path: str = "brisnet_output/pipeline.json",
    races_path: str = "brisnet_output/races.json",
) -> Dict[str, Dict]:
    """Load BRISNET data and build per-horse enrichment dict.

    Returns dict keyed by normalized horse name -> {
        'runstyle': str,
        'sire': str,
        'dam': str,
        'age': int (from race restrictions),
        'race_number': int,
        'post': int,
        'running_line_dates': [date objects, 0-back first],
        'running_line_surfaces': [str, 0-back first],
    }
    """
    enrichment: Dict[str, Dict] = {}

    # --- Load pipeline.json for per-horse data ---
    if not os.path.exists(pipeline_path):
        return enrichment

    with open(pipeline_path, encoding="utf-8") as f:
        pipeline = json.load(f)

    # --- Load races.json for restrictions (age info) ---
    race_restrictions: Dict[int, str] = {}
    race_surfaces: Dict[int, str] = {}
    if os.path.exists(races_path):
        with open(races_path, encoding="utf-8") as f:
            races_data = json.load(f)
        for race in races_data.get("races", []):
            rn = race.get("race_number", 0)
            cond = race.get("conditions", "")
            race_restrictions[rn] = cond
            race_surfaces[rn] = race.get("surface", "")

    # --- Also parse restrictions from race text files ---
    race_dir = os.path.dirname(pipeline_path)
    for i in range(1, 20):
        txt_path = os.path.join(race_dir, f"race_{i}.txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("restrictions="):
                    race_restrictions[i] = line.split("=", 1)[1]
                if line.startswith("surface="):
                    surf = line.split("=", 1)[1].upper()
                    if surf in ("AW", "SYNTH", "POLY"):
                        race_surfaces[i] = "POLY"
                    elif i not in race_surfaces:
                        race_surfaces[i] = surf

    # --- Extract age from restrictions ---
    def age_from_restrictions(restr: str) -> int:
        restr = (restr or "").lower()
        m = re.search(r"(\d)yo", restr)
        if m:
            return int(m.group(1))
        if "3&up" in restr or "3 & up" in restr:
            return 3  # conservative: could be older
        if "4&up" in restr or "4 & up" in restr:
            return 4  # conservative
        return 0

    # --- Build enrichment per horse ---
    for h in pipeline.get("horses", []):
        name = h.get("horse_name", "")
        if not name:
            continue

        rn = h.get("race_number", 0)
        restr = race_restrictions.get(rn, "")
        age = age_from_restrictions(restr)

        # Parse running line dates
        rl_dates = []
        rl_surfaces = []
        for line in h.get("lines", []):
            dt = parse_brisnet_date(line.get("race_date", ""))
            rl_dates.append(dt)
            rl_surfaces.append(line.get("surface", "DIRT").upper())

        key = normalize_name(name)
        entry = {
            "runstyle": h.get("runstyle", ""),
            "sire": h.get("sire") or "",
            "dam": h.get("dam") or "",
            "age": age,
            "race_number": rn,
            "post": h.get("post", 0),
            "running_line_dates": rl_dates,
            "running_line_surfaces": rl_surfaces,
            "race_surface": race_surfaces.get(rn, ""),
            "match_confidence": "high",
        }
        enrichment[key] = entry

    return enrichment


def load_brisnet_from_csv(csv_path: str) -> Dict[str, Dict]:
    """Fallback: load enrichment from a flat CSV with columns:
    horse_name, age, sire, runstyle, race_date_0, race_date_1, ...

    Returns same dict format as load_brisnet_enrichment.
    """
    import csv
    enrichment: Dict[str, Dict] = {}
    if not os.path.exists(csv_path):
        return enrichment

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("horse_name", "")
            if not name:
                continue
            key = normalize_name(name)

            # Parse race dates from columns race_date_0, race_date_1, etc.
            rl_dates = []
            for i in range(20):
                dt_str = row.get(f"race_date_{i}", "")
                if dt_str:
                    rl_dates.append(parse_brisnet_date(dt_str))
                else:
                    break

            enrichment[key] = {
                "runstyle": row.get("runstyle", ""),
                "sire": row.get("sire", ""),
                "dam": row.get("dam", ""),
                "age": int(row.get("age", 0) or 0),
                "race_number": int(row.get("race_number", 0) or 0),
                "post": int(row.get("post", 0) or 0),
                "running_line_dates": rl_dates,
                "running_line_surfaces": [],
                "race_surface": "",
                "match_confidence": "med",
            }

    return enrichment


# ======================================================================
# Merge: Ragozin primary + BRISNET enrichment
# ======================================================================

def merge_enrichment(
    all_horses: List[Dict],
    enrichment: Dict[str, Dict],
) -> tuple:
    """Merge BRISNET enrichment into Ragozin horse dicts.

    Match by normalized horse name. Enriches in-place and returns
    (matched_count, unmatched_ragozin, unmatched_brisnet).
    """
    matched = 0
    unmatched_rag = []
    used_keys = set()

    for h in all_horses:
        name = h.get("horse_name", "")
        key = normalize_name(name)

        if key in enrichment:
            enr = enrichment[key]
            h["_enrichment"] = enr
            h["_match_confidence"] = enr.get("match_confidence", "med")
            used_keys.add(key)
            matched += 1
        else:
            unmatched_rag.append(name)

    unmatched_bris = [k for k in enrichment if k not in used_keys]

    return matched, unmatched_rag, unmatched_bris


# ======================================================================
# Load Ragozin horses
# ======================================================================

def load_all_horses(output_dir="output"):
    """Load all horses from all Ragozin JSON files."""
    all_horses = []
    for f in sorted(glob.glob(os.path.join(output_dir, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            try:
                d = json.load(fh)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        horses = d.get("horses", [])
        if isinstance(horses, list):
            for h in horses:
                if h.get("horse_name") and h["horse_name"] not in ("No Data Found", "Unknown"):
                    all_horses.append(h)
        h = d.get("horse", {})
        if isinstance(h, dict) and h.get("horse_name") and h["horse_name"] not in ("No Data Found", "Unknown"):
            all_horses.append(h)
    return all_horses


# ======================================================================
# Pattern finder (enrichment-aware)
# ======================================================================

def find_pattern_instances(all_horses: List[Dict]) -> List[Dict]:
    """Find all paired-poly-to-dirt instances.

    Returns list of dicts with keys:
      name, age, total_races, poly_1_fig, poly_0_fig, dirt_fig,
      delta, pair_gap, style, sire, days_between, match_confidence
    """
    seen = set()
    instances = []

    for h in all_horses:
        name = h.get("horse_name", "")
        lines = h.get("lines", [])
        if len(lines) < 3:
            continue

        # --- Age: prefer BRISNET enrichment, fallback to foal year ---
        enr = h.get("_enrichment", {})
        age = enr.get("age", 0)
        if not age:
            raw_age = h.get("age", 0)
            age = derive_age_from_foal_year(raw_age) if raw_age > 0 else 0

        # --- Runstyle from BRISNET ---
        style = enr.get("runstyle", "") or ""

        # --- Sire from BRISNET ---
        sire = enr.get("sire", "") or ""

        # --- Running line dates from BRISNET ---
        rl_dates = enr.get("running_line_dates", [])

        total_races = len(lines)

        # Derive figure + surface per Ragozin line
        derived = []
        for idx, line in enumerate(lines):
            fig = line.get("parsed_figure")
            if fig is None:
                try:
                    fig = float(line.get("fig", "0"))
                except (ValueError, TypeError):
                    fig = 0.0
            track = line.get("track", "")
            surf = derive_surface(track, line.get("surface", "DIRT"))
            raw_dt = rl_dates[idx] if idx < len(rl_dates) else None
            # Ensure dt is a date object (DB may return raw strings)
            if isinstance(raw_dt, str) and raw_dt:
                dt = parse_brisnet_date(raw_dt)
            else:
                dt = raw_dt
            derived.append({"fig": fig, "surface": surf, "track": track, "date": dt})

        for i in range(len(derived) - 2):
            dirt = derived[i]
            poly_0 = derived[i + 1]
            poly_1 = derived[i + 2]

            if (dirt["surface"] == "DIRT"
                    and poly_0["surface"] == "POLY"
                    and poly_1["surface"] == "POLY"
                    and poly_0["fig"] > 0 and poly_1["fig"] > 0 and dirt["fig"] > 0
                    and abs(poly_0["fig"] - poly_1["fig"]) <= 1.0):

                dedup_key = (name, poly_1["fig"], poly_0["fig"], dirt["fig"])
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                delta = poly_0["fig"] - dirt["fig"]  # positive = forward

                # Days between poly_0 and dirt start
                days_between = None
                if dirt["date"] and poly_0["date"]:
                    days_between = (poly_0["date"] - dirt["date"]).days
                    if days_between < 0:
                        days_between = abs(days_between)

                instances.append({
                    "name": name,
                    "age": age,
                    "total_races": total_races,
                    "poly_1_fig": poly_1["fig"],
                    "poly_0_fig": poly_0["fig"],
                    "dirt_fig": dirt["fig"],
                    "delta": delta,
                    "pair_gap": abs(poly_0["fig"] - poly_1["fig"]),
                    "style": style,
                    "sire": sire,
                    "days_between": days_between,
                    "match_confidence": h.get("_match_confidence", "none"),
                })

    return instances


# ======================================================================
# Reporting
# ======================================================================

def classify_result(delta: float) -> str:
    if delta >= 2:
        return "forward_A"
    elif delta >= 1:
        return "forward_B"
    elif abs(delta) <= 1:
        return "same"
    else:
        return "regress"


def print_breakdown(label: str, instances: list):
    n = len(instances)
    if n == 0:
        print(f"  {label}: N=0")
        return

    fwd_a = sum(1 for inst in instances if inst["delta"] >= 2)
    fwd_b = sum(1 for inst in instances if inst["delta"] >= 1)
    same = sum(1 for inst in instances if abs(inst["delta"]) <= 1)
    regress = sum(1 for inst in instances if inst["delta"] <= -2)

    print(f"  {label}: N={n}")
    print(f"    Forward A (>=2 pts): {fwd_a:3d} ({fwd_a/n*100:5.1f}%)")
    print(f"    Forward B (>=1 pt):  {fwd_b:3d} ({fwd_b/n*100:5.1f}%)")
    print(f"    Same (within +/-1):  {same:3d} ({same/n*100:5.1f}%)")
    print(f"    Regress (>=2 worse): {regress:3d} ({regress/n*100:5.1f}%)")
    avg_delta = sum(inst["delta"] for inst in instances) / n
    print(f"    Avg delta: {avg_delta:+.1f} pts")


def print_sire_breakdown(instances: list):
    by_sire = defaultdict(list)
    for inst in instances:
        s = inst.get("sire") or "Unknown"
        by_sire[s].append(inst)

    ranked = sorted(by_sire.items(), key=lambda x: -len(x[1]))
    for sire, group in ranked[:10]:
        print_breakdown(f"Sire: {sire}", group)


def print_style_breakdown(instances: list):
    by_style = defaultdict(list)
    for inst in instances:
        s = inst.get("style") or "Unknown"
        by_style[s].append(inst)

    for style in ["E", "EP", "P", "S", "Unknown"]:
        if style in by_style:
            print_breakdown(f"Style: {style}", by_style[style])


def print_layoff_breakdown(instances: list):
    short = [i for i in instances if i.get("days_between") is not None and i["days_between"] <= 21]
    mid = [i for i in instances if i.get("days_between") is not None and 22 <= i["days_between"] <= 45]
    long = [i for i in instances if i.get("days_between") is not None and i["days_between"] >= 46]
    unknown = [i for i in instances if i.get("days_between") is None]

    print_breakdown("<=21 days", short)
    print_breakdown("22-45 days", mid)
    print_breakdown("46+ days", long)
    if unknown:
        print_breakdown("Days unknown", unknown)


# ======================================================================
# DB-aware loading
# ======================================================================

def load_all_horses_from_db(db_path="horses.db"):
    """Load all horses from the cumulative database.

    Returns (all_horses_list, enrichment_dict) or ([], {}) if DB unavailable.
    Enrichment is already wired in via reconciliation — no separate merge needed.
    """
    from pathlib import Path
    if not Path(db_path).exists():
        return [], {}
    from persistence import Persistence
    db = Persistence(Path(db_path))
    stats = db.get_db_stats()
    if stats["sheets_horses"] == 0:
        return [], {}
    return db.load_enriched_horses()


# ======================================================================
# Main
# ======================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Paired Poly-to-Dirt study")
    ap.add_argument("--db", default="horses.db", help="Database path (default: horses.db)")
    ap.add_argument("--no-db", action="store_true", help="Force JSON-only loading (skip DB)")
    args = ap.parse_args()

    # --- Try cumulative DB first ---
    enrichment = {}
    if not args.no_db:
        all_horses, enrichment = load_all_horses_from_db(args.db)
        if all_horses:
            enriched_count = sum(1 for h in all_horses if "_enrichment" in h)
            print(f"Loaded {len(all_horses)} Ragozin horses from cumulative database ({args.db})")
            print(f"  BRISNET enrichment: {enriched_count}/{len(all_horses)} horses")
        else:
            all_horses = None
    else:
        all_horses = None

    # --- Fallback: JSON files ---
    if all_horses is None:
        all_horses = load_all_horses()
        print(f"Loaded {len(all_horses)} Ragozin horses from output/*.json")

        enrichment_data = load_brisnet_enrichment()
        if not enrichment_data:
            csv_path = "brisnet_enrichment.csv"
            if os.path.exists(csv_path):
                enrichment_data = load_brisnet_from_csv(csv_path)
                print(f"Loaded {len(enrichment_data)} horses from CSV enrichment: {csv_path}")
            else:
                print("No BRISNET enrichment found (checked pipeline.json and CSV).")

        if enrichment_data:
            matched, unmatched_rag, unmatched_bris = merge_enrichment(all_horses, enrichment_data)
            print(f"BRISNET merge: {matched} matched, "
                  f"{len(unmatched_rag)} Ragozin-only, {len(unmatched_bris)} BRISNET-only")
        else:
            print("Running without enrichment (Sheets-only mode).")

    # --- Find pattern instances ---
    instances = find_pattern_instances(all_horses)

    print(f"\n{'=' * 72}")
    print(f"PAIRED POLY -> DIRT STUDY  (Sheets primary, BRISNET enrichment)")
    print(f"Criteria: consecutive POLY starts within 0-1 pts, then DIRT next")
    print(f"Lower figure = better (Ragozin convention)")
    print(f"{'=' * 72}")

    # --- Overall ---
    print(f"\n--- ALL AGES ---")
    print_breakdown("Overall", instances)

    # --- By age ---
    print(f"\n--- BY AGE ---")
    by_age = defaultdict(list)
    for inst in instances:
        by_age[inst["age"]].append(inst)
    for age in sorted(by_age.keys()):
        label = f"Age {age}" if age > 0 else "Age unknown"
        print_breakdown(label, by_age[age])

    # --- 3yo only ---
    three_yo = [inst for inst in instances if inst["age"] == 3]
    print(f"\n--- 3-YEAR-OLDS ONLY ---")
    print_breakdown("3yo", three_yo)

    # --- By runstyle ---
    styled = [i for i in instances if i.get("style")]
    print(f"\n--- BY RUNNING STYLE ---")
    if styled:
        print_style_breakdown(instances)
    else:
        print("  (no runstyle data — need BRISNET enrichment)")

    # --- By sire ---
    sired = [i for i in instances if i.get("sire")]
    print(f"\n--- BY SIRE (top 10) ---")
    if sired:
        print_sire_breakdown(instances)
    else:
        print("  (no sire data — not in current BRISNET format)")

    # --- By layoff ---
    print(f"\n--- BY LAYOFF (days between poly and dirt) ---")
    dated = [i for i in instances if i.get("days_between") is not None]
    if dated:
        print_layoff_breakdown(instances)
    else:
        print("  (no race date data — need BRISNET enrichment)")

    # --- By career length (proxy for age) ---
    print(f"\n--- BY CAREER LENGTH ---")
    short = [inst for inst in instances if inst["total_races"] <= 5]
    mid = [inst for inst in instances if 6 <= inst["total_races"] <= 12]
    long_ = [inst for inst in instances if inst["total_races"] > 12]
    print_breakdown("1-5 starts (likely young)", short)
    print_breakdown("6-12 starts", mid)
    print_breakdown("13+ starts", long_)

    # --- Individual results ---
    print(f"\n--- INDIVIDUAL INSTANCES (N={len(instances)}) ---")
    header = (f"{'Horse':<22} {'Age':>3} {'Sty':>3} {'Sire':<15} "
              f"{'P1':>4} {'P0':>4} {'Dirt':>4} {'D':>5} {'Days':>4} {'Result':<10}")
    print(header)
    print("-" * len(header))
    for inst in sorted(instances, key=lambda x: -x["delta"]):
        result = classify_result(inst["delta"])
        age_str = str(inst["age"]) if inst["age"] > 0 else "?"
        sty = inst.get("style") or "?"
        sire = (inst.get("sire") or "?")[:15]
        days = str(inst["days_between"]) if inst.get("days_between") is not None else "?"
        print(f"{inst['name']:<22} {age_str:>3} {sty:>3} {sire:<15} "
              f"{inst['poly_1_fig']:4.0f} {inst['poly_0_fig']:4.0f} "
              f"{inst['dirt_fig']:4.0f} {inst['delta']:+5.1f} {days:>4} {result:<10}")

    # --- Enrichment audit ---
    print(f"\n{'=' * 72}")
    print("ENRICHMENT AUDIT:")
    enriched = sum(1 for i in instances if i.get("match_confidence") != "none")
    print(f"  Instances with BRISNET enrichment: {enriched}/{len(instances)}")
    print(f"  Ages resolved: {sum(1 for i in instances if i['age'] > 0)}/{len(instances)}")
    print(f"  Runstyles resolved: {sum(1 for i in instances if i.get('style'))}/{len(instances)}")
    print(f"  Sires resolved: {sum(1 for i in instances if i.get('sire'))}/{len(instances)}")
    print(f"  Race dates resolved: {sum(1 for i in instances if i.get('days_between') is not None)}/{len(instances)}")
    if not enrichment:
        print("  -> To enrich: place BRISNET pipeline.json in brisnet_output/")
        print("     or provide brisnet_enrichment.csv with columns:")
        print("     horse_name, age, sire, runstyle, race_date_0, race_date_1, ...")


if __name__ == "__main__":
    main()
