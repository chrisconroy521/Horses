"""CLI for ingesting race results into the cumulative database.

Usage:
    python ingest_results.py --csv results.csv --track GP --date 02/26/2026
    python ingest_results.py --csv results.csv  (track/date from CSV if present)

CSV format:
    race_number,post,horse_name,finish_pos,odds,win_payoff,place_payoff,show_payoff

Optional columns: beaten_lengths, surface, distance
"""
import argparse
import csv
import sys
from pathlib import Path

from persistence import Persistence

# Confidence threshold for auto-ingest of PDF results
PDF_CONFIDENCE_THRESHOLD = 0.8

# Maximum sample rows returned in preview
_PREVIEW_SAMPLE_SIZE = 10


def _float(val):
    """Safely parse a float, returning None on failure."""
    try:
        return float(val) if val and str(val).strip() else None
    except (ValueError, TypeError):
        return None


def _ingest_rows(
    db: Persistence, rows: list, track: str = "", race_date: str = "",
    session_id: str = "", program_map: str = "",
) -> dict:
    """Shared insertion logic: take a list of row dicts and insert into DB.

    Each row dict should have keys matching the CSV schema:
        race_number, post, horse_name, finish_pos, odds,
        win_payoff, place_payoff, show_payoff,
        [optional] beaten_lengths, surface, distance, track, race_date,
                   program, scratched

    *program_map*: "program" to use program column as post, "post" (default)
    to use post column.
    """
    races_seen = set()
    entries_inserted = 0
    errors = 0

    for row in rows:
        try:
            # Skip scratched entries
            scratched = row.get("scratched")
            if scratched is True or str(scratched).lower() in ("true", "1", "yes"):
                continue

            rn = int(row.get("race_number", 0))

            # Program vs post mapping
            if program_map == "program":
                raw_pgm = str(row.get("program", "")).strip()
                # Extract leading digits for DB post column
                import re as _re
                m = _re.match(r"(\d+)", raw_pgm)
                post = int(m.group(1)) if m else int(row.get("post", 0))
            else:
                post = int(row.get("post", 0))

            name = row.get("horse_name", "").strip()
            finish = int(row.get("finish_pos", 0) or 0)

            r_track = row.get("track", "").strip() or track
            r_date = row.get("race_date", "").strip() or race_date

            if not r_track or not r_date or not rn:
                errors += 1
                continue

            odds = _float(row.get("odds"))
            win_p = _float(row.get("win_payoff"))
            place_p = _float(row.get("place_payoff"))
            show_p = _float(row.get("show_payoff"))
            beaten = _float(row.get("beaten_lengths"))

            # Insert race result (once per race)
            race_key = (r_track, r_date, rn)
            if race_key not in races_seen:
                winner_name = name if finish == 1 else ""
                winner_post = post if finish == 1 else 0
                db.insert_race_result(
                    track=r_track, race_date=r_date, race_number=rn,
                    surface=row.get("surface", "") or "",
                    distance=row.get("distance", "") or "",
                    winner_post=winner_post, winner_name=winner_name,
                )
                races_seen.add(race_key)
            elif finish == 1:
                db.insert_race_result(
                    track=r_track, race_date=r_date, race_number=rn,
                    surface=row.get("surface", "") or "",
                    distance=row.get("distance", "") or "",
                    winner_post=post, winner_name=name,
                )

            db.insert_entry_result(
                track=r_track, race_date=r_date, race_number=rn, post=post,
                horse_name=name, finish_pos=finish,
                beaten_lengths=beaten, odds=odds,
                win_payoff=win_p, place_payoff=place_p, show_payoff=show_p,
                session_id=session_id,
            )
            entries_inserted += 1
        except Exception as e:
            print(f"  [err] row: {e}")
            errors += 1

    # Validate: each race should have at least one finish_pos=1
    for race_key in races_seen:
        r_track, r_date, rn = race_key
        winner_count = db.conn.execute(
            "SELECT COUNT(*) FROM result_entries WHERE track = ? AND race_date = ? AND race_number = ? AND finish_pos = 1",
            (r_track, r_date, rn),
        ).fetchone()[0]
        if winner_count < 1:
            db.mark_race_parse_error(r_track, r_date, rn)
            print(f"  [warn] Race {rn}: no winner (finish_pos=1) — marked PARSE_ERROR")

    # Link to existing horses/projections
    link_result = db.link_results_to_entries(
        track=track, race_date=race_date, session_id=session_id,
    )

    return {
        "races": len(races_seen),
        "entries": entries_inserted,
        "errors": errors,
        "linked": link_result["linked"],
        "link_rate": link_result["link_rate"],
        "unmatched": link_result["unmatched"],
    }


def ingest_csv(
    db: Persistence, csv_path: str, track: str = "", race_date: str = "",
    session_id: str = "",
) -> dict:
    """Parse a results CSV and insert into DB. Returns summary dict."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return _ingest_rows(db, rows, track=track, race_date=race_date,
                        session_id=session_id)


def preview_pdf(pdf_path: str, track: str = "", race_date: str = "") -> dict:
    """Parse an Equibase chart PDF and return a dry-run preview (no DB writes).

    Returns:
        races (int): number of races detected
        entries (int): number of entries (excl. scratches)
        scratches (int): number of scratched entries
        sample_rows (list): first N row dicts
        all_rows (list): all row dicts (for confirm step)
        confidence (float): 0.0–1.0
        missing_fields (list[str]): per-field issues
        track (str): detected or provided track
        race_date (str): detected or provided date
        raw_text (str): truncated raw text for debugging
    """
    from results_chart_parser import ResultsChartParser, chart_to_rows

    parser = ResultsChartParser()
    result = parser.parse(pdf_path)
    rows = chart_to_rows(result)

    # Use parsed track/date if not provided by caller
    eff_track = track or result.card.track
    eff_date = race_date or result.card.race_date

    active_rows = [r for r in rows if not r.get("scratched")]
    scratch_rows = [r for r in rows if r.get("scratched")]

    race_nums = {r["race_number"] for r in active_rows}

    return {
        "races": len(race_nums),
        "entries": len(active_rows),
        "scratches": len(scratch_rows),
        "sample_rows": active_rows[:_PREVIEW_SAMPLE_SIZE],
        "all_rows": rows,
        "confidence": result.confidence,
        "missing_fields": result.missing_fields,
        "track": eff_track,
        "race_date": eff_date,
        "raw_text": result.raw_text[:10000],
    }


def ingest_pdf(
    db: Persistence, pdf_path: str, track: str = "", race_date: str = "",
    session_id: str = "",
) -> dict:
    """Parse an Equibase chart PDF, return preview, and auto-ingest if high confidence.

    Always returns preview data.  Adds ``ingested: True`` and DB summary
    fields when confidence >= threshold; otherwise ``needs_review: True``.
    """
    preview = preview_pdf(pdf_path, track=track, race_date=race_date)

    eff_track = preview["track"]
    eff_date = preview["race_date"]

    base = {
        "preview_races": preview["races"],
        "preview_entries": preview["entries"],
        "preview_scratches": preview["scratches"],
        "sample_rows": preview["sample_rows"],
        "parse_confidence": preview["confidence"],
        "missing_fields": preview["missing_fields"],
        "track": eff_track,
        "race_date": eff_date,
    }

    if preview["confidence"] >= PDF_CONFIDENCE_THRESHOLD:
        summary = _ingest_rows(
            db, preview["all_rows"], track=eff_track, race_date=eff_date,
            session_id=session_id,
        )
        return {
            **base,
            "ingested": True,
            "needs_review": False,
            **summary,
        }
    else:
        return {
            **base,
            "ingested": False,
            "needs_review": True,
            "extracted_rows": preview["all_rows"],
            "raw_text": preview["raw_text"],
        }


def ingest_rows(
    db: Persistence, rows: list, track: str = "", race_date: str = "",
    session_id: str = "", program_map: str = "",
) -> dict:
    """Public entry for confirmed PDF data (from fallback mapping UI)."""
    return _ingest_rows(db, rows, track=track, race_date=race_date,
                        session_id=session_id, program_map=program_map)


def main():
    ap = argparse.ArgumentParser(description="Ingest race results CSV")
    ap.add_argument("--csv", required=True, help="Path to results CSV")
    ap.add_argument("--track", default="", help="Track code (e.g. GP)")
    ap.add_argument("--date", default="", help="Race date (MM/DD/YYYY)")
    ap.add_argument("--db", default="horses.db", help="Database path")
    args = ap.parse_args()

    if not Path(args.csv).exists():
        print(f"File not found: {args.csv}")
        sys.exit(1)

    db = Persistence(Path(args.db))
    result = ingest_csv(db, args.csv, track=args.track, race_date=args.date)

    print(f"Results ingested:")
    print(f"  Races:   {result['races']}")
    print(f"  Entries: {result['entries']}")
    print(f"  Linked:  {result['linked']}")
    print(f"  Link %%:  {result['link_rate']:.1f}%")
    print(f"  Errors:  {result['errors']}")
    if result.get("unmatched"):
        print(f"\nUnmatched entries ({len(result['unmatched'])}):")
        for u in result["unmatched"][:10]:
            print(f"  R{u['race']} P{u['post']} {u['name']}: {u['reason']}")

    # Show quick stats
    stats = db.get_results_stats(track=args.track)
    print(f"\nResults DB:")
    print(f"  Total races:     {stats['total_races']}")
    print(f"  Total entries:   {stats['total_entries']}")
    print(f"  Linking rate:    {stats['linking_rate']:.1f}%")
    if stats["roi_by_rank"]:
        print(f"\nROI by pick rank ($2 win):")
        for r in stats["roi_by_rank"]:
            print(f"  Rank {r['rank']}: {r['bets']} bets, {r['wins']} wins "
                  f"({r['win_pct']}%), ROI {r['roi_pct']:+.1f}%")


if __name__ == "__main__":
    main()
