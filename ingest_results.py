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


def ingest_csv(
    db: Persistence, csv_path: str, track: str = "", race_date: str = "",
    session_id: str = "",
) -> dict:
    """Parse a results CSV and insert into DB. Returns summary dict."""
    races_seen = set()
    entries_inserted = 0
    errors = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rn = int(row.get("race_number", 0))
                post = int(row.get("post", 0))
                name = row.get("horse_name", "").strip()
                finish = int(row.get("finish_pos", 0) or 0)

                r_track = row.get("track", "").strip() or track
                r_date = row.get("race_date", "").strip() or race_date

                if not r_track or not r_date or not rn:
                    errors += 1
                    continue

                # Parse optional numeric fields
                def _float(val):
                    try:
                        return float(val) if val and val.strip() else None
                    except (ValueError, TypeError):
                        return None

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
                        surface=row.get("surface", ""),
                        distance=row.get("distance", ""),
                        winner_post=winner_post, winner_name=winner_name,
                    )
                    races_seen.add(race_key)
                elif finish == 1:
                    db.insert_race_result(
                        track=r_track, race_date=r_date, race_number=rn,
                        surface=row.get("surface", ""),
                        distance=row.get("distance", ""),
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
