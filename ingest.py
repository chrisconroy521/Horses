"""CLI for ingesting PDFs into the cumulative horse database.

Usage:
    python ingest.py --sheets path/to/ragozin.pdf
    python ingest.py --brisnet path/to/brisnet.pdf
    python ingest.py --stats
    python ingest.py --backfill-json
"""
import argparse
import hashlib
import json
import glob
import os
import sys
import uuid
from pathlib import Path

from persistence import Persistence


def compute_pdf_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_sheets(db: Persistence, pdf_path: str) -> None:
    pdf_hash = compute_pdf_hash(pdf_path)
    existing = db.get_upload_by_hash(pdf_hash)
    if existing:
        print(f"PDF already ingested (upload_id={existing['upload_id']}). Skipping.")
        return

    from ragozin_parser import RagozinParser
    from api import _trad_to_dict

    parser = RagozinParser()
    race_data = parser.parse_ragozin_sheet(pdf_path)
    parsed_json = _trad_to_dict(race_data)

    upload_id = str(uuid.uuid4())
    track = parsed_json.get("track_name", "")
    race_date = parsed_json.get("race_date", "")

    db.record_upload(
        upload_id=upload_id,
        source_type="sheets",
        pdf_filename=Path(pdf_path).name,
        pdf_hash=pdf_hash,
        track=track,
        race_date=race_date,
        horses_count=len(parsed_json.get("horses", [])),
    )
    result = db.ingest_sheets_card(upload_id, parsed_json)
    recon = db.reconcile(upload_id)

    print(f"Ingested sheets: {result['inserted']} horses inserted, "
          f"{result['skipped']} skipped (duplicates)")
    print(f"Reconciliation: {recon['new_matches']} new matches, "
          f"{recon['existing']} total reconciled pairs")


def ingest_brisnet(db: Persistence, pdf_path: str) -> None:
    pdf_hash = compute_pdf_hash(pdf_path)
    existing = db.get_upload_by_hash(pdf_hash)
    if existing:
        print(f"PDF already ingested (upload_id={existing['upload_id']}). Skipping.")
        return

    from brisnet_parser import BrisnetParser, to_pipeline_json, to_races_json

    bp = BrisnetParser()
    card = bp.parse(pdf_path)
    pipeline_json = to_pipeline_json(card)
    races_json = to_races_json(card)

    upload_id = str(uuid.uuid4())
    track = pipeline_json.get("track_name", "") or pipeline_json.get("track", "")
    race_date = pipeline_json.get("race_date", "") or pipeline_json.get("date", "")

    db.record_upload(
        upload_id=upload_id,
        source_type="brisnet",
        pdf_filename=Path(pdf_path).name,
        pdf_hash=pdf_hash,
        track=track,
        race_date=race_date,
        horses_count=len(pipeline_json.get("horses", [])),
    )
    result = db.ingest_brisnet_card(upload_id, pipeline_json, races_json)
    recon = db.reconcile(upload_id)

    print(f"Ingested BRISNET: {result['inserted']} horses inserted, "
          f"{result['skipped']} skipped (duplicates)")
    print(f"Reconciliation: {recon['new_matches']} new matches, "
          f"{recon['existing']} total reconciled pairs")


def backfill_json(db: Persistence) -> None:
    """One-time import of existing output/*.json files into the DB."""
    count = 0

    # Import Ragozin JSON files
    for json_path in sorted(glob.glob("output/*.json")):
        basename = os.path.basename(json_path)
        # Skip session-specific files
        if "_primary" in basename or "_secondary" in basename:
            continue
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"  Skipping {basename} (decode error)")
            continue

        # Detect source type from JSON
        parse_source = data.get("parse_source", "traditional")
        upload_id = str(uuid.uuid4())
        track = data.get("track_name", "")
        race_date = data.get("race_date", "")
        horses = data.get("horses", [])

        if parse_source == "brisnet":
            db.record_upload(upload_id, "brisnet", basename, "", track, race_date,
                             horses_count=len(horses))
            db.ingest_brisnet_card(upload_id, data, {})
        else:
            db.record_upload(upload_id, "sheets", basename, "", track, race_date,
                             horses_count=len(horses))
            db.ingest_sheets_card(upload_id, data)
        count += 1

    # Import BRISNET pipeline.json if it exists
    pipe_path = "brisnet_output/pipeline.json"
    races_path = "brisnet_output/races.json"
    if os.path.exists(pipe_path):
        try:
            with open(pipe_path, encoding="utf-8") as f:
                pipeline = json.load(f)
            races_data = {}
            if os.path.exists(races_path):
                with open(races_path, encoding="utf-8") as f:
                    races_data = json.load(f)

            upload_id = str(uuid.uuid4())
            track = pipeline.get("track_name", "")
            race_date = pipeline.get("race_date", "")
            db.record_upload(upload_id, "brisnet", "pipeline.json", "", track, race_date,
                             horses_count=len(pipeline.get("horses", [])))
            db.ingest_brisnet_card(upload_id, pipeline, races_data)
            count += 1
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  Skipping pipeline.json ({e})")

    # Reconcile everything
    recon = db.reconcile("")
    stats = db.get_db_stats()

    print(f"Backfill complete ({count} files processed):")
    print(f"  Sheets horses:    {stats['sheets_horses']}")
    print(f"  Sheets lines:     {stats['sheets_lines']}")
    print(f"  BRISNET horses:   {stats['brisnet_horses']}")
    print(f"  BRISNET lines:    {stats['brisnet_lines']}")
    print(f"  Reconciled pairs: {stats['reconciled_pairs']}")
    print(f"  Tracks:           {', '.join(stats['tracks']) or 'none'}")


def show_stats(db: Persistence) -> None:
    stats = db.get_db_stats()
    print("Cumulative Database Statistics:")
    print(f"  Sheets horses:    {stats['sheets_horses']}")
    print(f"  Sheets lines:     {stats['sheets_lines']}")
    print(f"  BRISNET horses:   {stats['brisnet_horses']}")
    print(f"  BRISNET lines:    {stats['brisnet_lines']}")
    print(f"  Reconciled pairs: {stats['reconciled_pairs']}")
    print(f"  Uploads:          {stats['uploads_count']}")
    print(f"  Tracks:           {', '.join(stats['tracks']) or 'none'}")
    print(f"  Coverage:         {stats['coverage_pct']:.1f}%")


def main():
    ap = argparse.ArgumentParser(description="Ingest PDFs into cumulative database")
    ap.add_argument("--sheets", metavar="PDF", help="Ingest a Ragozin sheets PDF")
    ap.add_argument("--brisnet", metavar="PDF", help="Ingest a BRISNET PDF")
    ap.add_argument("--stats", action="store_true", help="Show DB statistics")
    ap.add_argument("--backfill-json", action="store_true",
                    help="Import existing output/*.json into DB")
    ap.add_argument("--db", default="horses.db", help="Database path (default: horses.db)")
    args = ap.parse_args()

    db = Persistence(Path(args.db))

    if args.sheets:
        if not os.path.exists(args.sheets):
            print(f"File not found: {args.sheets}")
            sys.exit(1)
        ingest_sheets(db, args.sheets)
    elif args.brisnet:
        if not os.path.exists(args.brisnet):
            print(f"File not found: {args.brisnet}")
            sys.exit(1)
        ingest_brisnet(db, args.brisnet)
    elif args.stats:
        show_stats(db)
    elif args.backfill_json:
        backfill_json(db)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
