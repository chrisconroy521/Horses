#!/usr/bin/env python3
"""One-time migration: copy all data from local horses.db to a Postgres instance.

Usage:
    DATABASE_URL=postgresql://user:pass@host:5432/dbname python scripts/migrate_sqlite_to_postgres.py

The script:
  1. Connects to the local horses.db (SQLite)
  2. Connects to the Postgres instance via DATABASE_URL
  3. Creates the schema (via Persistence.__init__)
  4. Copies all rows from each table
  5. Resets Postgres sequences to match max IDs
"""
import os
import sys
import sqlite3
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: Set DATABASE_URL env var to the target Postgres connection string.")
    sys.exit(1)

SQLITE_PATH = Path(__file__).resolve().parent.parent / "horses.db"
if not SQLITE_PATH.exists():
    print(f"ERROR: {SQLITE_PATH} not found.")
    sys.exit(1)

# Import Persistence to create schema on Postgres
from persistence import Persistence

print(f"[1/4] Initializing Postgres schema via Persistence...")
pg_db = Persistence()  # picks up DATABASE_URL automatically
assert pg_db.db_backend == "postgres", "DATABASE_URL not detected — got sqlite backend"
print("       Schema created.")

# Connect to both databases directly for raw copy
import psycopg2
import psycopg2.extras

pg_url = DATABASE_URL
if pg_url.startswith("postgres://"):
    pg_url = pg_url.replace("postgres://", "postgresql://", 1)
pg_conn = psycopg2.connect(pg_url)
pg_conn.autocommit = False
pg_cur = pg_conn.cursor()

sq_conn = sqlite3.connect(str(SQLITE_PATH))
sq_conn.row_factory = sqlite3.Row

# Tables to migrate (order matters for foreign keys)
TABLES = [
    "sessions",
    "race_inputs",
    "race_outputs",
    "race_results",
    "uploads",
    "sheets_horses",
    "sheets_lines",
    "brisnet_horses",
    "brisnet_lines",
    "brisnet_workouts",
    "reconciliation",
    "horse_aliases",
    "result_races",
    "result_entries",
    "result_predictions",
    "bet_plans",
    "odds_snapshots",
]

# Serial (auto-increment) primary key columns per table
SERIAL_COLUMNS = {
    "sheets_horses": "horse_id",
    "sheets_lines": "line_id",
    "brisnet_horses": "brisnet_id",
    "brisnet_lines": "bline_id",
    "brisnet_workouts": "workout_id",
    "reconciliation": "recon_id",
    "result_races": "result_id",
    "result_entries": "entry_result_id",
    "result_predictions": "prediction_id",
    "bet_plans": "plan_id",
    "odds_snapshots": "snapshot_id",
}

print(f"[2/4] Migrating {len(TABLES)} tables from {SQLITE_PATH}...")

for table in TABLES:
    rows = sq_conn.execute(f"SELECT * FROM {table}").fetchall()
    if not rows:
        print(f"       {table}: 0 rows (skip)")
        continue

    columns = rows[0].keys()
    col_list = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))

    # Clear existing data (in case of re-run)
    pg_cur.execute(f"DELETE FROM {table}")

    for row in rows:
        values = tuple(row[c] for c in columns)
        try:
            pg_cur.execute(
                f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",
                values,
            )
        except Exception as e:
            print(f"       WARN: {table} row failed: {e}")

    pg_conn.commit()
    print(f"       {table}: {len(rows)} rows migrated")

print(f"[3/4] Resetting Postgres sequences...")
for table, pk_col in SERIAL_COLUMNS.items():
    try:
        seq_name = f"{table}_{pk_col}_seq"
        pg_cur.execute(f"SELECT MAX({pk_col}) FROM {table}")
        max_val = pg_cur.fetchone()[0]
        if max_val is not None:
            pg_cur.execute(f"SELECT setval('{seq_name}', {max_val})")
            print(f"       {seq_name} -> {max_val}")
        pg_conn.commit()
    except Exception as e:
        pg_conn.rollback()
        print(f"       WARN: could not reset {table}.{pk_col} seq: {e}")

# Verify counts
print(f"[4/4] Verifying row counts...")
ok = True
for table in TABLES:
    sq_count = sq_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    pg_cur.execute(f"SELECT COUNT(*) FROM {table}")
    pg_count = pg_cur.fetchone()[0]
    status = "OK" if sq_count == pg_count else "MISMATCH"
    if status != "OK":
        ok = False
    print(f"       {table}: sqlite={sq_count} postgres={pg_count} [{status}]")

pg_conn.close()
sq_conn.close()

if ok:
    print("\nMigration complete. All row counts match.")
else:
    print("\nMigration complete with MISMATCHES — review above.")
