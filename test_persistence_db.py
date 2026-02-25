"""Tests for the cumulative horse database (persistence.py)."""
import pytest
from pathlib import Path
from persistence import Persistence


@pytest.fixture
def db(tmp_path):
    """Fresh DB for each test."""
    return Persistence(tmp_path / "test.db")


# ==================================================================
# Uploads
# ==================================================================

class TestUploads:
    def test_record_and_list(self, db):
        db.record_upload("u1", "sheets", "card.pdf", "abc123", "GP", "02/26")
        uploads = db.list_uploads()
        assert len(uploads) == 1
        assert uploads[0]["source_type"] == "sheets"
        assert uploads[0]["pdf_hash"] == "abc123"

    def test_duplicate_hash_detected(self, db):
        db.record_upload("u1", "sheets", "card.pdf", "abc123", "GP", "02/26")
        found = db.get_upload_by_hash("abc123")
        assert found is not None
        assert found["upload_id"] == "u1"

    def test_no_hash_returns_none(self, db):
        assert db.get_upload_by_hash("") is None
        assert db.get_upload_by_hash("nonexistent") is None

    def test_list_filter_by_type(self, db):
        db.record_upload("u1", "sheets", "a.pdf", "h1", "GP", "02/26")
        db.record_upload("u2", "brisnet", "b.pdf", "h2", "GP", "02/26")
        sheets = db.list_uploads(source_type="sheets")
        assert len(sheets) == 1
        assert sheets[0]["source_type"] == "sheets"


# ==================================================================
# Sheets ingestion
# ==================================================================

class TestSheetsIngestion:
    def _make_horse(self, name="TEST HORSE", race_num=1, age=3, lines=None):
        if lines is None:
            lines = [
                {"parsed_figure": 15.0, "surface": "DIRT", "track": "GP",
                 "flags": ["FAST"], "data_text": "", "race_date": "02/26", "race_type": ""},
                {"parsed_figure": 18.0, "surface": "POLY", "track": "AWGP",
                 "flags": [], "data_text": "", "race_date": "02/19", "race_type": ""},
            ]
        return {
            "horse_name": name, "race_number": race_num,
            "age": age, "sex": "M", "top_fig": "15.0",
            "lines": lines,
        }

    def test_ingest_single_horse(self, db):
        db.record_upload("u1", "sheets", "t.pdf", "h1", "GP", "02/26")
        hid = db.ingest_sheets_horse("u1", self._make_horse(), "GP", "02/26")
        assert hid is not None
        assert isinstance(hid, int)

    def test_duplicate_horse_skipped(self, db):
        db.record_upload("u1", "sheets", "a.pdf", "h1", "GP", "02/26")
        db.record_upload("u2", "sheets", "b.pdf", "h2", "GP", "02/26")
        horse = self._make_horse()
        id1 = db.ingest_sheets_horse("u1", horse, "GP", "02/26")
        id2 = db.ingest_sheets_horse("u2", horse, "GP", "02/26")
        assert id1 is not None
        assert id2 is None

    def test_ingest_card(self, db):
        db.record_upload("u1", "sheets", "t.pdf", "h1", "GP", "02/26")
        card = {
            "track_name": "GP",
            "race_date": "02/26",
            "horses": [
                self._make_horse("HORSE A", 1),
                self._make_horse("HORSE B", 2),
                {"horse_name": "No Data Found", "race_number": 0, "lines": []},
            ],
        }
        result = db.ingest_sheets_card("u1", card)
        assert result["inserted"] == 2
        assert result["skipped"] == 0

    def test_load_all_returns_correct_shape(self, db):
        db.record_upload("u1", "sheets", "t.pdf", "h1", "GP", "02/26")
        db.ingest_sheets_horse("u1", self._make_horse(), "GP", "02/26")
        horses = db.load_all_sheets_horses()
        assert len(horses) == 1
        assert horses[0]["horse_name"] == "TEST HORSE"
        assert len(horses[0]["lines"]) == 2
        assert horses[0]["lines"][0]["parsed_figure"] == 15.0
        assert horses[0]["lines"][0]["surface"] == "DIRT"

    def test_load_filter_by_track(self, db):
        db.record_upload("u1", "sheets", "t.pdf", "h1", "GP", "02/26")
        db.record_upload("u2", "sheets", "t2.pdf", "h2", "CD", "02/26")
        db.ingest_sheets_horse("u1", self._make_horse("A"), "GP", "02/26")
        db.ingest_sheets_horse("u2", self._make_horse("B", 2), "CD", "02/26")
        gp_only = db.load_all_sheets_horses(track="GP")
        assert len(gp_only) == 1
        assert gp_only[0]["horse_name"] == "A"

    def test_load_min_lines(self, db):
        db.record_upload("u1", "sheets", "t.pdf", "h1", "GP", "02/26")
        db.ingest_sheets_horse("u1", self._make_horse("SHORT", lines=[
            {"parsed_figure": 10.0, "surface": "DIRT", "track": "GP",
             "flags": [], "data_text": "", "race_date": "", "race_type": ""},
        ]), "GP", "02/26")
        db.ingest_sheets_horse("u2", self._make_horse("LONG", race_num=2), "GP", "02/26")
        horses = db.load_all_sheets_horses(min_lines=2)
        assert len(horses) == 1
        assert horses[0]["horse_name"] == "LONG"


# ==================================================================
# BRISNET ingestion
# ==================================================================

class TestBrisnetIngestion:
    def _make_brisnet_horse(self, name="FAST RUNNER", post=3, race_num=1):
        return {
            "horse_name": name, "race_number": race_num, "post": post,
            "runstyle": "E", "runstyle_rating": 95,
            "prime_power": 122.5,
            "sire": "BIG SIRE", "dam": "FAST DAM", "breeder": "ACME",
            "weight": 118, "life_starts": 5,
            "life_record": "2-1-1", "life_earnings": "$50,000",
            "life_speed": 88, "surface_records": {"dirt": "2-1-0"},
            "lines": [
                {"race_date": "04Dec25", "track": "GP", "surface": "DIRT",
                 "parsed_figure": 72, "race_type": "Mdn"},
            ],
            "workouts": [
                {"date": "08Feb", "track": "GP", "distance": "5f",
                 "surface": "ft", "time": "1:01.2", "rank_letter": "B", "rank": "3/12"},
            ],
        }

    def test_ingest_brisnet_horse(self, db):
        db.record_upload("u1", "brisnet", "t.pdf", "h1", "GP", "02/19")
        race = {"track": "GP", "race_date": "02/19", "race_number": 1, "restrictions": "3yo Fillies"}
        bid = db.ingest_brisnet_horse("u1", self._make_brisnet_horse(), race)
        assert bid is not None

    def test_brisnet_duplicate_skipped(self, db):
        db.record_upload("u1", "brisnet", "a.pdf", "h1", "GP", "02/19")
        db.record_upload("u2", "brisnet", "b.pdf", "h2", "GP", "02/19")
        race = {"track": "GP", "race_date": "02/19", "race_number": 1, "restrictions": ""}
        h = self._make_brisnet_horse()
        id1 = db.ingest_brisnet_horse("u1", h, race)
        id2 = db.ingest_brisnet_horse("u2", h, race)
        assert id1 is not None
        assert id2 is None

    def test_ingest_brisnet_card(self, db):
        db.record_upload("u1", "brisnet", "t.pdf", "h1", "GP", "02/19")
        pipeline = {
            "track_name": "Gulfstream Park",
            "race_date": "02/19",
            "horses": [
                self._make_brisnet_horse("A", 1, 1),
                self._make_brisnet_horse("B", 2, 1),
            ],
        }
        races = {
            "races": [{
                "race_number": 1,
                "conditions": {"restrictions": "3yo Fillies"},
            }],
        }
        result = db.ingest_brisnet_card("u1", pipeline, races)
        assert result["inserted"] == 2

    def test_age_from_restrictions(self, db):
        assert db._age_from_restrictions("3yo Fillies") == 3
        assert db._age_from_restrictions("4&up") == 4
        assert db._age_from_restrictions("Open") == 0


# ==================================================================
# Reconciliation
# ==================================================================

class TestReconciliation:
    def test_high_confidence_match(self, db):
        db.record_upload("u1", "sheets", "a.pdf", "h1", "GP", "02/26")
        db.record_upload("u2", "brisnet", "b.pdf", "h2", "GP", "02/26")

        db.ingest_sheets_horse("u1", {
            "horse_name": "FAST RUNNER", "race_number": 1, "post": "3",
            "age": 3, "lines": [],
        }, "GP", "02/26")

        db.ingest_brisnet_horse("u2", {
            "horse_name": "Fast Runner", "race_number": 1, "post": 3,
            "runstyle": "E", "sire": "BIG SIRE", "dam": "FAST DAM",
            "lines": [], "workouts": [],
        }, {"track": "GP", "race_date": "02/26", "race_number": 1, "restrictions": "3yo"})

        result = db.reconcile("u2")
        assert result["new_matches"] >= 1

    def test_name_fallback_match(self, db):
        db.record_upload("u1", "sheets", "a.pdf", "h1", "GP", "02/26")
        db.record_upload("u2", "brisnet", "b.pdf", "h2", "GP", "02/26")

        # Same name + track + date, different post
        db.ingest_sheets_horse("u1", {
            "horse_name": "KING'S HONOR", "race_number": 5, "post": "7",
            "age": 4, "lines": [],
        }, "GP", "02/26")

        db.ingest_brisnet_horse("u2", {
            "horse_name": "King's Honor", "race_number": 5, "post": 2,
            "runstyle": "P", "sire": "", "dam": "",
            "lines": [], "workouts": [],
        }, {"track": "GP", "race_date": "02/26", "race_number": 5, "restrictions": ""})

        result = db.reconcile("u2")
        assert result["new_matches"] >= 1

    def test_cross_date_low_match(self, db):
        """Same unique name on different dates matches at LOW confidence."""
        db.record_upload("u1", "sheets", "a.pdf", "h1", "GP", "02/26")
        db.record_upload("u2", "brisnet", "b.pdf", "h2", "GP", "02/19")

        db.ingest_sheets_horse("u1", {
            "horse_name": "SAME NAME", "race_number": 1, "post": "1",
            "age": 3, "lines": [],
        }, "GP", "02/26")

        db.ingest_brisnet_horse("u2", {
            "horse_name": "Same Name", "race_number": 1, "post": 1,
            "runstyle": "E", "sire": "", "dam": "",
            "lines": [], "workouts": [],
        }, {"track": "GP", "race_date": "02/19", "race_number": 1, "restrictions": ""})

        result = db.reconcile("")
        assert result["new_matches"] == 1  # LOW tier: unique name match
        row = db.conn.execute("SELECT confidence FROM reconciliation LIMIT 1").fetchone()
        assert row["confidence"] == "low"

    def test_reconcile_idempotent(self, db):
        db.record_upload("u1", "sheets", "a.pdf", "h1", "GP", "02/26")
        db.record_upload("u2", "brisnet", "b.pdf", "h2", "GP", "02/26")

        db.ingest_sheets_horse("u1", {
            "horse_name": "HORSE A", "race_number": 1, "post": "1",
            "age": 3, "lines": [],
        }, "GP", "02/26")
        db.ingest_brisnet_horse("u2", {
            "horse_name": "Horse A", "race_number": 1, "post": 1,
            "runstyle": "E", "sire": "", "dam": "",
            "lines": [], "workouts": [],
        }, {"track": "GP", "race_date": "02/26", "race_number": 1, "restrictions": ""})

        r1 = db.reconcile("")
        r2 = db.reconcile("")
        assert r2["new_matches"] == 0  # second run finds nothing new


# ==================================================================
# Load enriched
# ==================================================================

class TestLoadEnriched:
    def test_enriched_horse_has_brisnet_data(self, db):
        db.record_upload("u1", "sheets", "a.pdf", "h1", "GP", "02/26")
        db.record_upload("u2", "brisnet", "b.pdf", "h2", "GP", "02/26")

        db.ingest_sheets_horse("u1", {
            "horse_name": "POLY STAR", "race_number": 3, "post": "1",
            "age": 3,
            "lines": [
                {"parsed_figure": 20.0, "surface": "POLY", "track": "AWGP",
                 "flags": [], "data_text": "", "race_date": "", "race_type": ""},
            ],
        }, "GP", "02/26")

        db.ingest_brisnet_horse("u2", {
            "horse_name": "Poly Star", "race_number": 3, "post": 1,
            "runstyle": "S", "sire": "GOLD SIRE", "dam": "SILVER DAM",
            "lines": [{"race_date": "04Dec25", "track": "GP", "surface": "DIRT",
                        "parsed_figure": 72}],
            "workouts": [],
        }, {"track": "GP", "race_date": "02/26", "race_number": 3, "restrictions": "3yo"})

        db.reconcile("")

        horses, enrichment = db.load_enriched_horses()
        assert len(horses) == 1
        assert "_enrichment" in horses[0]
        assert horses[0]["_enrichment"]["sire"] == "GOLD SIRE"
        assert horses[0]["_enrichment"]["runstyle"] == "S"
        assert "POLY STAR" in enrichment

    def test_unenriched_horse_has_no_enrichment(self, db):
        db.record_upload("u1", "sheets", "a.pdf", "h1", "GP", "02/26")
        db.ingest_sheets_horse("u1", {
            "horse_name": "LONELY", "race_number": 1, "post": "1",
            "age": 0, "lines": [],
        }, "GP", "02/26")

        horses, enrichment = db.load_enriched_horses()
        assert len(horses) == 1
        assert "_enrichment" not in horses[0]
        assert len(enrichment) == 0


# ==================================================================
# Stats
# ==================================================================

class TestDbStats:
    def test_empty_stats(self, db):
        stats = db.get_db_stats()
        assert stats["sheets_horses"] == 0
        assert stats["brisnet_horses"] == 0
        assert stats["reconciled_pairs"] == 0
        assert stats["coverage_pct"] == 0.0
        assert stats["tracks"] == []

    def test_stats_with_data(self, db):
        db.record_upload("u1", "sheets", "a.pdf", "h1", "GP", "02/26")
        db.ingest_sheets_horse("u1", {
            "horse_name": "A", "race_number": 1, "age": 3,
            "lines": [
                {"parsed_figure": 10.0, "surface": "DIRT", "track": "GP",
                 "flags": [], "data_text": "", "race_date": "", "race_type": ""},
            ],
        }, "GP", "02/26")

        stats = db.get_db_stats()
        assert stats["sheets_horses"] == 1
        assert stats["sheets_lines"] == 1
        assert stats["uploads_count"] == 1
        assert "GP" in stats["tracks"]


# ==================================================================
# Track + name normalization
# ==================================================================

class TestNormalization:
    def test_normalize_name(self, db):
        assert db._normalize_name("Tiger Moon") == "TIGER MOON"
        assert db._normalize_name("K's Pick") == "KS PICK"
        assert db._normalize_name("  Curlin's Gesture  ") == "CURLINS GESTURE"

    def test_normalize_track_short_code(self, db):
        assert db._normalize_track("GP") == "GP"
        assert db._normalize_track("CD") == "CD"

    def test_normalize_track_full_name(self, db):
        assert db._normalize_track("Gulfstream Park") == "GP"
        assert db._normalize_track("Churchill Downs") == "CD"
        assert db._normalize_track("Santa Anita") == "SA"

    def test_normalize_track_unknown(self, db):
        assert db._normalize_track("Some Unknown Track") == "SOME UNKNOWN TRACK"

    def test_normalize_track_empty(self, db):
        assert db._normalize_track("") == ""

    def test_normalize_date_already_mm_dd_yyyy(self, db):
        assert db._normalize_date("02/26/2026") == "02/26/2026"

    def test_normalize_date_full_month(self, db):
        assert db._normalize_date("February 19, 2026") == "02/19/2026"

    def test_normalize_date_short_ragozin(self, db):
        result = db._normalize_date("29JAN")
        assert result.startswith("01/29/")  # month/day correct, year = current

    def test_normalize_date_ragozin_with_year(self, db):
        assert db._normalize_date("06JUN25") == "06/06/2025"

    def test_normalize_date_unknown(self, db):
        assert db._normalize_date("Unknown Date") == "Unknown Date"

    def test_normalize_date_empty(self, db):
        assert db._normalize_date("") == ""


# ==================================================================
# Extended reconciliation tiers + aliases
# ==================================================================

class TestReconTiers:
    """Tests for the 4-tier reconciliation and alias support."""

    def _setup_sheets_horse(self, db, name, race_num=1, post="3", track="GP", date="02/26/2026"):
        db.record_upload("us1", "sheets", "s.pdf", None, track, date)
        db.ingest_sheets_horse("us1", {
            "horse_name": name, "race_number": race_num, "post": post,
            "age": 3, "lines": [],
        }, track, date)

    def _setup_brisnet_horse(self, db, name, race_num=1, post=3, track="GP", date="02/26/2026"):
        db.record_upload("ub1", "brisnet", "b.pdf", None, track, date)
        db.ingest_brisnet_horse("ub1", {
            "horse_name": name, "race_number": race_num, "post": post,
            "runstyle": "E", "sire": "SIRE", "dam": "DAM",
            "lines": [], "workouts": [],
        }, {"track": track, "race_date": date, "race_number": race_num, "restrictions": ""})

    def test_high_confidence_post_match(self, db):
        """Pass 1: exact track+date+race+post should produce HIGH."""
        self._setup_sheets_horse(db, "ALPHA", race_num=3, post="5")
        self._setup_brisnet_horse(db, "Alpha", race_num=3, post=5)
        result = db.reconcile()
        assert result["new_matches"] >= 1
        row = db.conn.execute(
            "SELECT confidence FROM reconciliation LIMIT 1"
        ).fetchone()
        assert row["confidence"] == "high"

    def test_alias_resolves_match(self, db):
        """An alias should let a different Sheets name match a BRISNET name."""
        self._setup_sheets_horse(db, "FLYIN TIGER", race_num=2, post="0", track="GP", date="02/26/2026")
        self._setup_brisnet_horse(db, "Flying Tiger", race_num=2, post=0, track="GP", date="02/26/2026")
        # Without alias, names differ: FLYIN TIGER vs FLYING TIGER
        r1 = db.reconcile()
        assert r1["existing"] == 0
        # Add alias
        db.add_alias("FLYING TIGER", "FLYIN TIGER")
        r2 = db.reconcile()
        assert r2["new_matches"] >= 1

    def test_low_refuses_ambiguous_collision(self, db):
        """LOW tier should NOT match when a name appears multiple times on one side."""
        # Two sheets horses with same name but different races (different tracks/dates)
        db.record_upload("us1", "sheets", "s.pdf", None, "GP", "02/26/2026")
        db.record_upload("us2", "sheets", "s2.pdf", None, "CD", "03/01/2026")
        db.ingest_sheets_horse("us1", {
            "horse_name": "DUPLICATE NAME", "race_number": 1, "post": "1",
            "age": 3, "lines": [],
        }, "GP", "02/26/2026")
        db.ingest_sheets_horse("us2", {
            "horse_name": "DUPLICATE NAME", "race_number": 1, "post": "1",
            "age": 3, "lines": [],
        }, "CD", "03/01/2026")
        # One brisnet horse with same name, different track entirely
        db.record_upload("ub1", "brisnet", "b.pdf", None, "SA", "04/01/2026")
        db.ingest_brisnet_horse("ub1", {
            "horse_name": "Duplicate Name", "race_number": 1, "post": 1,
            "runstyle": "E", "sire": "", "dam": "",
            "lines": [], "workouts": [],
        }, {"track": "SA", "race_date": "04/01/2026", "race_number": 1, "restrictions": ""})

        result = db.reconcile()
        # Should NOT match — two sheets horses have the same name, ambiguous
        assert result["existing"] == 0


# ==================================================================
# Session stats + auto-reconcile
# ==================================================================

class TestSessionStats:
    def test_session_stats_returns_scoped_data(self, db):
        """get_session_stats returns counts scoped to that upload only."""
        # Session 1: sheets
        db.record_upload("sess1", "sheets", "a.pdf", "h1", "GP", "02/26/2026",
                         session_id="sess1")
        db.ingest_sheets_horse("sess1", {
            "horse_name": "STAR", "race_number": 1, "post": "1",
            "age": 3, "lines": [],
        }, "GP", "02/26/2026")
        # Session 2: brisnet (same card)
        db.record_upload("sess2", "brisnet", "b.pdf", "h2", "GP", "02/26/2026",
                         session_id="sess1")
        db.ingest_brisnet_horse("sess2", {
            "horse_name": "Star", "race_number": 1, "post": 1,
            "runstyle": "E", "sire": "S", "dam": "D",
            "lines": [], "workouts": [],
        }, {"track": "GP", "race_date": "02/26/2026", "race_number": 1, "restrictions": ""})

        db.reconcile()

        stats = db.get_session_stats("sess1")
        assert stats["sheets_horses"] == 1
        assert stats["reconciled_pairs"] >= 1
        assert stats["coverage_pct"] > 0
        assert "confidence_breakdown" in stats

    def test_auto_reconcile_on_ingest(self, db):
        """Reconcile after each ingest picks up matches automatically."""
        db.record_upload("u1", "sheets", "s.pdf", None, "GP", "02/26/2026")
        db.ingest_sheets_horse("u1", {
            "horse_name": "GOLDEN BOY", "race_number": 2, "post": "4",
            "age": 3, "lines": [],
        }, "GP", "02/26/2026")
        r1 = db.reconcile()
        assert r1["existing"] == 0  # no brisnet yet

        # Now add brisnet
        db.record_upload("u2", "brisnet", "b.pdf", None, "GP", "02/26/2026")
        db.ingest_brisnet_horse("u2", {
            "horse_name": "Golden Boy", "race_number": 2, "post": 4,
            "runstyle": "P", "sire": "GOLD", "dam": "LADY",
            "lines": [], "workouts": [],
        }, {"track": "GP", "race_date": "02/26/2026", "race_number": 2, "restrictions": ""})
        r2 = db.reconcile()
        assert r2["new_matches"] >= 1  # auto-matched on reconcile


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
