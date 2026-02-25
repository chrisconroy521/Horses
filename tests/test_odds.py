"""Tests for BRISNET morning line odds parsing, persistence, and bet builder integration."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from brisnet_parser import parse_odds_decimal
from persistence import Persistence
from bet_builder import BetSettings, build_day_plan


class TestParseOddsDecimal:
    def test_fraction_simple(self):
        assert parse_odds_decimal("3/1") == 3.0

    def test_fraction_non_unit(self):
        assert abs(parse_odds_decimal("9/5") - 1.8) < 1e-9

    def test_decimal_string(self):
        assert parse_odds_decimal("5.0") == 5.0

    def test_asterisk_prefix(self):
        assert parse_odds_decimal("*6.5") == 6.5

    def test_dash_format(self):
        assert parse_odds_decimal("4-1") == 4.0

    def test_even(self):
        assert parse_odds_decimal("even") == 1.0

    def test_evs(self):
        assert parse_odds_decimal("evs") == 1.0

    def test_empty(self):
        assert parse_odds_decimal("") is None

    def test_none_input(self):
        assert parse_odds_decimal(None) is None

    def test_garbage(self):
        assert parse_odds_decimal("SCR") is None

    def test_asterisk_only(self):
        assert parse_odds_decimal("*") is None

    def test_fraction_2_5(self):
        assert abs(parse_odds_decimal("2/5") - 0.4) < 1e-9


class TestOddsSnapshotPersistence:
    def test_save_and_retrieve(self, tmp_path):
        db = Persistence(tmp_path / "test.db")
        snapshots = [
            {"race_number": 1, "post": 3, "horse_name": "Alpha", "odds_raw": "3/1", "odds_decimal": 3.0},
            {"race_number": 1, "post": 5, "horse_name": "Beta", "odds_raw": "9/5", "odds_decimal": 1.8},
            {"race_number": 2, "post": 1, "horse_name": "Gamma", "odds_raw": "even", "odds_decimal": 1.0},
        ]
        count = db.save_odds_snapshots(
            session_id="s1", track="GP", race_date="2026-02-26",
            snapshots=snapshots, source="morning_line",
        )
        assert count == 3

        result = db.get_odds_snapshots(track="GP", race_date="2026-02-26", source="morning_line")
        assert (1, "ALPHA") in result
        assert result[(1, "ALPHA")] == 3.0
        assert abs(result[(1, "BETA")] - 1.8) < 1e-9
        assert result[(2, "GAMMA")] == 1.0

    def test_upsert(self, tmp_path):
        db = Persistence(tmp_path / "test.db")
        snap1 = [{"race_number": 1, "post": 3, "horse_name": "Alpha", "odds_raw": "3/1", "odds_decimal": 3.0}]
        snap2 = [{"race_number": 1, "post": 3, "horse_name": "Alpha", "odds_raw": "5/1", "odds_decimal": 5.0}]
        db.save_odds_snapshots("s1", "GP", "2026-02-26", snap1)
        db.save_odds_snapshots("s1", "GP", "2026-02-26", snap2)
        result = db.get_odds_snapshots("GP", "2026-02-26")
        assert result[(1, "ALPHA")] == 5.0

    def test_null_odds_filtered(self, tmp_path):
        db = Persistence(tmp_path / "test.db")
        snapshots = [
            {"race_number": 1, "post": 3, "horse_name": "Alpha", "odds_raw": "", "odds_decimal": None},
        ]
        db.save_odds_snapshots("s1", "GP", "2026-02-26", snapshots)
        result = db.get_odds_snapshots("GP", "2026-02-26")
        assert len(result) == 0


class TestBetBuilderMLOdds:
    def _make_preds(self, odds=None):
        """Create a minimal set of predictions for one A-grade race."""
        return {
            1: [
                {
                    "horse_name": "Alpha",
                    "race_number": 1,
                    "pick_rank": 1,
                    "projection_type": "PAIRED",
                    "bias_score": 12.0,
                    "confidence": 0.72,
                    "projected_low": 10.0,
                    "projected_high": 12.0,
                    "tags": [],
                    "new_top_setup": False,
                    "bounce_risk": False,
                    "tossed": False,
                    "odds": odds,
                    "post": 3,
                },
                {
                    "horse_name": "Beta",
                    "race_number": 1,
                    "pick_rank": 2,
                    "projection_type": "NEUTRAL",
                    "bias_score": 8.0,
                    "confidence": 0.45,
                    "projected_low": 14.0,
                    "projected_high": 16.0,
                    "tags": [],
                    "new_top_setup": False,
                    "bounce_risk": False,
                    "tossed": False,
                    "odds": None,
                    "post": 5,
                },
            ]
        }

    def test_ml_odds_enables_kelly(self):
        """With ML odds injected, bet builder should use Kelly sizing, not flat stake."""
        preds = self._make_preds(odds=5.0)
        settings = BetSettings(bankroll=1000, min_odds_a=2.0, paper_mode=True)
        plan = build_day_plan(preds, settings)
        # Should have a WIN ticket with Kelly method
        tickets = plan.race_plans[0].tickets
        win_tickets = [t for t in tickets if t.bet_type == "WIN"]
        assert len(win_tickets) == 1
        assert win_tickets[0].details.get("method") == "kelly"

    def test_no_odds_blocked(self):
        """With no odds and allow_missing_odds=False, should be blocked."""
        preds = self._make_preds(odds=None)
        settings = BetSettings(bankroll=1000, min_odds_a=2.0, paper_mode=True, allow_missing_odds=False)
        plan = build_day_plan(preds, settings)
        rp = plan.race_plans[0]
        assert any("BRISNET ML" in b for b in rp.blockers)

    def test_flat_stake_with_missing_allowed(self):
        """With allow_missing_odds=True and no odds, should get flat-stake WIN."""
        preds = self._make_preds(odds=None)
        settings = BetSettings(bankroll=1000, paper_mode=True, allow_missing_odds=True)
        plan = build_day_plan(preds, settings)
        tickets = plan.race_plans[0].tickets
        win_tickets = [t for t in tickets if t.bet_type == "WIN"]
        assert len(win_tickets) == 1
        assert win_tickets[0].details.get("method") == "flat"


class TestPipelineJsonOddsDecimal:
    def test_odds_decimal_in_output(self):
        from brisnet_parser import BrisnetCard, BrisnetRace, BrisnetHorse, to_pipeline_json
        horse = BrisnetHorse(post=1, name="Test Horse", odds="3/1")
        race = BrisnetRace(track="GP", date="2026-02-26", race_number=1, horses=[horse])
        card = BrisnetCard(track="GP", date="2026-02-26", races=[race])
        result = to_pipeline_json(card)
        horses = result.get("horses", [])
        assert len(horses) == 1
        assert horses[0]["odds"] == "3/1"
        assert horses[0]["odds_decimal"] == 3.0

    def test_odds_decimal_empty(self):
        from brisnet_parser import BrisnetCard, BrisnetRace, BrisnetHorse, to_pipeline_json
        horse = BrisnetHorse(post=1, name="Test Horse", odds="")
        race = BrisnetRace(track="GP", date="2026-02-26", race_number=1, horses=[horse])
        card = BrisnetCard(track="GP", date="2026-02-26", races=[race])
        result = to_pipeline_json(card)
        horses = result.get("horses", [])
        assert horses[0]["odds_decimal"] is None
