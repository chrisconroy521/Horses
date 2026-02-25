"""Tests for bet_builder module — deterministic output, caps, grading, Kelly math."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from bet_builder import (
    BetSettings, kelly_fraction, grade_race,
    build_win_ticket, build_exacta_tickets,
    build_race_plan, build_day_plan,
    day_plan_to_dict, day_plan_to_text, day_plan_to_csv,
    _estimate_win_prob, _risk_multiplier,
    _MAX_KELLY_FRACTION, _BET_BASE,
)


# ---------------------------------------------------------------------------
# Fixture projections
# ---------------------------------------------------------------------------

def _make_proj(name, bias_score, confidence, projection_type="PAIRED",
               odds=None, tossed=False, bounce_risk=False, new_top_setup=False):
    return {
        "name": name,
        "bias_score": bias_score,
        "confidence": confidence,
        "projection_type": projection_type,
        "odds": odds,
        "tossed": tossed,
        "bounce_risk": bounce_risk,
        "new_top_setup": new_top_setup,
        "raw_score": bias_score,
        "projected_low": 70.0,
        "projected_high": 80.0,
        "tags": [],
    }


# A-grade race: high conf, strong cycle, large gap
A_RACE = [
    _make_proj("ALPHA", 12.0, 0.82, "PAIRED", odds=5.0),
    _make_proj("BETA", 8.0, 0.55, "NEUTRAL", odds=8.0),
    _make_proj("GAMMA", 7.5, 0.50, "TAIL_OFF", odds=12.0),
    _make_proj("DELTA", 6.0, 0.40, "NEUTRAL", odds=20.0),
]

# B-grade race: mid confidence
B_RACE = [
    _make_proj("BRAVO", 9.0, 0.60, "IMPROVING", odds=6.0),
    _make_proj("CHARLIE", 7.8, 0.50, "NEUTRAL", odds=10.0),
    _make_proj("FOXTROT", 7.0, 0.40, "NEUTRAL", odds=15.0),
]

# C-grade race: top pick tossed
C_RACE_TOSSED = [
    _make_proj("ECHO", 10.0, 0.70, "PAIRED", odds=3.0, tossed=True),
    _make_proj("HOTEL", 6.0, 0.30, "NEUTRAL", odds=15.0),
]

# C-grade race: low confidence
C_RACE_LOW_CONF = [
    _make_proj("INDIA", 8.0, 0.30, "NEUTRAL", odds=4.0),
    _make_proj("JULIET", 7.0, 0.25, "TAIL_OFF", odds=8.0),
]


class TestKellyFraction:
    def test_positive_edge(self):
        # 5-1 odds, 30% win prob → positive edge
        f = kelly_fraction(5.0, 0.30)
        assert f > 0

    def test_no_edge(self):
        # 2-1 odds, 20% win prob → b*p - q = 2*0.2 - 0.8 = -0.4
        f = kelly_fraction(2.0, 0.20)
        assert f == 0.0

    def test_capped_at_max(self):
        # 10-1 odds, 50% win prob → huge edge, should be capped
        f = kelly_fraction(10.0, 0.50)
        assert f == _MAX_KELLY_FRACTION

    def test_zero_odds(self):
        assert kelly_fraction(0, 0.5) == 0.0

    def test_zero_prob(self):
        assert kelly_fraction(5.0, 0.0) == 0.0

    def test_deterministic(self):
        # Same inputs → same output
        f1 = kelly_fraction(4.0, 0.35)
        f2 = kelly_fraction(4.0, 0.35)
        assert f1 == f2


class TestGradeRace:
    def test_a_grade(self):
        settings = BetSettings()
        grade, reasons = grade_race(A_RACE, settings)
        assert grade == "A"

    def test_b_grade(self):
        settings = BetSettings()
        grade, reasons = grade_race(B_RACE, settings)
        assert grade == "B"

    def test_c_grade_tossed(self):
        settings = BetSettings()
        grade, reasons = grade_race(C_RACE_TOSSED, settings)
        assert grade == "C"
        assert any("TOSSED" in r for r in reasons)

    def test_c_grade_low_conf(self):
        settings = BetSettings()
        grade, reasons = grade_race(C_RACE_LOW_CONF, settings)
        assert grade == "C"

    def test_empty_projections(self):
        grade, reasons = grade_race([], BetSettings())
        assert grade == "C"

    def test_custom_min_confidence(self):
        # Raise min_confidence so A_RACE's 82% still qualifies
        settings = BetSettings(min_confidence=0.80)
        grade, _ = grade_race(A_RACE, settings)
        assert grade == "A"

        # Raise above top pick's confidence → should downgrade
        settings2 = BetSettings(min_confidence=0.90)
        grade2, _ = grade_race(A_RACE, settings2)
        assert grade2 != "A"


class TestBuildWinTicket:
    def test_win_ticket_generated(self):
        settings = BetSettings(bankroll=1000)
        ranked = sorted(A_RACE, key=lambda p: p["bias_score"], reverse=True)
        ticket = build_win_ticket(ranked, "A", settings, 15.0)
        assert ticket is not None
        assert ticket.bet_type == "WIN"
        assert ticket.selections == ["ALPHA"]
        assert ticket.cost >= _BET_BASE
        assert ticket.cost <= 15.0

    def test_odds_too_low(self):
        settings = BetSettings(min_odds_a=10.0)  # require 10-1 minimum
        ranked = sorted(A_RACE, key=lambda p: p["bias_score"], reverse=True)
        ticket = build_win_ticket(ranked, "A", settings, 15.0)
        assert ticket is None  # ALPHA is only 5-1

    def test_no_odds_flat_stake(self):
        no_odds = [_make_proj("NOPRICES", 10.0, 0.80, "PAIRED", odds=None)]
        settings = BetSettings(bankroll=1000)
        ticket = build_win_ticket(no_odds, "A", settings, 15.0)
        assert ticket is not None
        assert ticket.details["method"] == "flat"

    def test_stake_rounded_to_two(self):
        settings = BetSettings(bankroll=1000)
        ranked = sorted(A_RACE, key=lambda p: p["bias_score"], reverse=True)
        ticket = build_win_ticket(ranked, "A", settings, 50.0)
        if ticket:
            assert ticket.cost % _BET_BASE == 0


class TestBuildExactaTickets:
    def test_exacta_only_a_grade(self):
        ranked = sorted(A_RACE, key=lambda p: p["bias_score"], reverse=True)
        tickets = build_exacta_tickets(ranked, "A", 20.0)
        assert len(tickets) >= 1
        assert all(t.bet_type == "EXACTA" for t in tickets)

    def test_no_exacta_for_b(self):
        ranked = sorted(B_RACE, key=lambda p: p["bias_score"], reverse=True)
        tickets = build_exacta_tickets(ranked, "B", 20.0)
        assert len(tickets) == 0

    def test_key_structure(self):
        ranked = sorted(A_RACE, key=lambda p: p["bias_score"], reverse=True)
        tickets = build_exacta_tickets(ranked, "A", 20.0)
        key_ticket = tickets[0]
        assert key_ticket.details["structure"] == "key"
        assert key_ticket.selections[0] == "ALPHA"


class TestBuildRacePlan:
    def test_c_grade_passed(self):
        settings = BetSettings()
        plan = build_race_plan(1, C_RACE_TOSSED, settings, 100.0)
        assert plan.grade == "C"
        assert plan.passed is True
        assert plan.total_cost == 0.0
        assert len(plan.tickets) == 0

    def test_a_grade_has_tickets(self):
        settings = BetSettings(bankroll=1000)
        plan = build_race_plan(1, A_RACE, settings, 15.0)
        assert plan.grade == "A"
        assert plan.passed is False
        assert len(plan.tickets) >= 1
        assert plan.total_cost > 0

    def test_budget_cap_respected(self):
        settings = BetSettings(bankroll=1000)
        plan = build_race_plan(1, A_RACE, settings, 15.0)
        assert plan.total_cost <= 15.0

    def test_zero_remaining_budget(self):
        settings = BetSettings(bankroll=1000)
        plan = build_race_plan(1, A_RACE, settings, 0.0)
        assert plan.passed is True


class TestBuildDayPlan:
    def test_day_plan_structure(self):
        race_projs = {1: A_RACE, 2: B_RACE, 3: C_RACE_TOSSED}
        settings = BetSettings(bankroll=1000)
        plan = build_day_plan(race_projs, settings)
        assert len(plan.race_plans) == 3
        assert plan.total_risk >= 0
        assert isinstance(plan.settings, dict)

    def test_no_c_grade_bets(self):
        race_projs = {1: A_RACE, 2: C_RACE_TOSSED, 3: C_RACE_LOW_CONF}
        settings = BetSettings(bankroll=1000)
        plan = build_day_plan(race_projs, settings)
        for rp in plan.race_plans:
            if rp.grade == "C":
                assert rp.passed is True
                assert len(rp.tickets) == 0

    def test_day_risk_cap(self):
        settings = BetSettings(bankroll=1000, max_risk_per_day_pct=6.0)
        # max day = $60
        race_projs = {i: A_RACE for i in range(1, 11)}  # 10 A-races
        plan = build_day_plan(race_projs, settings)
        assert plan.total_risk <= settings.max_risk_per_day + 0.01

    def test_deterministic(self):
        race_projs = {1: A_RACE, 2: B_RACE}
        settings = BetSettings(bankroll=1000)
        p1 = build_day_plan(race_projs, settings)
        p2 = build_day_plan(race_projs, settings)
        assert p1.total_risk == p2.total_risk
        assert len(p1.race_plans) == len(p2.race_plans)

    def test_all_pass_warning(self):
        race_projs = {1: C_RACE_TOSSED, 2: C_RACE_LOW_CONF}
        settings = BetSettings()
        plan = build_day_plan(race_projs, settings)
        assert any("No bets" in w for w in plan.warnings)


class TestSerialization:
    def test_to_dict_roundtrip(self):
        race_projs = {1: A_RACE, 2: B_RACE}
        settings = BetSettings(bankroll=500)
        plan = build_day_plan(race_projs, settings)
        d = day_plan_to_dict(plan)
        assert "total_risk" in d
        assert "race_plans" in d
        assert len(d["race_plans"]) == 2

    def test_to_text(self):
        race_projs = {1: A_RACE}
        settings = BetSettings(bankroll=500, paper_mode=True)
        plan = build_day_plan(race_projs, settings)
        txt = day_plan_to_text(plan)
        assert "PAPER MODE" in txt
        assert "Bankroll: $500" in txt

    def test_to_csv(self):
        race_projs = {1: A_RACE, 2: C_RACE_TOSSED}
        settings = BetSettings(bankroll=1000)
        plan = build_day_plan(race_projs, settings)
        csv = day_plan_to_csv(plan)
        lines = csv.strip().split("\n")
        assert lines[0].startswith("race,grade")
        assert len(lines) >= 2  # header + at least one row


class TestRiskMultiplier:
    def test_conservative(self):
        assert _risk_multiplier("conservative") == 0.6

    def test_standard(self):
        assert _risk_multiplier("standard") == 1.0

    def test_aggressive(self):
        assert _risk_multiplier("aggressive") == 1.4

    def test_unknown(self):
        assert _risk_multiplier("unknown") == 1.0


class TestEstimateWinProb:
    def test_within_bounds(self):
        p = _estimate_win_prob(0.80, 10.0, 8)
        assert 0.05 <= p <= 0.60

    def test_small_field_boost(self):
        p_small = _estimate_win_prob(0.80, 10.0, 4)
        p_large = _estimate_win_prob(0.80, 10.0, 12)
        assert p_small > p_large
