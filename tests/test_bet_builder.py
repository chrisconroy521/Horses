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
    MultiRaceStrategy, build_multi_race_plan,
    multi_race_plan_to_dict, multi_race_plan_to_text, multi_race_plan_to_csv,
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
        settings = BetSettings(bankroll=1000, allow_missing_odds=True)
        ticket = build_win_ticket(no_odds, "A", settings, 15.0)
        assert ticket is not None
        assert ticket.details["method"] == "flat"

    def test_no_odds_blocked_by_default(self):
        no_odds = [_make_proj("NOPRICES", 10.0, 0.80, "PAIRED", odds=None)]
        settings = BetSettings(bankroll=1000)  # allow_missing_odds=False
        ticket = build_win_ticket(no_odds, "A", settings, 15.0)
        assert ticket is None

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


class TestBlockersDiagnostics:
    def test_missing_odds_blocker(self):
        """No-odds race reports blocker when allow_missing_odds=False."""
        # B-grade so no EXACTA fallback either
        no_odds_race = [
            _make_proj("NOPRICES", 9.0, 0.60, "IMPROVING", odds=None),
            _make_proj("SECOND", 7.5, 0.50, "NEUTRAL", odds=None),
        ]
        settings = BetSettings(bankroll=1000, allow_missing_odds=False)
        plan = build_race_plan(1, no_odds_race, settings, 15.0)
        assert plan.passed is True
        assert any("no brisnet ml odds" in b.lower() for b in plan.blockers)

    def test_odds_too_low_blocker(self):
        """Odds below min reports blocker."""
        low_odds_race = [
            _make_proj("CHEAP", 10.0, 0.80, "PAIRED", odds=1.5),
            _make_proj("SECOND", 7.0, 0.50, "NEUTRAL", odds=3.0),
        ]
        settings = BetSettings(bankroll=1000, min_odds_a=3.0)
        plan = build_race_plan(1, low_odds_race, settings, 15.0)
        assert any("below min" in b for b in plan.blockers)

    def test_diagnostics_in_day_plan_dict(self):
        """day_plan_to_dict includes diagnostics with grade_counts and blockers."""
        race_projs = {1: A_RACE, 2: C_RACE_TOSSED}
        settings = BetSettings(bankroll=1000)
        plan = build_day_plan(race_projs, settings)
        d = day_plan_to_dict(plan)
        assert "diagnostics" in d
        assert "grade_counts" in d["diagnostics"]
        assert "blockers" in d["diagnostics"]
        assert d["diagnostics"]["grade_counts"]["C"] >= 1

    def test_blockers_in_race_plan_dict(self):
        """Race plan dict includes blockers list."""
        race_projs = {1: C_RACE_LOW_CONF}
        settings = BetSettings(bankroll=1000)
        plan = build_day_plan(race_projs, settings)
        d = day_plan_to_dict(plan)
        rp = d["race_plans"][0]
        assert "blockers" in rp


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


# ---------------------------------------------------------------------------
# Ticket → Result linking tests (require persistence)
# ---------------------------------------------------------------------------

import json
import tempfile
from persistence import Persistence


def _setup_db_with_results():
    """Create a temp DB with result_entries for GP 02/26/2026 race 1."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db = Persistence(Path(tmp.name))
    db.insert_race_result(
        track="GP", race_date="02/26/2026", race_number=1,
        surface="Dirt", distance="6f",
    )
    # Post 8 = winner (ALPHA HORSE), post 4 = 2nd (BETA HORSE)
    db.insert_entry_result(
        track="GP", race_date="02/26/2026", race_number=1,
        post=8, horse_name="Alpha Horse", finish_pos=1,
        odds=5.0, win_payoff=12.00,
    )
    db.insert_entry_result(
        track="GP", race_date="02/26/2026", race_number=1,
        post=4, horse_name="Beta Horse", finish_pos=2,
        odds=8.0,
    )
    db.insert_entry_result(
        track="GP", race_date="02/26/2026", race_number=1,
        post=5, horse_name="Gamma Horse", finish_pos=3,
        odds=12.0,
    )
    return db


class TestTicketResultLinking:
    """Test that evaluate_bet_plan_roi resolves tickets against results."""

    def test_resolve_by_post(self):
        """HIGH tier: match ticket to result by (track, date, race_number, post)."""
        db = _setup_db_with_results()
        # Save a plan with a WIN ticket that has post=8
        plan_dict = {
            "race_plans": [{
                "race_number": 1,
                "grade": "A",
                "passed": False,
                "total_cost": 10.0,
                "tickets": [{
                    "bet_type": "WIN",
                    "selections": ["Alpha Horse"],
                    "cost": 10.0,
                    "rationale": "test",
                    "details": {
                        "post": 8,
                        "horse_name": "Alpha Horse",
                        "normalized_name": "ALPHA HORSE",
                    },
                }],
            }],
        }
        plan_id = db.save_bet_plan(
            session_id="test-sess", track="GP", race_date="02/26/2026",
            settings_dict={"bankroll": 1000}, plan_dict=plan_dict,
            total_risk=10.0, paper_mode=True,
        )
        result = db.evaluate_bet_plan_roi(plan_id)
        assert result["resolved"] == 1
        assert result["unresolved"] == 0
        tr = result["ticket_results"][0]
        assert tr["outcome"] == "won"
        assert tr["returned"] > 0  # $10/$2 * $12 = $60
        assert tr["match_tier"] == "high"

    def test_resolve_by_name_fallback(self):
        """LOW tier: match by normalized horse name when post is missing."""
        db = _setup_db_with_results()
        # Save a plan where post is missing but horse_name matches
        plan_dict = {
            "race_plans": [{
                "race_number": 1,
                "grade": "B",
                "passed": False,
                "total_cost": 6.0,
                "tickets": [{
                    "bet_type": "WIN",
                    "selections": ["Beta Horse"],
                    "cost": 6.0,
                    "rationale": "test",
                    "details": {
                        "post": None,
                        "horse_name": "Beta Horse",
                        "normalized_name": "",
                    },
                }],
            }],
        }
        plan_id = db.save_bet_plan(
            session_id="test-sess", track="GP", race_date="02/26/2026",
            settings_dict={"bankroll": 1000}, plan_dict=plan_dict,
            total_risk=6.0, paper_mode=True,
        )
        result = db.evaluate_bet_plan_roi(plan_id)
        assert result["resolved"] == 1
        tr = result["ticket_results"][0]
        assert tr["outcome"] == "lost"  # Beta Horse finished 2nd
        assert tr["match_tier"] == "low"  # fell through to selection name


class TestDashboardSharedPipeline:
    """Verify shared helpers produce identical output (deterministic pipeline)."""

    def test_grade_race_deterministic(self):
        """grade_race is a pure function — same input = same output."""
        settings = BetSettings()
        g1, r1 = grade_race(A_RACE, settings)
        g2, r2 = grade_race(A_RACE, settings)
        assert g1 == g2
        assert r1 == r2

    def test_build_day_plan_deterministic(self):
        """build_day_plan is deterministic — same projections = same plan."""
        projs = {1: A_RACE, 2: B_RACE}
        settings = BetSettings(bankroll=1000)
        p1 = day_plan_to_dict(build_day_plan(projs, settings))
        p2 = day_plan_to_dict(build_day_plan(projs, settings))
        assert p1 == p2


class TestMultiRacePlan:
    """Tests for Pick 3 / Pick 6 multi-race ticket builder."""

    def test_pick3_combinations(self):
        """Pick 3 with A(1) x B(3) x A(1) = 3 combos, cost $6."""
        race_projs = {1: A_RACE, 2: B_RACE, 3: A_RACE}
        strategy = MultiRaceStrategy(a_count=1, b_count=3, c_count=5)
        plan = build_multi_race_plan(
            race_projs, start_race=1, bet_type="PICK3",
            budget=100, strategy=strategy, settings=BetSettings(),
        )
        assert len(plan.legs) == 3
        assert plan.legs[0].grade == "A"
        assert plan.legs[0].horse_count == 1
        assert plan.legs[1].grade == "B"
        assert plan.legs[1].horse_count == 3
        assert plan.combinations == 1 * 3 * 1
        assert plan.cost == 3 * 2.0  # $6
        assert not plan.over_budget

    def test_pick6_combinations(self):
        """Pick 6 correct combo math — 6 A-legs with single horses."""
        race_projs = {i: A_RACE for i in range(1, 7)}
        strategy = MultiRaceStrategy(a_count=1, b_count=3, c_count=5)
        plan = build_multi_race_plan(
            race_projs, start_race=1, bet_type="PICK6",
            budget=100, strategy=strategy, settings=BetSettings(),
        )
        assert len(plan.legs) == 6
        assert plan.combinations == 1  # all A-legs with 1 horse each
        assert plan.cost == 2.0

    def test_max_one_c_leg(self):
        """More than 1 C-leg triggers warning unless overridden."""
        race_projs = {1: C_RACE_LOW_CONF, 2: C_RACE_LOW_CONF, 3: A_RACE}
        strategy = MultiRaceStrategy(a_count=1, b_count=3, c_count=5)
        plan = build_multi_race_plan(
            race_projs, start_race=1, bet_type="PICK3",
            budget=500, strategy=strategy, settings=BetSettings(),
        )
        assert plan.c_leg_count == 2
        assert plan.c_leg_warning is True

    def test_c_leg_override(self):
        """Override suppresses c_leg_warning."""
        race_projs = {1: C_RACE_LOW_CONF, 2: C_RACE_LOW_CONF, 3: A_RACE}
        strategy = MultiRaceStrategy(a_count=1, b_count=3, c_count=5)
        plan = build_multi_race_plan(
            race_projs, start_race=1, bet_type="PICK3",
            budget=500, strategy=strategy, settings=BetSettings(),
            c_leg_override=True,
        )
        assert plan.c_leg_warning is False

    def test_budget_cap_enforced(self):
        """Over-budget flagged."""
        race_projs = {1: A_RACE, 2: B_RACE, 3: B_RACE}
        strategy = MultiRaceStrategy(a_count=1, b_count=3, c_count=5)
        plan = build_multi_race_plan(
            race_projs, start_race=1, bet_type="PICK3",
            budget=5, strategy=strategy, settings=BetSettings(),
        )
        # 1 x 3 x 3 = 9 combos = $18 > $5
        assert plan.over_budget is True

    def test_figure_quality_blocks_leg(self):
        """Low figure quality on a leg adds warning."""
        race_projs = {1: A_RACE, 2: A_RACE, 3: A_RACE}
        quality = {1: 0.50}  # 50% missing → 50% quality < 80% threshold
        strategy = MultiRaceStrategy(a_count=1, b_count=3, c_count=5)
        plan = build_multi_race_plan(
            race_projs, start_race=1, bet_type="PICK3",
            budget=100, strategy=strategy, settings=BetSettings(),
            race_quality=quality,
        )
        assert any("quality" in w.lower() for w in plan.warnings)

    def test_serialization(self):
        """Multi-race plan serializes to dict, text, and csv."""
        race_projs = {1: A_RACE, 2: B_RACE, 3: A_RACE}
        strategy = MultiRaceStrategy(a_count=1, b_count=3, c_count=5)
        plan = build_multi_race_plan(
            race_projs, start_race=1, bet_type="PICK3",
            budget=100, strategy=strategy, settings=BetSettings(),
        )
        d = multi_race_plan_to_dict(plan)
        assert d["bet_type"] == "PICK3"
        assert len(d["legs"]) == 3

        txt = multi_race_plan_to_text(plan)
        assert "PICK3" in txt
        assert "Leg 1" in txt

        csv = multi_race_plan_to_csv(plan)
        lines = csv.strip().split("\n")
        assert lines[0].startswith("leg,race")
