"""Bet Builder v1 — WIN + EXACTA ticket generation with bankroll guardrails.

Uses engine projections (confidence, projection_type, bias_score, bounce_risk)
to grade races A/B/C, then generates Kelly-sized WIN bets and structured
EXACTA tickets with per-race and per-day risk caps.

Usage:
    from bet_builder import build_day_plan, BetSettings
    settings = BetSettings(bankroll=1000)
    plan = build_day_plan(race_projections, settings)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default thresholds for race grading
_DEFAULT_MIN_CONFIDENCE_A = 0.65
_DEFAULT_MIN_CONFIDENCE_B = 0.45
_SCORE_GAP_LARGE = 3.0        # bias_score gap #1 vs #2 for "large separation"
_SCORE_GAP_MEDIUM = 1.5

# Kelly fraction cap (never bet more than this fraction of bankroll)
_MAX_KELLY_FRACTION = 0.05

# Exacta budget as % of race budget
_EXACTA_BUDGET_PCT = 0.40

# $2 base for all bets
_BET_BASE = 2.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BetSettings:
    """User-configurable betting parameters."""
    bankroll: float = 1000.0
    risk_profile: str = "standard"          # conservative / standard / aggressive
    max_risk_per_race_pct: float = 1.5      # % of bankroll
    max_risk_per_day_pct: float = 6.0       # % of bankroll
    min_confidence: float = 0.0             # 0 = use defaults per grade
    min_odds_a: float = 2.0                 # minimum odds for A-race WIN bet
    min_odds_b: float = 4.0                 # minimum odds for B-race WIN bet
    paper_mode: bool = True                 # paper bets (no real money implied)
    allow_missing_odds: bool = False         # allow flat-stake WIN when odds absent

    @property
    def max_risk_per_race(self) -> float:
        return self.bankroll * self.max_risk_per_race_pct / 100.0

    @property
    def max_risk_per_day(self) -> float:
        return self.bankroll * self.max_risk_per_day_pct / 100.0


@dataclass
class Ticket:
    """A single bet ticket."""
    bet_type: str              # "WIN" or "EXACTA"
    selections: List[str]      # horse names involved
    cost: float                # total ticket cost in $
    rationale: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RacePlan:
    """Bet plan for one race."""
    race_number: int = 0
    grade: str = "C"           # A / B / C
    grade_reasons: List[str] = field(default_factory=list)
    tickets: List[Ticket] = field(default_factory=list)
    total_cost: float = 0.0
    rationale: str = ""
    passed: bool = False       # True if grade=C → no bets
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)


@dataclass
class DayPlan:
    """Complete bet plan for a card."""
    race_plans: List[RacePlan] = field(default_factory=list)
    total_risk: float = 0.0
    warnings: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_multiplier(profile: str) -> float:
    """Scale factor for stake sizing by risk profile."""
    return {"conservative": 0.6, "standard": 1.0, "aggressive": 1.4}.get(profile, 1.0)


def _horse_name(p: Dict[str, Any]) -> str:
    """Extract horse name from a projection dict.

    Predictions from persistence use 'horse_name'; engine dicts use 'name'.
    """
    return p.get("horse_name") or p.get("name") or "?"


def _normalize_name(name: str) -> str:
    """Uppercase, strip punctuation, collapse whitespace (same logic as Persistence)."""
    import re as _re
    n = (name or "").upper().strip()
    n = _re.sub(r"[\u2018\u2019\u201C\u201D'\-.,()\"']", "", n)
    n = _re.sub(r"\s+", " ", n)
    return n


def kelly_fraction(odds: float, win_prob: float) -> float:
    """Fractional Kelly criterion for a $2 win bet.

    Returns the fraction of bankroll to wager (0.0 if negative edge).
    Capped at _MAX_KELLY_FRACTION.

    *odds* is decimal-style (e.g. 3.0 means 3-1).
    *win_prob* is 0.0–1.0.
    """
    if odds <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.0
    # Kelly: f = (b*p - q) / b  where b = decimal payout, p = prob, q = 1-p
    b = odds  # profit per $1 wagered
    q = 1.0 - win_prob
    f = (b * win_prob - q) / b
    if f <= 0:
        return 0.0
    return min(f, _MAX_KELLY_FRACTION)


def _estimate_win_prob(confidence: float, bias_score: float, field_size: int) -> float:
    """Rough win probability from engine confidence + bias score rank.

    This is a heuristic, not a model — uses confidence as base probability
    scaled by relative score strength. Minimum floor of 5%.
    """
    # Base: confidence itself is a reasonable starting point
    base = confidence
    # Boost slightly if field is small
    if field_size <= 5:
        base *= 1.15
    elif field_size >= 10:
        base *= 0.85
    return max(min(base * 0.5, 0.60), 0.05)


# ---------------------------------------------------------------------------
# Race Grading
# ---------------------------------------------------------------------------

def grade_race(projections: List[Dict[str, Any]], settings: BetSettings) -> tuple:
    """Grade a race A/B/C based on engine projections.

    Returns (grade: str, reasons: List[str]).

    A = high confidence top pick AND (paired/rebound OR large gap) AND no major risk
    B = mid confidence or small gap or one uncertainty
    C = low confidence / chaos / tossed top pick → PASS
    """
    reasons = []

    if not projections:
        return ("C", ["no projections available"])

    # Sort by bias_score descending
    ranked = sorted(projections, key=lambda p: p.get("bias_score", 0), reverse=True)
    top = ranked[0]

    top_conf = top.get("confidence", 0)
    top_type = top.get("projection_type", "NEUTRAL")
    top_tossed = top.get("tossed", False)
    top_bounce = top.get("bounce_risk", False)
    top_name = _horse_name(top)

    # Effective min confidence
    min_conf_a = settings.min_confidence if settings.min_confidence > 0 else _DEFAULT_MIN_CONFIDENCE_A
    min_conf_b = settings.min_confidence if settings.min_confidence > 0 else _DEFAULT_MIN_CONFIDENCE_B

    # Score gap
    gap = 0.0
    if len(ranked) >= 2:
        gap = ranked[0].get("bias_score", 0) - ranked[1].get("bias_score", 0)

    strong_cycle = top_type in ("PAIRED", "REBOUND", "IMPROVING")
    large_gap = gap >= _SCORE_GAP_LARGE
    medium_gap = gap >= _SCORE_GAP_MEDIUM

    # --- C grade (PASS) checks ---
    if top_tossed:
        return ("C", [f"top pick {top_name} is TOSSED"])
    if top_conf < min_conf_b:
        return ("C", [f"top confidence {top_conf:.0%} below min {min_conf_b:.0%}"])

    # --- A grade checks ---
    is_a = (
        top_conf >= min_conf_a
        and (strong_cycle or large_gap)
        and not top_bounce
    )
    if is_a:
        reasons.append(f"top {top_name}: conf={top_conf:.0%}, cycle={top_type}, gap={gap:.1f}")
        if strong_cycle:
            reasons.append(f"strong cycle: {top_type}")
        if large_gap:
            reasons.append(f"large score gap: {gap:.1f}")
        return ("A", reasons)

    # --- B grade ---
    reasons.append(f"top {top_name}: conf={top_conf:.0%}, cycle={top_type}, gap={gap:.1f}")
    if top_bounce:
        reasons.append("bounce risk on top pick")
    if not strong_cycle:
        reasons.append(f"cycle {top_type} not ideal")
    if not medium_gap:
        reasons.append(f"small score gap: {gap:.1f}")
    return ("B", reasons)


# ---------------------------------------------------------------------------
# Ticket Generation
# ---------------------------------------------------------------------------

def build_win_ticket(
    ranked: List[Dict[str, Any]], grade: str,
    settings: BetSettings, race_budget: float,
) -> Optional[Ticket]:
    """Build a WIN ticket on the #1 pick if odds qualify."""
    if not ranked:
        return None

    top = ranked[0]
    name = _horse_name(top)
    norm = _normalize_name(name)
    post = top.get("post", "")
    odds = top.get("odds")  # may be None if not yet available

    # Common identifier block stored on every ticket
    _ids = {"post": post, "horse_name": name, "normalized_name": norm}

    min_odds = settings.min_odds_a if grade == "A" else settings.min_odds_b

    # If no odds available, use flat stake only if allowed
    if odds is None:
        if not settings.allow_missing_odds:
            return None
        stake = round(race_budget * 0.5 / _BET_BASE) * _BET_BASE
        stake = max(stake, _BET_BASE)
        stake = min(stake, race_budget)
        return Ticket(
            bet_type="WIN",
            selections=[name],
            cost=stake,
            rationale=f"WIN #{name} — no odds, flat {stake:.0f}",
            details={**_ids, "odds": None, "method": "flat"},
        )

    if odds < min_odds:
        return None  # odds too low

    # Kelly sizing
    conf = top.get("confidence", 0.5)
    field_size = len(ranked)
    win_prob = _estimate_win_prob(conf, top.get("bias_score", 0), field_size)
    kf = kelly_fraction(odds, win_prob)

    if kf <= 0:
        return None

    multiplier = _risk_multiplier(settings.risk_profile)
    raw_stake = settings.bankroll * kf * multiplier
    # Round to nearest $2
    stake = round(raw_stake / _BET_BASE) * _BET_BASE
    stake = max(stake, _BET_BASE)
    stake = min(stake, race_budget)

    return Ticket(
        bet_type="WIN",
        selections=[name],
        cost=stake,
        rationale=(
            f"WIN #{name} @ {odds:.1f}-1 — "
            f"Kelly={kf:.3f}, stake=${stake:.0f}"
        ),
        details={
            **_ids,
            "odds": odds,
            "kelly_fraction": round(kf, 4),
            "win_prob_est": round(win_prob, 3),
            "method": "kelly",
        },
    )


def build_exacta_tickets(
    ranked: List[Dict[str, Any]], grade: str, race_budget: float,
) -> List[Ticket]:
    """Build EXACTA tickets for A-grade races only.

    Key: #1 over (#2, #3, #4)
    Saver: (#2, #3) over #1 — only if duel risk (gap < SCORE_GAP_MEDIUM between #2/#3)
    """
    if grade != "A" or len(ranked) < 2:
        return []

    exacta_budget = race_budget * _EXACTA_BUDGET_PCT
    tickets = []

    names = [_horse_name(p) for p in ranked]
    # Build identifier lookup for each horse
    horse_ids = {
        _horse_name(p): {
            "post": p.get("post", ""),
            "horse_name": _horse_name(p),
            "normalized_name": _normalize_name(_horse_name(p)),
        }
        for p in ranked
    }

    # Key: 1 over 2,3,4
    key_count = min(len(ranked) - 1, 3)
    under_names = names[1:1 + key_count]
    key_cost = key_count * _BET_BASE
    if key_cost <= exacta_budget:
        tickets.append(Ticket(
            bet_type="EXACTA",
            selections=[names[0]] + under_names,
            cost=key_cost,
            rationale=f"EX KEY {names[0]} / {','.join(under_names)} — ${key_cost:.0f}",
            details={
                "structure": "key", "top": names[0], "unders": under_names,
                "horses": {n: horse_ids.get(n, {}) for n in [names[0]] + under_names},
            },
        ))
        exacta_budget -= key_cost

    # Saver: 2,3 over 1 if duel risk
    if len(ranked) >= 3 and exacta_budget >= _BET_BASE * 2:
        gap_23 = abs(ranked[1].get("bias_score", 0) - ranked[2].get("bias_score", 0))
        if gap_23 < _SCORE_GAP_MEDIUM:
            saver_names = names[1:3]
            saver_cost = len(saver_names) * _BET_BASE
            if saver_cost <= exacta_budget:
                tickets.append(Ticket(
                    bet_type="EXACTA",
                    selections=saver_names + [names[0]],
                    cost=saver_cost,
                    rationale=f"EX SAVER {','.join(saver_names)} / {names[0]} — ${saver_cost:.0f}",
                    details={
                        "structure": "saver", "overs": saver_names, "under": names[0],
                        "horses": {n: horse_ids.get(n, {}) for n in saver_names + [names[0]]},
                    },
                ))

    return tickets


# ---------------------------------------------------------------------------
# Race Plan Builder
# ---------------------------------------------------------------------------

def build_race_plan(
    race_number: int,
    projections: List[Dict[str, Any]],
    settings: BetSettings,
    remaining_day_budget: float,
) -> RacePlan:
    """Build a complete bet plan for one race."""
    grade, reasons = grade_race(projections, settings)

    if grade == "C":
        return RacePlan(
            race_number=race_number,
            grade="C",
            grade_reasons=reasons,
            passed=True,
            rationale="PASS — " + "; ".join(reasons),
        )

    race_budget = min(settings.max_risk_per_race, remaining_day_budget)
    if race_budget < _BET_BASE:
        return RacePlan(
            race_number=race_number,
            grade=grade,
            grade_reasons=reasons,
            passed=True,
            rationale="PASS — daily risk cap reached",
            warnings=["daily cap reached"],
        )

    ranked = sorted(projections, key=lambda p: p.get("bias_score", 0), reverse=True)
    tickets = []
    blockers = []

    # WIN ticket
    win_ticket = build_win_ticket(ranked, grade, settings, race_budget)
    if win_ticket:
        tickets.append(win_ticket)
        race_budget -= win_ticket.cost
    elif ranked:
        # Diagnose why no WIN ticket
        top = ranked[0]
        top_name = _horse_name(top)
        odds = top.get("odds")
        min_odds = settings.min_odds_a if grade == "A" else settings.min_odds_b
        if odds is None:
            if not settings.allow_missing_odds:
                blockers.append(f"No BRISNET ML odds found for {top_name} — enable 'Allow missing odds' for flat-stake bet")
            else:
                blockers.append(f"No odds for {top_name} — budget too small for flat-stake")
        elif odds < min_odds:
            blockers.append(f"{top_name} odds {odds:.1f} below min {min_odds:.1f} for grade {grade}")
        else:
            conf = top.get("confidence", 0.5)
            win_prob = _estimate_win_prob(conf, top.get("bias_score", 0), len(ranked))
            blockers.append(f"Negative edge for {top_name} at {odds:.1f}-1 (est win prob {win_prob:.0%}) — Kelly says no bet")

    # EXACTA tickets (A only)
    if grade == "A" and race_budget >= _BET_BASE:
        ex_tickets = build_exacta_tickets(ranked, grade, race_budget)
        for t in ex_tickets:
            if t.cost <= race_budget:
                tickets.append(t)
                race_budget -= t.cost
    elif grade != "A" and len(ranked) >= 2:
        blockers.append(f"Exactas only for A-grade races (this is grade {grade})")

    total_cost = sum(t.cost for t in tickets)
    rationale_parts = [f"Grade {grade}"]
    for t in tickets:
        rationale_parts.append(t.rationale)

    return RacePlan(
        race_number=race_number,
        grade=grade,
        grade_reasons=reasons,
        tickets=tickets,
        total_cost=total_cost,
        rationale=" | ".join(rationale_parts),
        passed=len(tickets) == 0,
        blockers=blockers,
    )


# ---------------------------------------------------------------------------
# Day Plan Builder
# ---------------------------------------------------------------------------

def build_day_plan(
    race_projections: Dict[int, List[Dict[str, Any]]],
    settings: BetSettings,
) -> DayPlan:
    """Build a full day's bet plan across all races.

    *race_projections*: {race_number: [projection_dicts...]}

    Returns DayPlan with per-race plans, total risk, and any warnings.
    """
    warnings = []
    race_plans = []
    remaining = settings.max_risk_per_day

    for race_num in sorted(race_projections.keys()):
        projs = race_projections[race_num]
        plan = build_race_plan(race_num, projs, settings, remaining)
        race_plans.append(plan)
        remaining -= plan.total_cost

    total_risk = sum(rp.total_cost for rp in race_plans)

    if total_risk >= settings.max_risk_per_day * 0.95:
        warnings.append(
            f"Daily risk cap nearly reached: ${total_risk:.0f} / ${settings.max_risk_per_day:.0f}"
        )

    bet_count = sum(len(rp.tickets) for rp in race_plans)
    pass_count = sum(1 for rp in race_plans if rp.passed)

    if bet_count == 0:
        warnings.append("No bets generated — all races graded C or below thresholds")

    return DayPlan(
        race_plans=race_plans,
        total_risk=total_risk,
        warnings=warnings,
        settings=asdict(settings),
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def day_plan_to_dict(plan: DayPlan) -> dict:
    """Convert DayPlan to JSON-serializable dict."""
    # Build diagnostics summary
    grade_counts = {"A": 0, "B": 0, "C": 0}
    all_blockers = []
    for rp in plan.race_plans:
        grade_counts[rp.grade] = grade_counts.get(rp.grade, 0) + 1
        for b in rp.blockers:
            all_blockers.append({"race": rp.race_number, "grade": rp.grade, "reason": b})

    return {
        "total_risk": plan.total_risk,
        "warnings": plan.warnings,
        "settings": plan.settings,
        "diagnostics": {
            "grade_counts": grade_counts,
            "total_tickets": sum(len(rp.tickets) for rp in plan.race_plans),
            "total_passed": sum(1 for rp in plan.race_plans if rp.passed),
            "blockers": all_blockers,
        },
        "race_plans": [
            {
                "race_number": rp.race_number,
                "grade": rp.grade,
                "grade_reasons": rp.grade_reasons,
                "passed": rp.passed,
                "total_cost": rp.total_cost,
                "rationale": rp.rationale,
                "blockers": rp.blockers,
                "tickets": [
                    {
                        "bet_type": t.bet_type,
                        "selections": t.selections,
                        "cost": t.cost,
                        "rationale": t.rationale,
                        "details": t.details,
                    }
                    for t in rp.tickets
                ],
            }
            for rp in plan.race_plans
        ],
    }


def day_plan_to_text(plan: DayPlan) -> str:
    """Format DayPlan as human-readable text for export."""
    lines = []
    mode = "PAPER MODE" if plan.settings.get("paper_mode", True) else "LIVE"
    lines.append(f"=== BET PLAN ({mode}) ===")
    lines.append(f"Bankroll: ${plan.settings.get('bankroll', 0):.0f}")
    lines.append(f"Risk profile: {plan.settings.get('risk_profile', 'standard')}")
    lines.append(f"Total risk: ${plan.total_risk:.0f}")
    lines.append("")

    for rp in plan.race_plans:
        if rp.passed:
            lines.append(f"Race {rp.race_number}: PASS (Grade {rp.grade}) — {rp.rationale}")
            for b in rp.blockers:
                lines.append(f"  [blocker] {b}")
        else:
            lines.append(f"Race {rp.race_number}: Grade {rp.grade} — ${rp.total_cost:.0f}")
            for t in rp.tickets:
                lines.append(f"  {t.rationale}")
        lines.append("")

    if plan.warnings:
        lines.append("WARNINGS:")
        for w in plan.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def day_plan_to_csv(plan: DayPlan) -> str:
    """Export tickets as CSV."""
    rows = ["race,grade,bet_type,selections,cost,rationale"]
    for rp in plan.race_plans:
        if rp.passed:
            rows.append(f"{rp.race_number},{rp.grade},PASS,,0,{rp.rationale}")
        else:
            for t in rp.tickets:
                sels = " / ".join(t.selections)
                rows.append(
                    f"{rp.race_number},{rp.grade},{t.bet_type},"
                    f"\"{sels}\",{t.cost:.2f},\"{t.rationale}\""
                )
    return "\n".join(rows)
