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
    figure_quality_threshold: float = 0.80   # block race if < 80% figures present
    min_overlay: float = 1.10               # require 10% overlay min for daily wins

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
    figure_quality_pct: Optional[float] = None  # fraction of scoreable horses with figures


@dataclass
class DayPlan:
    """Complete bet plan for a card."""
    race_plans: List[RacePlan] = field(default_factory=list)
    total_risk: float = 0.0
    warnings: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DailyWinCandidate:
    """A single cross-track WIN candidate for the daily best bets page."""
    track: str
    session_id: str
    race_number: int
    horse_name: str
    post: str
    grade: str
    grade_reasons: List[str]
    confidence: float
    bias_score: float
    projection_type: str
    projected_low: float
    projected_high: float
    odds_decimal: Optional[float]
    odds_raw: str
    edge: float
    win_prob: float
    fair_odds: Optional[float]
    implied_prob: Optional[float]
    overlay: Optional[float]
    kelly_fraction: float
    stake: float
    best_bet_score: float
    tags: List[str]
    new_top_setup: bool
    bounce_risk: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_multiplier(profile: str) -> float:
    """Scale factor for stake sizing by risk profile."""
    return {"conservative": 0.6, "standard": 1.0, "aggressive": 1.4}.get(profile, 1.0)


def _fmt_odds(proj: Dict[str, Any]) -> str:
    """Format odds for ticket text: prefer raw ('9/5'), fallback decimal ('4.0-1')."""
    raw = proj.get("odds_raw") or ""
    dec = proj.get("odds")
    if raw:
        return raw
    if dec is not None:
        return f"{dec:.1f}-1"
    return ""


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

def grade_race(
    projections: List[Dict[str, Any]], settings: BetSettings,
    figure_quality_pct: Optional[float] = None,
) -> tuple:
    """Grade a race A/B/C based on engine projections.

    Returns (grade: str, reasons: List[str]).

    A = high confidence top pick AND (paired/rebound OR large gap) AND no major risk
    B = mid confidence or small gap or one uncertainty
    C = low confidence / chaos / tossed top pick → PASS
    """
    reasons = []

    if not projections:
        return ("C", ["no projections available"])

    # Quality guardrail: auto-PASS if figures too sparse
    if figure_quality_pct is not None and figure_quality_pct < settings.figure_quality_threshold:
        return ("C", [
            f"figure quality {figure_quality_pct:.0%} below threshold "
            f"{settings.figure_quality_threshold:.0%}"
        ])

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

    odds_str = _fmt_odds(top)
    return Ticket(
        bet_type="WIN",
        selections=[name],
        cost=stake,
        rationale=(
            f"WIN {name} @ {odds_str} — "
            f"Kelly={kf:.3f}, stake=${stake:.0f}"
        ),
        details={
            **_ids,
            "odds": odds,
            "odds_raw": top.get("odds_raw", ""),
            "kelly_fraction": round(kf, 4),
            "win_prob_est": round(win_prob, 3),
            "fair_odds": round((1.0 / win_prob) - 1.0, 2) if win_prob > 0 else None,
            "overlay": round(odds / ((1.0 / win_prob) - 1.0), 3) if win_prob > 0 and (1.0 / win_prob - 1.0) > 0 else None,
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
    # Build identifier lookup for each horse (includes odds for display)
    horse_ids = {
        _horse_name(p): {
            "post": p.get("post", ""),
            "horse_name": _horse_name(p),
            "normalized_name": _normalize_name(_horse_name(p)),
            "odds_raw": p.get("odds_raw", ""),
            "odds_decimal": p.get("odds"),
        }
        for p in ranked
    }

    # Key: 1 over 2,3,4
    key_count = min(len(ranked) - 1, 3)
    under_names = names[1:1 + key_count]
    key_cost = key_count * _BET_BASE
    if key_cost <= exacta_budget:
        top_odds = _fmt_odds(ranked[0])
        top_label = f"{names[0]} @ {top_odds}" if top_odds else names[0]
        tickets.append(Ticket(
            bet_type="EXACTA",
            selections=[names[0]] + under_names,
            cost=key_cost,
            rationale=f"EX KEY {top_label} over {','.join(under_names)} — ${key_cost:.0f}",
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
                saver_labels = []
                for sn in saver_names:
                    sp = next((p for p in ranked if _horse_name(p) == sn), {})
                    os = _fmt_odds(sp)
                    saver_labels.append(f"{sn} @ {os}" if os else sn)
                tickets.append(Ticket(
                    bet_type="EXACTA",
                    selections=saver_names + [names[0]],
                    cost=saver_cost,
                    rationale=f"EX SAVER {','.join(saver_labels)} / {names[0]} — ${saver_cost:.0f}",
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
    figure_quality_pct: Optional[float] = None,
) -> RacePlan:
    """Build a complete bet plan for one race."""
    grade, reasons = grade_race(projections, settings, figure_quality_pct=figure_quality_pct)

    if grade == "C":
        return RacePlan(
            race_number=race_number,
            grade="C",
            grade_reasons=reasons,
            passed=True,
            rationale="PASS — " + "; ".join(reasons),
            figure_quality_pct=figure_quality_pct,
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
            figure_quality_pct=figure_quality_pct,
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
        figure_quality_pct=figure_quality_pct,
    )


# ---------------------------------------------------------------------------
# Day Plan Builder
# ---------------------------------------------------------------------------

def build_day_plan(
    race_projections: Dict[int, List[Dict[str, Any]]],
    settings: BetSettings,
    race_quality: Optional[Dict[int, float]] = None,
) -> DayPlan:
    """Build a full day's bet plan across all races.

    *race_projections*: {race_number: [projection_dicts...]}
    *race_quality*: optional {race_number: figure_quality_pct} for guardrails

    Returns DayPlan with per-race plans, total risk, and any warnings.
    """
    warnings = []
    race_plans = []
    remaining = settings.max_risk_per_day

    for race_num in sorted(race_projections.keys()):
        projs = race_projections[race_num]
        quality = (race_quality or {}).get(race_num)
        plan = build_race_plan(race_num, projs, settings, remaining, figure_quality_pct=quality)
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
                "figure_quality_pct": rp.figure_quality_pct,
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


# ---------------------------------------------------------------------------
# Daily WIN Bets (cross-track)
# ---------------------------------------------------------------------------

def build_daily_wins(
    predictions_by_track: Dict[str, List[Dict[str, Any]]],
    odds_by_key: Dict[tuple, dict],
    settings: BetSettings,
    max_bets: int = 15,
    race_quality: Optional[Dict[tuple, float]] = None,
) -> List[DailyWinCandidate]:
    """Build cross-track daily WIN bet list from saved predictions.

    Args:
        predictions_by_track: {track: [pred_dicts]} from persistence
        odds_by_key: {(track, race_num, norm_name): odds_dict}
        settings: BetSettings with bankroll, risk caps, etc.
        max_bets: maximum number of bets to return (5-15)
        race_quality: optional {(track, race_num): quality_pct}

    Returns list of DailyWinCandidate sorted by edge descending.
    """
    candidates: List[DailyWinCandidate] = []
    multiplier = _risk_multiplier(settings.risk_profile)

    for track, preds in predictions_by_track.items():
        # Group by race_number
        by_race: Dict[int, List[Dict[str, Any]]] = {}
        for p in preds:
            rn = p.get("race_number", 0)
            by_race.setdefault(rn, []).append(p)

        for rn, projs in by_race.items():
            # Inject odds into each prediction
            for p in projs:
                if p.get("odds") is None:
                    key = (track, rn, p.get("normalized_name", ""))
                    snap = odds_by_key.get(key)
                    if snap and snap.get("odds_decimal") is not None:
                        p["odds"] = snap["odds_decimal"]
                        p["odds_raw"] = snap.get("odds_raw", "")

            # Grade the race
            quality = (race_quality or {}).get((track, rn))
            grade, reasons = grade_race(projs, settings, figure_quality_pct=quality)
            if grade != "A":
                continue

            # Take the #1 ranked horse
            ranked = sorted(projs, key=lambda p: p.get("bias_score", 0), reverse=True)
            top = ranked[0]
            if top.get("tossed", False):
                continue

            name = _horse_name(top)
            conf = top.get("confidence", 0.5)
            bias = top.get("bias_score", 0)
            odds = top.get("odds")
            odds_raw = top.get("odds_raw", "")
            field_size = len(ranked)

            # Compute edge + overlay
            win_prob = _estimate_win_prob(conf, bias, field_size)
            fair_odds_val = (1.0 / win_prob) - 1.0 if win_prob > 0 else None
            if odds is not None and odds > 0:
                implied_prob = 1.0 / (odds + 1.0)
                edge = win_prob - implied_prob
                if edge <= 0:
                    continue
                overlay = odds / fair_odds_val if fair_odds_val and fair_odds_val > 0 else None
                if overlay is not None and overlay < settings.min_overlay:
                    continue
                kf = kelly_fraction(odds, win_prob)
                if kf <= 0:
                    continue
                raw_stake = settings.bankroll * kf * multiplier
                stake = round(raw_stake / _BET_BASE) * _BET_BASE
                stake = max(stake, _BET_BASE)
                stake = min(stake, settings.max_risk_per_race)
            elif settings.allow_missing_odds:
                implied_prob = None
                overlay = None
                edge = win_prob * 0.5  # rough edge estimate for sort
                kf = 0.0
                stake = round(settings.max_risk_per_race * 0.5 / _BET_BASE) * _BET_BASE
                stake = max(stake, _BET_BASE)
            else:
                continue

            proj_low = top.get("projected_low", 0)
            proj_high = top.get("projected_high", 0)
            spread = proj_high - proj_low
            best_bet_score = bias + (conf * 10) - spread

            candidates.append(DailyWinCandidate(
                track=track,
                session_id=top.get("session_id", ""),
                race_number=rn,
                horse_name=name,
                post=str(top.get("post", "")),
                grade=grade,
                grade_reasons=reasons,
                confidence=conf,
                bias_score=bias,
                projection_type=top.get("projection_type", "NEUTRAL"),
                projected_low=proj_low,
                projected_high=proj_high,
                odds_decimal=odds,
                odds_raw=odds_raw,
                edge=edge,
                win_prob=win_prob,
                fair_odds=fair_odds_val,
                implied_prob=implied_prob,
                overlay=overlay,
                kelly_fraction=kf,
                stake=stake,
                best_bet_score=best_bet_score,
                tags=top.get("tags", []) if isinstance(top.get("tags"), list) else [],
                new_top_setup=bool(top.get("new_top_setup", False)),
                bounce_risk=bool(top.get("bounce_risk", False)),
            ))

    # Sort by overlay descending, then edge, then best_bet_score
    candidates.sort(key=lambda c: (c.overlay or 0, c.edge, c.best_bet_score), reverse=True)

    # Truncate to max_bets, enforce daily risk cap
    selected: List[DailyWinCandidate] = []
    total_risk = 0.0
    for c in candidates:
        if len(selected) >= max_bets:
            break
        if total_risk + c.stake > settings.max_risk_per_day:
            # Try reducing stake to fit
            remaining = settings.max_risk_per_day - total_risk
            if remaining >= _BET_BASE:
                c.stake = round(remaining / _BET_BASE) * _BET_BASE
            else:
                continue
        selected.append(c)
        total_risk += c.stake

    return selected
