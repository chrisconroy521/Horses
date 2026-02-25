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


def projection_to_dict(proj, odds: float = None, odds_raw: str = "") -> Dict[str, Any]:
    """Convert a HorseProjection dataclass to dict for bet builders.

    Handles both HorseProjection objects (from engine_outputs_by_session)
    and dicts (already converted or from API). This is the bridge between
    canonical session_state storage and bet builder functions.
    """
    if isinstance(proj, dict):
        return proj
    return {
        "horse_name": proj.name, "name": proj.name, "post": proj.post,
        "style": proj.style, "projection_type": proj.projection_type,
        "confidence": proj.confidence, "bias_score": proj.bias_score,
        "raw_score": proj.raw_score, "projected_low": proj.projected_low,
        "projected_high": proj.projected_high, "proj_mid": proj.proj_mid,
        "spread": proj.spread, "cycle_priority": proj.cycle_priority,
        "sheets_rank": proj.sheets_rank, "bounce_risk": proj.bounce_risk,
        "tossed": proj.tossed,
        "toss_reasons": getattr(proj, "toss_reasons", []),
        "tags": proj.tags, "new_top_setup": proj.new_top_setup,
        "tie_break_used": proj.tie_break_used,
        "dev_pattern": getattr(proj, "dev_pattern", ""),
        "odds": odds, "odds_raw": odds_raw,
    }


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
# Model Probability (Harville-style) + Overlay
# ---------------------------------------------------------------------------

_HARVILLE_EXP = 0.81


def _model_prob(sheets_rank: int, confidence: float, spread: float,
                field_size: int) -> float:
    """Harville-style win probability from sheets-first rank + confidence + spread.

    1. Base = rank weighting: 1/rank^0.81 normalised across field
    2. Confidence multiplier: 0.5 + confidence  (0.75 – 1.40)
    3. Spread penalty: 1/(1 + spread/20)  (tighter = better)
    Clamped to [0.02, 0.65].
    """
    rank = max(1, sheets_rank)
    fs = max(1, field_size)
    base = 1.0 / (rank ** _HARVILLE_EXP)
    denom = sum(1.0 / (r ** _HARVILLE_EXP) for r in range(1, fs + 1))
    harville = base / denom if denom > 0 else 1.0 / fs
    conf_mult = 0.5 + confidence
    spread_pen = 1.0 / (1.0 + max(spread, 0.0) / 20.0)
    return max(0.02, min(harville * conf_mult * spread_pen, 0.65))


def _compute_race_probs(
    projections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute normalised model_prob for every horse in a race.

    Returns list of dicts (one per horse, same order) with keys:
        horse_name, post, model_prob, implied_prob, overlay, odds
    """
    field_size = len(projections)
    if field_size == 0:
        return []

    raw: List[tuple] = []
    for p in projections:
        mp = _model_prob(
            p.get("sheets_rank", field_size),
            p.get("confidence", 0.5),
            p.get("spread", 3.0),
            field_size,
        )
        raw.append((p, mp))

    total = sum(mp for _, mp in raw)
    scale = 1.0 / total if total > 0 else 1.0

    results: List[Dict[str, Any]] = []
    for proj, mp in raw:
        normed = max(0.02, min(mp * scale, 0.65))
        odds = proj.get("odds")
        implied = 1.0 / (odds + 1.0) if odds and odds > 0 else None
        overlay = normed / implied if implied and implied > 0 else None
        results.append({
            "horse_name": _horse_name(proj),
            "post": str(proj.get("post", "")),
            "model_prob": round(normed, 4),
            "implied_prob": round(implied, 4) if implied is not None else None,
            "overlay": round(overlay, 3) if overlay is not None else None,
            "odds": odds,
        })
    return results


def is_true_single(proj: Dict[str, Any]) -> bool:
    """TRUE SINGLE: high-cycle, high-confidence, no bounce risk."""
    return (
        proj.get("cycle_priority", 0) >= 5
        and proj.get("confidence", 0) >= 0.65
        and not proj.get("bounce_risk", False)
    )


def _is_score_trigger(
    proj: Dict[str, Any],
    overlay: Optional[float],
    score_min_odds: float = 8.0,
    score_min_overlay: float = 1.60,
) -> bool:
    """Score-mode gate: longshot + big overlay + no bounce risk."""
    odds = proj.get("odds")
    if odds is None or odds < score_min_odds:
        return False
    if overlay is None or overlay < score_min_overlay:
        return False
    if proj.get("bounce_risk", False):
        return False
    return True


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


def build_trifecta_tickets(
    ranked: List[Dict[str, Any]],
    grade: str,
    race_budget: float,
    overrides: Optional["CountOverrides"] = None,
) -> List[Ticket]:
    """Build TRIFECTA tickets for A or B grade races.

    Key tri: A1 WITH (A2,B1) WITH (A2,B1,B2,B3)
    Cost = first × second × third × $2
    """
    if grade not in ("A", "B") or len(ranked) < 4:
        return []

    if overrides:
        ranked = apply_overrides(ranked, overrides)

    non_tossed = [p for p in ranked if not p.get("tossed", False)]
    if len(non_tossed) < 4:
        non_tossed = ranked[:4]

    names = [_horse_name(p) for p in non_tossed]
    horse_ids = {
        _horse_name(p): {
            "post": p.get("post", ""),
            "horse_name": _horse_name(p),
            "normalized_name": _normalize_name(_horse_name(p)),
            "odds_raw": p.get("odds_raw", ""),
            "odds_decimal": p.get("odds"),
        }
        for p in non_tossed
    }

    tri_budget = race_budget * 0.40  # 40% of race budget for trifecta

    # Determine counts
    if overrides:
        a_ct = min(overrides.a_count, len(non_tossed))
        b_ct = min(overrides.b_count, max(len(non_tossed) - a_ct, 0))
    else:
        a_ct = 1
        b_ct = min(2, len(non_tossed) - 1)

    first_pos = names[:a_ct]
    second_pos = names[:a_ct + b_ct]
    third_ct = min(a_ct + b_ct + 2, len(non_tossed))
    third_pos = names[:third_ct]

    # Remove duplicate combos: first != second != third
    combo_count = len(first_pos) * (len(second_pos) - 1) * max(len(third_pos) - 2, 1)
    cost = combo_count * _BET_BASE
    if cost > tri_budget and combo_count > 0:
        # Scale down third_pos
        while cost > tri_budget and len(third_pos) > len(second_pos):
            third_pos = third_pos[:-1]
            combo_count = len(first_pos) * (len(second_pos) - 1) * max(len(third_pos) - 2, 1)
            cost = combo_count * _BET_BASE

    if combo_count <= 0:
        return []

    cost = combo_count * _BET_BASE

    tickets = []
    top_odds = _fmt_odds(non_tossed[0])
    top_label = f"{names[0]} @ {top_odds}" if top_odds else names[0]
    tickets.append(Ticket(
        bet_type="TRIFECTA",
        selections=first_pos + second_pos + third_pos,
        cost=cost,
        rationale=(
            f"TRI KEY {top_label} / "
            f"{','.join(second_pos)} / {','.join(third_pos)} — "
            f"{combo_count} combos ${cost:.0f}"
        ),
        details={
            "structure": "key_tri",
            "first": first_pos,
            "second": second_pos,
            "third": third_pos,
            "combo_count": combo_count,
            "horses": {n: horse_ids.get(n, {}) for n in set(first_pos + second_pos + third_pos)},
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


# ---------------------------------------------------------------------------
# Multi-Race (Pick 3 / Pick 6)
# ---------------------------------------------------------------------------

@dataclass
class MultiRaceLeg:
    """One leg of a Pick 3 or Pick 6 ticket."""
    race_number: int
    grade: str
    grade_reasons: List[str]
    horses: List[str]           # selected horse names
    posts: List[str]            # corresponding posts
    horse_count: int
    top_confidence: float       # confidence of #1 pick
    top_projection_type: str    # cycle of #1 pick
    figure_quality_pct: Optional[float] = None


@dataclass
class MultiRaceStrategy:
    """How many horses to use per grade."""
    a_count: int = 1            # A-leg: single (1)
    b_count: int = 3            # B-leg: 2-3 horses
    c_count: int = 5            # C-leg: 4-6 horses


@dataclass
class CountOverrides:
    """Per-bet-type horse count overrides. Used by Bet Commander."""
    a_count: int = 1
    b_count: int = 2
    c_count: int = 0
    c_cap: int = 2
    pinned_a: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)


def apply_overrides(
    ranked: List[Dict[str, Any]],
    overrides: CountOverrides,
) -> List[Dict[str, Any]]:
    """Apply pin/toss overrides to a ranked projection list.

    - Excluded horses are removed entirely
    - Pinned-A horses are moved to front (preserving relative order)
    - Remaining horses follow in original sheets-first order
    """
    excl = {n.upper() for n in overrides.excluded}
    pinned = {n.upper() for n in overrides.pinned_a}
    filtered = [p for p in ranked if _horse_name(p).upper() not in excl]
    pins = [p for p in filtered if _horse_name(p).upper() in pinned]
    rest = [p for p in filtered if _horse_name(p).upper() not in pinned]
    return pins + rest


@dataclass
class MultiRacePlan:
    """Complete Pick 3 or Pick 6 ticket plan."""
    bet_type: str               # "PICK3" or "PICK6"
    start_race: int
    legs: List[MultiRaceLeg]
    combinations: int           # product of horses per leg
    cost: float                 # combinations × base_unit
    budget: float               # user-set cap
    over_budget: bool
    c_leg_count: int            # number of C-grade legs
    c_leg_warning: bool         # True if c_leg_count > 1 and not overridden
    warnings: List[str]
    settings: Dict[str, Any]


def build_multi_race_plan(
    race_projections: Dict[int, List[Dict[str, Any]]],
    start_race: int,
    bet_type: str,              # "PICK3" or "PICK6"
    budget: float,
    strategy: MultiRaceStrategy,
    settings: BetSettings,
    race_quality: Optional[Dict[int, float]] = None,
    c_leg_override: bool = False,
) -> MultiRacePlan:
    """Build a Pick 3 or Pick 6 ticket from graded race projections.

    Algorithm:
    - For each leg, grade the race A/B/C
    - A → strategy.a_count horses, B → strategy.b_count, C → strategy.c_count
    - Select top N non-tossed horses by bias_score
    - Flag >1 C-leg (unless overridden), over-budget, low quality
    """
    leg_count = 3 if bet_type == "PICK3" else 6
    base_unit = _BET_BASE
    race_quality = race_quality or {}

    legs: List[MultiRaceLeg] = []
    warnings: List[str] = []

    for i in range(leg_count):
        rn = start_race + i
        projs = race_projections.get(rn)

        if not projs:
            warnings.append(f"R{rn}: no projections available")
            legs.append(MultiRaceLeg(
                race_number=rn, grade="C", grade_reasons=["no projections"],
                horses=[], posts=[], horse_count=0,
                top_confidence=0, top_projection_type="UNKNOWN",
            ))
            continue

        quality = race_quality.get(rn)
        figure_quality_pct = quality if quality is not None else None
        grade, reasons = grade_race(projs, settings, figure_quality_pct=figure_quality_pct)

        # Determine horse count based on grade
        if grade == "A":
            n_horses = strategy.a_count
        elif grade == "B":
            n_horses = strategy.b_count
        else:
            n_horses = strategy.c_count

        # Select top N non-tossed horses
        ranked = sorted(projs, key=lambda p: p.get("bias_score", 0), reverse=True)
        non_tossed = [p for p in ranked if not p.get("tossed", False)]
        if not non_tossed:
            non_tossed = ranked  # fallback: use all if all tossed
        selected = non_tossed[:n_horses]

        top = selected[0] if selected else ranked[0]

        # Quality warning
        if figure_quality_pct is not None and figure_quality_pct < settings.figure_quality_threshold:
            warnings.append(f"R{rn}: figure quality {figure_quality_pct:.0%} below threshold")

        legs.append(MultiRaceLeg(
            race_number=rn,
            grade=grade,
            grade_reasons=reasons,
            horses=[_horse_name(p) for p in selected],
            posts=[str(p.get("post", "")) for p in selected],
            horse_count=len(selected),
            top_confidence=top.get("confidence", 0),
            top_projection_type=top.get("projection_type", "UNKNOWN"),
            figure_quality_pct=figure_quality_pct,
        ))

    c_leg_count = sum(1 for leg in legs if leg.grade == "C")
    c_leg_warning = c_leg_count > 1 and not c_leg_override

    if c_leg_warning:
        warnings.append(f"More than 1 C-leg ({c_leg_count}) — consider revising or overriding")

    # Compute combinations and cost
    horse_counts = [max(leg.horse_count, 1) for leg in legs]
    combinations = 1
    for hc in horse_counts:
        combinations *= hc
    cost = combinations * base_unit

    over_budget = cost > budget
    if over_budget:
        warnings.append(f"Cost ${cost:.0f} exceeds budget ${budget:.0f}")

    return MultiRacePlan(
        bet_type=bet_type,
        start_race=start_race,
        legs=legs,
        combinations=combinations,
        cost=cost,
        budget=budget,
        over_budget=over_budget,
        c_leg_count=c_leg_count,
        c_leg_warning=c_leg_warning,
        warnings=warnings,
        settings={
            "a_count": strategy.a_count,
            "b_count": strategy.b_count,
            "c_count": strategy.c_count,
            "budget": budget,
            "c_leg_override": c_leg_override,
        },
    )


def multi_race_plan_to_dict(plan: MultiRacePlan) -> Dict[str, Any]:
    """Serialize a MultiRacePlan to a JSON-serializable dict."""
    return {
        "bet_type": plan.bet_type,
        "start_race": plan.start_race,
        "legs": [
            {
                "race_number": leg.race_number,
                "grade": leg.grade,
                "grade_reasons": leg.grade_reasons,
                "horses": leg.horses,
                "posts": leg.posts,
                "horse_count": leg.horse_count,
                "top_confidence": leg.top_confidence,
                "top_projection_type": leg.top_projection_type,
                "figure_quality_pct": leg.figure_quality_pct,
            }
            for leg in plan.legs
        ],
        "combinations": plan.combinations,
        "cost": plan.cost,
        "budget": plan.budget,
        "over_budget": plan.over_budget,
        "c_leg_count": plan.c_leg_count,
        "c_leg_warning": plan.c_leg_warning,
        "warnings": plan.warnings,
        "settings": plan.settings,
    }


def multi_race_plan_to_text(plan: MultiRacePlan) -> str:
    """Human-readable text summary of a multi-race plan."""
    end_race = plan.start_race + len(plan.legs) - 1
    lines = [
        f"=== {plan.bet_type} (Races {plan.start_race}-{end_race}) ===",
        f"Budget: ${plan.budget:.0f} | Cost: ${plan.cost:.0f} | Combos: {plan.combinations}",
    ]
    if plan.over_budget:
        lines.append("*** OVER BUDGET ***")
    if plan.c_leg_warning:
        lines.append(f"*** WARNING: {plan.c_leg_count} C-legs ***")
    lines.append("")

    for i, leg in enumerate(plan.legs, 1):
        horses_str = ", ".join(leg.horses) if leg.horses else "(none)"
        tag = "single" if leg.horse_count == 1 else f"{leg.horse_count} horses"
        lines.append(f"Leg {i} (R{leg.race_number}) Grade {leg.grade}: {horses_str} [{tag}]")
        lines.append(f"  Top: {leg.top_projection_type} @ {leg.top_confidence:.0%}")

    if plan.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in plan.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def multi_race_plan_to_csv(plan: MultiRacePlan) -> str:
    """CSV export of multi-race plan legs."""
    lines = ["leg,race,grade,horses,horse_count,cost_contribution"]
    base_unit = _BET_BASE
    for i, leg in enumerate(plan.legs, 1):
        horses_str = "; ".join(leg.horses) if leg.horses else ""
        # Cost contribution: this leg's horse count as part of total
        other_combos = plan.combinations // max(leg.horse_count, 1) if leg.horse_count > 0 else 0
        cost_contrib = leg.horse_count * other_combos * base_unit if other_combos > 0 else 0
        lines.append(f"{i},{leg.race_number},{leg.grade},\"{horses_str}\",{leg.horse_count},{cost_contrib:.0f}")
    return "\n".join(lines)


# ======================================================================
# Daily Double planner
# ======================================================================

@dataclass
class DailyDoubleTicket:
    """One ticket within a Daily Double plan."""
    base: float
    leg1: List[str]
    leg2: List[str]
    cost: float
    reason: str


@dataclass
class DailyDoublePlan:
    """Complete Daily Double plan for two consecutive races."""
    bet_type: str = "DAILY_DOUBLE"
    start_race: int = 0
    tickets: List[DailyDoubleTicket] = field(default_factory=list)
    total_cost: float = 0.0
    passed: bool = False
    pass_reason: str = ""
    warnings: List[str] = field(default_factory=list)
    leg1_grade: str = ""
    leg2_grade: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dual Mode Betting — Settings + Plan dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DualModeSettings:
    """Settings for dual-mode (Profit + Score) betting."""
    bankroll: float = 1000.0
    risk_profile: str = "standard"

    # Budget split
    score_budget_pct: float = 0.20          # 20% of bankroll → Score mode

    # Profit mode thresholds
    profit_min_overlay: float = 1.25        # WIN requires 25% overlay
    profit_min_odds_a: float = 2.0
    profit_min_odds_b: float = 4.0
    figure_quality_threshold: float = 0.80

    # Score mode thresholds
    score_min_odds: float = 8.0             # 8/1 minimum
    score_min_overlay: float = 1.60         # 60% overlay
    score_max_c_per_leg: int = 2            # cap C horses in chaos leg
    score_require_singles: int = 2          # min singles for P6

    # DD allocation multipliers
    dd_aa_pct: float = 1.0                  # full alloc for A×A
    dd_ab_pct: float = 0.5                  # half alloc for A×B / B×A

    # Shared
    paper_mode: bool = True
    mandatory_payout: bool = False          # user flags big carryover day

    @property
    def profit_budget(self) -> float:
        return self.bankroll * (1.0 - self.score_budget_pct)

    @property
    def score_budget(self) -> float:
        return self.bankroll * self.score_budget_pct


@dataclass
class ProfitModePlan:
    """Profit mode output: WIN bets + Daily Double plans."""
    win_bets: List[Dict[str, Any]] = field(default_factory=list)
    dd_plans: List[Dict[str, Any]] = field(default_factory=list)
    total_risk: float = 0.0
    budget: float = 0.0
    passed_races: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ScoreModePlan:
    """Score mode output: Pick3 and/or Pick6 plans."""
    pick3_plans: List[Dict[str, Any]] = field(default_factory=list)
    pick6_plan: Optional[Dict[str, Any]] = None
    total_risk: float = 0.0
    budget: float = 0.0
    budget_exhausted: bool = False
    passed_sequences: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DualModeDayPlan:
    """Combined Profit + Score mode output for a full day."""
    mode: str = "both"
    profit: Optional[ProfitModePlan] = None
    score: Optional[ScoreModePlan] = None
    total_risk: float = 0.0
    profit_budget: float = 0.0
    score_budget: float = 0.0
    settings: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _tier_horses(
    projs: List[Dict[str, Any]],
    settings: BetSettings,
    strategy: MultiRaceStrategy,
    figure_quality_pct: Optional[float],
) -> tuple:
    """Assign horses to A/B/C tiers using sheets-first rank.

    Returns (a_horses, b_horses, c_horses, grade, is_chaos, has_single).
    """
    # Sort by sheets-first rank (cycle_priority desc, then raw_score desc)
    ranked = sorted(
        projs,
        key=lambda p: (-p.get("cycle_priority", 2), -p.get("raw_score", 0), -p.get("confidence", 0)),
    )
    non_tossed = [p for p in ranked if not p.get("tossed", False)]
    if not non_tossed:
        non_tossed = ranked

    # Exclude NEW_TOP_BOUNCE from A tier unless no alternatives
    clean = [p for p in non_tossed if p.get("projection_type") != "NEW_TOP_BOUNCE"]
    a_pool = clean if clean else non_tossed

    a_count = min(strategy.a_count, len(a_pool))
    a_horses = [_horse_name(p) for p in a_pool[:a_count]]

    # B = next horses after A
    remaining = [p for p in non_tossed if _horse_name(p) not in a_horses]
    b_count = min(strategy.b_count, len(remaining))
    b_horses = [_horse_name(p) for p in remaining[:b_count]]

    # C = only in chaos conditions
    is_chaos = False
    low_quality = figure_quality_pct is not None and figure_quality_pct < settings.figure_quality_threshold
    if low_quality:
        is_chaos = True
    c_remaining = [p for p in non_tossed if _horse_name(p) not in a_horses and _horse_name(p) not in b_horses]
    c_cap = min(2, strategy.c_count, len(c_remaining))
    c_horses = [_horse_name(p) for p in c_remaining[:c_cap]] if is_chaos else []

    # Grade the leg
    grade, _ = grade_race(projs, settings, figure_quality_pct=figure_quality_pct)

    # TRUE SINGLE detection
    top = a_pool[0] if a_pool else non_tossed[0]
    has_single = is_true_single(top)

    return a_horses, b_horses, c_horses, grade, is_chaos, has_single


def build_daily_double_plan(
    race_projections: Dict[int, List[Dict[str, Any]]],
    start_race: int,
    budget: float,
    settings: BetSettings,
    strategy: MultiRaceStrategy,
    race_quality: Optional[Dict[int, float]] = None,
) -> DailyDoublePlan:
    """Build a Daily Double ticket for two consecutive races.

    Uses A/B/C tier system, TRUE SINGLE detection, and PASS rules.
    """
    race_quality = race_quality or {}
    warnings: List[str] = []
    base = _BET_BASE

    r1, r2 = start_race, start_race + 1
    projs1 = race_projections.get(r1, [])
    projs2 = race_projections.get(r2, [])

    if not projs1 or not projs2:
        missing = []
        if not projs1:
            missing.append(f"R{r1}")
        if not projs2:
            missing.append(f"R{r2}")
        return DailyDoublePlan(
            start_race=start_race, passed=True,
            pass_reason=f"No projections for {', '.join(missing)}",
            warnings=[f"Missing projections: {', '.join(missing)}"],
        )

    q1 = race_quality.get(r1)
    q2 = race_quality.get(r2)

    a1, b1, c1, g1, chaos1, single1 = _tier_horses(projs1, settings, strategy, q1)
    a2, b2, c2, g2, chaos2, single2 = _tier_horses(projs2, settings, strategy, q2)

    # --- PASS checks ---
    if g1 == "C" and g2 == "C":
        return DailyDoublePlan(
            start_race=start_race, passed=True,
            pass_reason="Both legs are C-grade races",
            leg1_grade=g1, leg2_grade=g2,
        )

    low_q1 = q1 is not None and q1 < settings.figure_quality_threshold
    low_q2 = q2 is not None and q2 < settings.figure_quality_threshold
    if low_q1 and low_q2:
        return DailyDoublePlan(
            start_race=start_race, passed=True,
            pass_reason="Both legs have poor figure quality",
            leg1_grade=g1, leg2_grade=g2,
        )

    # Both legs dominated by bounce risk
    top1_bounce = projs1 and sorted(projs1, key=lambda p: -p.get("cycle_priority", 0))[0].get("projection_type") == "NEW_TOP_BOUNCE"
    top2_bounce = projs2 and sorted(projs2, key=lambda p: -p.get("cycle_priority", 0))[0].get("projection_type") == "NEW_TOP_BOUNCE"
    clean1 = [p for p in projs1 if p.get("projection_type") != "NEW_TOP_BOUNCE" and not p.get("tossed", False)]
    clean2 = [p for p in projs2 if p.get("projection_type") != "NEW_TOP_BOUNCE" and not p.get("tossed", False)]
    if top1_bounce and not clean1 and top2_bounce and not clean2:
        return DailyDoublePlan(
            start_race=start_race, passed=True,
            pass_reason="Both legs dominated by bounce risk with no clean alternatives",
            leg1_grade=g1, leg2_grade=g2,
        )

    # --- Ticket construction ---
    tickets: List[DailyDoubleTicket] = []

    if single1:
        # Single x (A+B) of leg2
        leg2_all = a2 + b2
        if leg2_all:
            cost = len(leg2_all) * base
            tickets.append(DailyDoubleTicket(
                base=base, leg1=a1[:1], leg2=leg2_all,
                cost=cost, reason=f"TRUE SINGLE R{r1} x A+B R{r2}",
            ))
    elif single2:
        # (A+B) of leg1 x Single
        leg1_all = a1 + b1
        if leg1_all:
            cost = len(leg1_all) * base
            tickets.append(DailyDoubleTicket(
                base=base, leg1=leg1_all, leg2=a2[:1],
                cost=cost, reason=f"A+B R{r1} x TRUE SINGLE R{r2}",
            ))
    else:
        # A x A
        if a1 and a2:
            cost = len(a1) * len(a2) * base
            tickets.append(DailyDoubleTicket(
                base=base, leg1=a1, leg2=a2,
                cost=cost, reason="A x A",
            ))
        # A x B
        if a1 and b2:
            cost = len(a1) * len(b2) * base
            tickets.append(DailyDoubleTicket(
                base=base, leg1=a1, leg2=b2,
                cost=cost, reason="A x B",
            ))
        # B x A
        if b1 and a2:
            cost = len(b1) * len(a2) * base
            tickets.append(DailyDoubleTicket(
                base=base, leg1=b1, leg2=a2,
                cost=cost, reason="B x A",
            ))

    # Chaos containment: A x C in chaos leg only
    if chaos1 and c1 and a2:
        cost = len(c1) * len(a2) * base
        tickets.append(DailyDoubleTicket(
            base=base, leg1=c1, leg2=a2,
            cost=cost, reason=f"Chaos R{r1}: C x A",
        ))
        warnings.append(f"R{r1} is chaos — added C-tier coverage")
    if chaos2 and a1 and c2:
        cost = len(a1) * len(c2) * base
        tickets.append(DailyDoubleTicket(
            base=base, leg1=a1, leg2=c2,
            cost=cost, reason=f"Chaos R{r2}: A x C",
        ))
        warnings.append(f"R{r2} is chaos — added C-tier coverage")

    total_cost = sum(t.cost for t in tickets)
    if total_cost > budget:
        warnings.append(f"Cost ${total_cost:.0f} exceeds budget ${budget:.0f}")

    return DailyDoublePlan(
        start_race=start_race,
        tickets=tickets,
        total_cost=total_cost,
        passed=False,
        warnings=warnings,
        leg1_grade=g1,
        leg2_grade=g2,
        settings={
            "a_count": strategy.a_count,
            "b_count": strategy.b_count,
            "c_count": strategy.c_count,
            "budget": budget,
        },
    )


def daily_double_plan_to_dict(plan: DailyDoublePlan) -> Dict[str, Any]:
    """Serialize a DailyDoublePlan to a JSON-friendly dict."""
    return {
        "bet_type": plan.bet_type,
        "start_race": plan.start_race,
        "tickets": [
            {
                "base": t.base,
                "leg1": t.leg1,
                "leg2": t.leg2,
                "cost": t.cost,
                "reason": t.reason,
            }
            for t in plan.tickets
        ],
        "total_cost": plan.total_cost,
        "passed": plan.passed,
        "pass_reason": plan.pass_reason,
        "warnings": plan.warnings,
        "leg1_grade": plan.leg1_grade,
        "leg2_grade": plan.leg2_grade,
        "settings": plan.settings,
    }


def daily_double_plan_to_text(plan: DailyDoublePlan) -> str:
    """Human-readable summary of a Daily Double plan."""
    r2 = plan.start_race + 1
    lines = [f"=== DAILY DOUBLE (R{plan.start_race}-R{r2}) ==="]
    if plan.passed:
        lines.append(f"PASS: {plan.pass_reason}")
        return "\n".join(lines)

    lines.append(f"Leg 1 (R{plan.start_race}) Grade {plan.leg1_grade}")
    lines.append(f"Leg 2 (R{r2}) Grade {plan.leg2_grade}")
    lines.append(f"Total cost: ${plan.total_cost:.0f}")
    lines.append("")
    for i, t in enumerate(plan.tickets, 1):
        lines.append(f"Ticket {i}: {', '.join(t.leg1)} x {', '.join(t.leg2)}  ${t.cost:.0f}  ({t.reason})")
    if plan.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in plan.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dual Mode Builders
# ---------------------------------------------------------------------------

def build_profit_mode_plan(
    race_projections: Dict[int, List[Dict[str, Any]]],
    settings: DualModeSettings,
    race_quality: Optional[Dict[int, float]] = None,
    track: str = "",
) -> ProfitModePlan:
    """Build Profit Mode plan: WIN bets + Daily Doubles.

    WIN entry: TRUE SINGLE or (Grade A + figure_quality >= threshold).
    WIN overlay gate: >= settings.profit_min_overlay (1.25).
    DD: one leg TRUE SINGLE or both legs A/B with quality ok.
    DD allocation: A×A full, A×B/B×A half.
    """
    rq = race_quality or {}
    multiplier = _risk_multiplier(settings.risk_profile)
    budget = settings.profit_budget

    # Build a BetSettings shim for grade_race() compatibility
    _bs = BetSettings(
        bankroll=budget,
        risk_profile=settings.risk_profile,
        figure_quality_threshold=settings.figure_quality_threshold,
        min_odds_a=settings.profit_min_odds_a,
        min_odds_b=settings.profit_min_odds_b,
    )

    win_bets: List[Dict[str, Any]] = []
    passed_races: List[Dict[str, Any]] = []
    warnings: List[str] = []
    win_risk = 0.0
    qualifying_races: set = set()  # races that pass the entry gate

    race_nums = sorted(race_projections.keys())

    for rn in race_nums:
        projs = race_projections[rn]
        if not projs:
            continue

        quality = rq.get(rn)
        grade, reasons = grade_race(projs, _bs, figure_quality_pct=quality)

        # Sort by sheets-first rank
        ranked = sorted(
            projs,
            key=lambda p: (-p.get("cycle_priority", 2), -p.get("raw_score", 0), -p.get("confidence", 0)),
        )
        non_tossed = [p for p in ranked if not p.get("tossed", False)]
        if not non_tossed:
            passed_races.append({"race": rn, "reason": "All horses tossed"})
            continue
        top = non_tossed[0]

        # Entry gate: TRUE SINGLE or (Grade A + quality)
        single = is_true_single(top)
        quality_ok = quality is None or quality >= settings.figure_quality_threshold
        if not single and not (grade == "A" and quality_ok):
            reason = f"Grade {grade}"
            if not single:
                reason += ", no TRUE SINGLE"
            if quality is not None and not quality_ok:
                reason += f", figure quality {quality:.0%}"
            passed_races.append({"race": rn, "reason": reason})
            continue

        qualifying_races.add(rn)

        # Compute model probs + overlay
        probs = _compute_race_probs(projs)
        top_name = _horse_name(top)
        top_prob = next((p for p in probs if p["horse_name"] == top_name), None)
        if top_prob is None:
            passed_races.append({"race": rn, "reason": "Could not compute model prob"})
            continue

        odds = top_prob["odds"]
        overlay = top_prob["overlay"]
        mp = top_prob["model_prob"]

        # Odds check
        if odds is None:
            passed_races.append({"race": rn, "reason": "No ML odds available"})
            continue

        # Overlay gate
        if overlay is None or overlay < settings.profit_min_overlay:
            ov_str = f"{overlay:.2f}" if overlay else "N/A"
            passed_races.append({
                "race": rn,
                "reason": f"Overlay {ov_str} below {settings.profit_min_overlay} threshold",
            })
            continue

        # Kelly sizing
        kf = kelly_fraction(odds, mp)
        if kf <= 0:
            passed_races.append({"race": rn, "reason": "Negative Kelly (no edge)"})
            continue

        raw_stake = budget * kf * multiplier
        stake = round(raw_stake / _BET_BASE) * _BET_BASE
        stake = max(stake, _BET_BASE)

        # Budget check
        if win_risk + stake > budget:
            warnings.append(f"R{rn}: stake ${stake:.0f} would exceed profit budget")
            stake = max(_BET_BASE, round((budget - win_risk) / _BET_BASE) * _BET_BASE)
            if stake <= 0:
                break

        fair_odds = (1.0 / mp) - 1.0 if mp > 0 else None
        win_bets.append({
            "race": rn,
            "track": track,
            "horse": top_name,
            "post": str(top.get("post", "")),
            "grade": grade,
            "odds": odds,
            "odds_raw": top.get("odds_raw", ""),
            "model_prob": round(mp, 4),
            "implied_prob": top_prob["implied_prob"],
            "overlay": round(overlay, 3),
            "fair_odds": round(fair_odds, 2) if fair_odds else None,
            "kelly_fraction": round(kf, 4),
            "stake": stake,
            "projection_type": top.get("projection_type", ""),
            "confidence": top.get("confidence", 0),
            "true_single": single,
        })
        win_risk += stake

    # --- DD scan: consecutive qualifying pairs ---
    dd_plans: List[Dict[str, Any]] = []
    dd_risk = 0.0
    dd_budget_remaining = budget - win_risk

    strat = MultiRaceStrategy(a_count=1, b_count=3, c_count=2)

    for i in range(len(race_nums) - 1):
        r1, r2 = race_nums[i], race_nums[i + 1]
        if r2 != r1 + 1:
            continue  # must be consecutive
        if r1 not in qualifying_races and r2 not in qualifying_races:
            continue  # at least one leg must qualify

        projs1 = race_projections.get(r1, [])
        projs2 = race_projections.get(r2, [])
        if not projs1 or not projs2:
            continue

        q1 = rq.get(r1)
        q2 = rq.get(r2)
        g1, _ = grade_race(projs1, _bs, figure_quality_pct=q1)
        g2, _ = grade_race(projs2, _bs, figure_quality_pct=q2)

        # Check top horses for TRUE SINGLE
        top1 = sorted(projs1, key=lambda p: (-p.get("cycle_priority", 2), -p.get("raw_score", 0)))
        top1 = [p for p in top1 if not p.get("tossed", False)] or top1
        top2 = sorted(projs2, key=lambda p: (-p.get("cycle_priority", 2), -p.get("raw_score", 0)))
        top2 = [p for p in top2 if not p.get("tossed", False)] or top2

        single1 = is_true_single(top1[0]) if top1 else False
        single2 = is_true_single(top2[0]) if top2 else False

        # DD gate: one leg TRUE SINGLE or both legs A/B with quality ok
        both_ab = g1 in ("A", "B") and g2 in ("A", "B")
        q_ok = (q1 is None or q1 >= settings.figure_quality_threshold) and (
            q2 is None or q2 >= settings.figure_quality_threshold
        )
        if not single1 and not single2 and not (both_ab and q_ok):
            continue

        if dd_budget_remaining <= 0:
            break

        dd_projs = {r1: projs1, r2: projs2}
        dd_quality = {}
        if q1 is not None:
            dd_quality[r1] = q1
        if q2 is not None:
            dd_quality[r2] = q2

        dd_plan = build_daily_double_plan(
            dd_projs, r1, dd_budget_remaining, _bs, strat,
            race_quality=dd_quality or None,
        )
        if dd_plan.passed:
            continue

        # Scale tickets by allocation pcts
        for t in dd_plan.tickets:
            reason_lower = t.reason.lower()
            if "a x a" in reason_lower or "single" in reason_lower:
                t.cost = round(t.cost * settings.dd_aa_pct / _BET_BASE) * _BET_BASE
            else:
                t.cost = round(t.cost * settings.dd_ab_pct / _BET_BASE) * _BET_BASE
            t.cost = max(t.cost, _BET_BASE)
        dd_plan.total_cost = sum(t.cost for t in dd_plan.tickets)

        if dd_risk + dd_plan.total_cost > dd_budget_remaining:
            warnings.append(f"DD R{r1}-R{r2}: cost ${dd_plan.total_cost:.0f} exceeds remaining budget")
            continue

        dd_plans.append(daily_double_plan_to_dict(dd_plan))
        dd_risk += dd_plan.total_cost

    total = win_risk + dd_risk
    return ProfitModePlan(
        win_bets=win_bets,
        dd_plans=dd_plans,
        total_risk=total,
        budget=budget,
        passed_races=passed_races,
        warnings=warnings,
    )


def build_score_mode_plan(
    race_projections: Dict[int, List[Dict[str, Any]]],
    settings: DualModeSettings,
    race_quality: Optional[Dict[int, float]] = None,
) -> ScoreModePlan:
    """Build Score Mode plan: Pick3 (exactly 1 chaos leg) and/or Pick6 (mandatory payout).

    Score trigger: odds >= 8/1, overlay >= 1.6, no bounce risk.
    Pick3: exactly 1 chaos leg in 3-race window, C capped at score_max_c_per_leg.
    Pick6: mandatory_payout + >= score_require_singles singles.
    Hard stop when score_budget exhausted.
    """
    rq = race_quality or {}
    budget = settings.score_budget
    if budget <= 0:
        return ScoreModePlan(budget=0.0, budget_exhausted=True,
                             warnings=["Score budget is zero"])

    _bs = BetSettings(
        bankroll=budget,
        risk_profile=settings.risk_profile,
        figure_quality_threshold=settings.figure_quality_threshold,
    )

    race_nums = sorted(race_projections.keys())
    score_spent = 0.0
    pick3_plans: List[Dict[str, Any]] = []
    pick6_plan: Optional[Dict[str, Any]] = None
    passed_sequences: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # Pre-compute grades and score triggers per race
    race_grades: Dict[int, str] = {}
    race_has_trigger: Dict[int, bool] = {}
    race_has_single: Dict[int, bool] = {}

    for rn in race_nums:
        projs = race_projections[rn]
        if not projs:
            continue
        quality = rq.get(rn)
        grade, _ = grade_race(projs, _bs, figure_quality_pct=quality)
        race_grades[rn] = grade

        probs = _compute_race_probs(projs)
        ranked = sorted(projs, key=lambda p: (-p.get("cycle_priority", 2), -p.get("raw_score", 0)))
        non_tossed = [p for p in ranked if not p.get("tossed", False)] or ranked
        top = non_tossed[0]

        top_name = _horse_name(top)
        top_prob = next((p for p in probs if p["horse_name"] == top_name), None)
        overlay = top_prob["overlay"] if top_prob else None

        race_has_trigger[rn] = _is_score_trigger(
            top, overlay,
            score_min_odds=settings.score_min_odds,
            score_min_overlay=settings.score_min_overlay,
        )
        race_has_single[rn] = is_true_single(top)

    # --- Pick3 scan: 3-race sliding window ---
    for i in range(len(race_nums) - 2):
        if score_spent >= budget:
            break

        window = [race_nums[i], race_nums[i + 1], race_nums[i + 2]]
        # Must be consecutive
        if window[1] != window[0] + 1 or window[2] != window[1] + 1:
            continue

        chaos_legs = []
        for rn in window:
            quality = rq.get(rn)
            is_chaos = (race_grades.get(rn) == "C" or
                        (quality is not None and quality < settings.figure_quality_threshold))
            if is_chaos:
                chaos_legs.append(rn)

        if len(chaos_legs) != 1:
            passed_sequences.append({
                "races": window,
                "reason": f"Need exactly 1 chaos leg, found {len(chaos_legs)}",
            })
            continue

        # Need at least one score trigger
        has_trigger = any(race_has_trigger.get(rn, False) for rn in window)
        if not has_trigger:
            passed_sequences.append({
                "races": window,
                "reason": "No score-trigger horse (odds >= 8/1 + overlay >= 1.6)",
            })
            continue

        # Build Pick3 with capped C count for chaos leg
        p3_projs = {rn: race_projections[rn] for rn in window}
        p3_quality = {rn: rq[rn] for rn in window if rn in rq}
        strategy = MultiRaceStrategy(
            a_count=1, b_count=3,
            c_count=min(2, settings.score_max_c_per_leg),
        )
        p3_plan = build_multi_race_plan(
            p3_projs, window[0], "PICK3", budget - score_spent,
            strategy, _bs,
            race_quality=p3_quality or None,
            c_leg_override=True,  # we already validated chaos
        )
        if p3_plan.cost > budget - score_spent:
            passed_sequences.append({
                "races": window,
                "reason": f"Cost ${p3_plan.cost:.0f} exceeds remaining score budget ${budget - score_spent:.0f}",
            })
            continue

        pick3_plans.append(multi_race_plan_to_dict(p3_plan))
        score_spent += p3_plan.cost

    # --- Pick6 gate ---
    if not settings.mandatory_payout:
        passed_sequences.append({
            "races": race_nums[:6] if len(race_nums) >= 6 else race_nums,
            "reason": "Mandatory payout not flagged by user",
        })
    elif len(race_nums) < 6:
        passed_sequences.append({
            "races": race_nums,
            "reason": f"Need 6 consecutive races, only {len(race_nums)} available",
        })
    elif score_spent >= budget:
        warnings.append("Score budget exhausted before Pick6")
    else:
        # Find best 6-race consecutive window with most singles
        best_window = None
        best_singles = 0
        best_has_trigger = False
        for i in range(len(race_nums) - 5):
            window = race_nums[i:i + 6]
            # Check consecutive
            if any(window[j + 1] != window[j] + 1 for j in range(5)):
                continue
            singles = sum(1 for rn in window if race_has_single.get(rn, False))
            trigger = any(race_has_trigger.get(rn, False) for rn in window)
            if singles > best_singles or (singles == best_singles and trigger and not best_has_trigger):
                best_window = window
                best_singles = singles
                best_has_trigger = trigger

        if best_window is None:
            passed_sequences.append({
                "races": race_nums[:6],
                "reason": "No valid 6-race consecutive window found",
            })
        elif best_singles < settings.score_require_singles:
            passed_sequences.append({
                "races": best_window,
                "reason": f"Only {best_singles} singles, need >= {settings.score_require_singles}",
            })
        elif not best_has_trigger:
            passed_sequences.append({
                "races": best_window,
                "reason": "No score-trigger horse in window",
            })
        else:
            p6_projs = {rn: race_projections[rn] for rn in best_window}
            p6_quality = {rn: rq[rn] for rn in best_window if rn in rq}
            strategy = MultiRaceStrategy(
                a_count=1, b_count=3,
                c_count=min(2, settings.score_max_c_per_leg),
            )
            p6_plan = build_multi_race_plan(
                p6_projs, best_window[0], "PICK6", budget - score_spent,
                strategy, _bs,
                race_quality=p6_quality or None,
                c_leg_override=True,
            )
            if p6_plan.cost > budget - score_spent:
                warnings.append(
                    f"Pick6 cost ${p6_plan.cost:.0f} exceeds remaining score budget "
                    f"${budget - score_spent:.0f}"
                )
            else:
                pick6_plan = multi_race_plan_to_dict(p6_plan)
                score_spent += p6_plan.cost

    exhausted = score_spent >= budget
    return ScoreModePlan(
        pick3_plans=pick3_plans,
        pick6_plan=pick6_plan,
        total_risk=score_spent,
        budget=budget,
        budget_exhausted=exhausted,
        passed_sequences=passed_sequences,
        warnings=warnings,
    )


def build_dual_mode_day_plan(
    race_projections: Dict[int, List[Dict[str, Any]]],
    settings: DualModeSettings,
    mode: str = "both",
    race_quality: Optional[Dict[int, float]] = None,
    track: str = "",
) -> DualModeDayPlan:
    """Orchestrate Profit + Score modes under separate budgets."""
    profit_plan = None
    score_plan = None
    top_warnings: List[str] = []

    if mode in ("profit", "both"):
        profit_plan = build_profit_mode_plan(
            race_projections, settings,
            race_quality=race_quality, track=track,
        )
    if mode in ("score", "both"):
        score_plan = build_score_mode_plan(
            race_projections, settings,
            race_quality=race_quality,
        )

    total = 0.0
    if profit_plan:
        total += profit_plan.total_risk
    if score_plan:
        total += score_plan.total_risk

    if total > settings.bankroll:
        top_warnings.append(
            f"Total risk ${total:.0f} exceeds bankroll ${settings.bankroll:.0f}"
        )

    return DualModeDayPlan(
        mode=mode,
        profit=profit_plan,
        score=score_plan,
        total_risk=total,
        profit_budget=settings.profit_budget,
        score_budget=settings.score_budget,
        settings={
            "bankroll": settings.bankroll,
            "risk_profile": settings.risk_profile,
            "score_budget_pct": settings.score_budget_pct,
            "profit_min_overlay": settings.profit_min_overlay,
            "score_min_odds": settings.score_min_odds,
            "score_min_overlay": settings.score_min_overlay,
            "mandatory_payout": settings.mandatory_payout,
        },
        warnings=top_warnings,
    )


# ---------------------------------------------------------------------------
# Dual Mode Serializers
# ---------------------------------------------------------------------------

def _profit_plan_to_dict(plan: ProfitModePlan) -> Dict[str, Any]:
    return {
        "win_bets": plan.win_bets,
        "dd_plans": plan.dd_plans,
        "total_risk": plan.total_risk,
        "budget": plan.budget,
        "passed_races": plan.passed_races,
        "warnings": plan.warnings,
    }


def _score_plan_to_dict(plan: ScoreModePlan) -> Dict[str, Any]:
    return {
        "pick3_plans": plan.pick3_plans,
        "pick6_plan": plan.pick6_plan,
        "total_risk": plan.total_risk,
        "budget": plan.budget,
        "budget_exhausted": plan.budget_exhausted,
        "passed_sequences": plan.passed_sequences,
        "warnings": plan.warnings,
    }


def dual_mode_plan_to_dict(plan: DualModeDayPlan) -> Dict[str, Any]:
    """Serialize DualModeDayPlan to JSON-friendly dict."""
    return {
        "mode": plan.mode,
        "profit": _profit_plan_to_dict(plan.profit) if plan.profit else None,
        "score": _score_plan_to_dict(plan.score) if plan.score else None,
        "total_risk": plan.total_risk,
        "profit_budget": plan.profit_budget,
        "score_budget": plan.score_budget,
        "settings": plan.settings,
        "warnings": plan.warnings,
    }


def dual_mode_plan_to_text(plan: DualModeDayPlan) -> str:
    """Human-readable summary of a dual-mode day plan."""
    lines = ["=== DUAL MODE BETTING ==="]
    lines.append(f"Mode: {plan.mode.upper()}")
    lines.append(f"Bankroll: ${plan.settings.get('bankroll', 0):.0f}")
    lines.append(f"Profit budget: ${plan.profit_budget:.0f}  |  Score budget: ${plan.score_budget:.0f}")
    lines.append(f"Total risk: ${plan.total_risk:.0f}")
    lines.append("")

    if plan.profit:
        p = plan.profit
        lines.append("--- PROFIT MODE ---")
        if p.win_bets:
            lines.append(f"WIN bets ({len(p.win_bets)}):")
            for w in p.win_bets:
                lines.append(
                    f"  R{w['race']} {w['horse']}  odds={w.get('odds', '?')}"
                    f"  overlay={w.get('overlay', '?')}  stake=${w.get('stake', 0):.0f}"
                )
        if p.dd_plans:
            lines.append(f"Daily Doubles ({len(p.dd_plans)}):")
            for dd in p.dd_plans:
                lines.append(f"  R{dd['start_race']}-R{dd['start_race']+1}  ${dd['total_cost']:.0f}")
        if p.passed_races:
            lines.append("Passed races:")
            for pr in p.passed_races:
                lines.append(f"  R{pr['race']}: {pr['reason']}")
        lines.append(f"Profit risk: ${p.total_risk:.0f} / ${p.budget:.0f}")
        lines.append("")

    if plan.score:
        s = plan.score
        lines.append("--- SCORE MODE ---")
        if s.budget_exhausted:
            lines.append("BUDGET EXHAUSTED — hard stop")
        if s.pick3_plans:
            lines.append(f"Pick3 plans ({len(s.pick3_plans)}):")
            for p3 in s.pick3_plans:
                lines.append(f"  R{p3['start_race']}-R{p3['start_race']+2}  ${p3['cost']:.0f}")
        if s.pick6_plan:
            lines.append(f"Pick6: R{s.pick6_plan['start_race']}-R{s.pick6_plan['start_race']+5}  ${s.pick6_plan['cost']:.0f}")
        if s.passed_sequences:
            lines.append("Passed sequences:")
            for ps in s.passed_sequences:
                races_str = "-".join(str(r) for r in ps["races"])
                lines.append(f"  R{races_str}: {ps['reason']}")
        lines.append(f"Score risk: ${s.total_risk:.0f} / ${s.budget:.0f}")
        lines.append("")

    if plan.warnings:
        lines.append("Warnings:")
        for w in plan.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bet Commander — Race Analysis & Recommender
# ---------------------------------------------------------------------------

@dataclass
class RaceAnalysis:
    """Per-race analysis for Bet Commander card summary."""
    race_number: int
    grade: str
    grade_reasons: List[str]
    edge_score: float          # 0-100
    chaos_index: float         # 0.0-1.0
    has_true_single: bool
    top_overlay: Optional[float]
    figure_quality_pct: float
    field_size: int
    top_horse: str
    top_confidence: float
    top_cycle: str
    a_horses: List[str]
    b_horses: List[str]
    probs: List[Dict]


@dataclass
class BetRecommendation:
    """Auto-recommended play from the recommender engine."""
    bet_type: str              # WIN, EXACTA, TRIFECTA, DD, PICK3
    races: List[int]
    confidence: float          # 0-1
    reason_codes: List[str]
    reason_text: str
    key_horse: Optional[str]
    suggested_counts: Dict[int, "CountOverrides"]  # {race_num: overrides}


def analyze_race_for_commander(
    race_number: int,
    projs: List[Dict],
    settings: "BetSettings",
    quality: float,
) -> RaceAnalysis:
    """Compute edge_score and chaos_index for a single race.

    Parameters
    ----------
    projs : list of projection dicts (already sorted sheets-first)
    settings : BetSettings with odds/overlay info
    quality : figure_quality_pct for this race (0-1)
    """
    if not projs:
        return RaceAnalysis(
            race_number=race_number, grade="D",
            grade_reasons=["no projections"], edge_score=0,
            chaos_index=1.0, has_true_single=False,
            top_overlay=None, figure_quality_pct=quality,
            field_size=0, top_horse="", top_confidence=0,
            top_cycle="", a_horses=[], b_horses=[], probs=[],
        )

    grade, reasons = grade_race(projs, quality)
    non_tossed = [p for p in projs if not p.get("tossed", False)]
    if not non_tossed:
        non_tossed = projs

    top = non_tossed[0]
    top_name = _horse_name(top)
    top_conf = top.get("confidence", 0)
    top_cycle = top.get("projection_type", "NEUTRAL")
    top_cycle_pri = top.get("cycle_priority", 2)

    single = is_true_single(projs)

    # overlay
    top_odds = top.get("odds")
    top_overlay = None
    if top_odds and top_odds > 0:
        probs_list = _compute_race_probs(projs)
        if probs_list:
            fair = 1.0 / probs_list[0]["model_prob"] if probs_list[0]["model_prob"] > 0 else 99
            top_overlay = top_odds / fair if fair > 0 else None
    else:
        probs_list = _compute_race_probs(projs)

    # A/B horses
    a_horses = []
    b_horses = []
    for i, p in enumerate(non_tossed):
        nm = _horse_name(p)
        if i == 0:
            a_horses.append(nm)
        elif p.get("cycle_priority", 0) >= 4:
            a_horses.append(nm)
        elif p.get("cycle_priority", 0) >= 2:
            b_horses.append(nm)
        if len(a_horses) + len(b_horses) >= 6:
            break

    # --- edge_score (0-100) ---
    edge = 0.0
    if single:
        edge += 30
    edge += 20 * top_conf
    if top_cycle_pri >= 5:  # PAIRED or PAIRED_TOP
        edge += 15
    elif top_cycle_pri == 4:  # REBOUND
        edge += 10
    if grade == "A":
        edge += 10
    elif grade == "B":
        edge += 5
    if top_overlay and top_overlay > 1.0:
        edge += 15 * min((top_overlay - 1.0) / 1.0, 1.0)
    if quality >= 0.90:
        edge += 10
    edge = min(edge, 100.0)

    # --- chaos_index (0-1) ---
    chaos = 0.0
    if quality < 0.80:
        chaos += 0.25
    if top_conf < 0.50:
        chaos += 0.20
    if len(non_tossed) >= 2:
        gap_1_2 = abs(non_tossed[0].get("bias_score", 0) - non_tossed[1].get("bias_score", 0))
        if gap_1_2 < 1.0:
            chaos += 0.15
    # cluster: >3 within 2.0 pts of #1
    top_mid = non_tossed[0].get("proj_mid", 0)
    cluster = sum(1 for p in non_tossed[1:] if abs(p.get("proj_mid", 0) - top_mid) <= 2.0)
    if cluster > 3:
        chaos += 0.15
    if top.get("bounce_risk", False):
        chaos += 0.10
    if top.get("tossed", False):
        chaos += 0.10
    if len(projs) >= 12:
        chaos += 0.05
    chaos = min(chaos, 1.0)

    return RaceAnalysis(
        race_number=race_number,
        grade=grade,
        grade_reasons=reasons,
        edge_score=round(edge, 1),
        chaos_index=round(chaos, 2),
        has_true_single=single,
        top_overlay=round(top_overlay, 2) if top_overlay else None,
        figure_quality_pct=round(quality, 2),
        field_size=len(projs),
        top_horse=top_name,
        top_confidence=round(top_conf, 2),
        top_cycle=top_cycle,
        a_horses=a_horses,
        b_horses=b_horses,
        probs=probs_list if probs_list else [],
    )


def recommend_bets_for_card(
    race_projections: Dict[int, List[Dict]],
    race_quality: Dict[int, float],
    settings: "BetSettings",
) -> Tuple[Dict[int, RaceAnalysis], List[BetRecommendation]]:
    """Analyze all races and produce ordered bet recommendations.

    Returns
    -------
    analyses : {race_number: RaceAnalysis}
    recommendations : list of BetRecommendation sorted by confidence desc
    """
    analyses: Dict[int, RaceAnalysis] = {}
    for rn in sorted(race_projections.keys()):
        projs = race_projections[rn]
        q = race_quality.get(rn, 0.0)
        analyses[rn] = analyze_race_for_commander(rn, projs, settings, q)

    recs: List[BetRecommendation] = []
    race_nums = sorted(analyses.keys())

    for rn in race_nums:
        a = analyses[rn]

        # --- WIN ---
        if ((a.has_true_single or (len(a.a_horses) >= 1 and a.top_confidence >= 0.60))
                and a.top_overlay is not None and a.top_overlay >= 1.25
                and a.chaos_index <= 0.55):
            reasons = []
            if a.has_true_single:
                reasons.append("TRUE_SINGLE")
            if a.top_overlay >= 1.25:
                reasons.append(f"OVERLAY_{a.top_overlay:.2f}")
            if a.top_confidence >= 0.60:
                reasons.append(f"CONF_{a.top_confidence:.0%}")
            recs.append(BetRecommendation(
                bet_type="WIN",
                races=[rn],
                confidence=round(min(a.edge_score / 100, 0.95), 2),
                reason_codes=reasons,
                reason_text=f"R{rn} {a.top_horse}: {', '.join(reasons)}",
                key_horse=a.top_horse,
                suggested_counts={rn: CountOverrides(a_count=1, b_count=0, c_count=0)},
            ))

        # --- EXACTA ---
        if (len(a.a_horses) >= 1 and len(a.a_horses) + len(a.b_horses) >= 2
                and a.chaos_index <= 0.55):
            top2_prob = sum(p["model_prob"] for p in a.probs[:2]) if len(a.probs) >= 2 else 0
            if top2_prob >= 0.45:
                recs.append(BetRecommendation(
                    bet_type="EXACTA",
                    races=[rn],
                    confidence=round(top2_prob * 0.85, 2),
                    reason_codes=["TOP2_STRONG", f"PROB_{top2_prob:.0%}"],
                    reason_text=f"R{rn} exacta: top-2 combined prob {top2_prob:.0%}",
                    key_horse=a.top_horse,
                    suggested_counts={rn: CountOverrides(a_count=1, b_count=3, c_count=0)},
                ))

        # --- TRIFECTA ---
        if (a.chaos_index <= 0.45 and len(a.probs) >= 4):
            top3_prob = sum(p["model_prob"] for p in a.probs[:3])
            gap_3_4 = 0
            if len(a.probs) >= 4:
                gap_3_4 = a.probs[2].get("model_prob", 0) - a.probs[3].get("model_prob", 0)
            if top3_prob >= 0.55 and gap_3_4 >= 0.02:
                recs.append(BetRecommendation(
                    bet_type="TRIFECTA",
                    races=[rn],
                    confidence=round(top3_prob * 0.75, 2),
                    reason_codes=["TOP3_SEPARATION", f"PROB_{top3_prob:.0%}"],
                    reason_text=f"R{rn} tri: top-3 prob {top3_prob:.0%}, clear separation",
                    key_horse=a.top_horse,
                    suggested_counts={rn: CountOverrides(a_count=1, b_count=2, c_count=2)},
                ))

    # --- DD (adjacent pairs) ---
    for i in range(len(race_nums) - 1):
        r1, r2 = race_nums[i], race_nums[i + 1]
        a1, a2 = analyses[r1], analyses[r2]
        if r2 - r1 != 1:
            continue
        has_anchor = a1.has_true_single or a2.has_true_single
        both_decent = a1.figure_quality_pct >= 0.70 and a2.figure_quality_pct >= 0.70
        if has_anchor and both_decent and min(a1.chaos_index, a2.chaos_index) <= 0.55:
            anchor_race = r1 if a1.edge_score >= a2.edge_score else r2
            anchor_a = analyses[anchor_race]
            conf = round((a1.edge_score + a2.edge_score) / 200 * 0.9, 2)
            recs.append(BetRecommendation(
                bet_type="DD",
                races=[r1, r2],
                confidence=conf,
                reason_codes=["ANCHOR_SINGLE" if has_anchor else "STRONG_PAIR"],
                reason_text=f"DD R{r1}-R{r2}: anchor {anchor_a.top_horse}",
                key_horse=anchor_a.top_horse,
                suggested_counts={
                    r1: CountOverrides(a_count=1, b_count=2, c_count=0),
                    r2: CountOverrides(a_count=1, b_count=2, c_count=0),
                },
            ))

    # --- PICK3 (3 consecutive, exactly 1 chaos leg) ---
    for i in range(len(race_nums) - 2):
        r1, r2, r3 = race_nums[i], race_nums[i + 1], race_nums[i + 2]
        if r3 - r1 != 2:
            continue
        trio = [analyses[r1], analyses[r2], analyses[r3]]
        chaos_legs = [a for a in trio if a.chaos_index > 0.55]
        single_legs = [a for a in trio if a.has_true_single]
        if len(chaos_legs) == 1 and len(single_legs) >= 1:
            conf = round(sum(a.edge_score for a in trio) / 300 * 0.85, 2)
            chaos_rn = chaos_legs[0].race_number
            counts = {}
            for a in trio:
                if a.race_number == chaos_rn:
                    counts[a.race_number] = CountOverrides(a_count=1, b_count=2, c_count=3)
                else:
                    counts[a.race_number] = CountOverrides(a_count=1, b_count=2, c_count=0)
            recs.append(BetRecommendation(
                bet_type="PICK3",
                races=[r1, r2, r3],
                confidence=conf,
                reason_codes=["1_CHAOS_LEG", f"SINGLES_{len(single_legs)}"],
                reason_text=f"P3 R{r1}-R{r3}: spread R{chaos_rn}, anchor elsewhere",
                key_horse=single_legs[0].top_horse if single_legs else None,
                suggested_counts=counts,
            ))

    recs.sort(key=lambda r: r.confidence, reverse=True)
    return analyses, recs


# ---------------------------------------------------------------------------
# Override-Aware Wrappers
# ---------------------------------------------------------------------------

def build_daily_double_with_overrides(
    race_projections: Dict[int, List[Dict[str, Any]]],
    start_race: int,
    budget: float,
    settings: "BetSettings",
    strategy: MultiRaceStrategy,
    per_leg_overrides: Dict[int, CountOverrides],
    race_quality: Optional[Dict[int, float]] = None,
) -> "DailyDoublePlan":
    """Build DD with per-leg CountOverrides applied."""
    modified = {}
    for rn, projs in race_projections.items():
        if rn in per_leg_overrides:
            ov = per_leg_overrides[rn]
            modified[rn] = apply_overrides(projs, ov)
        else:
            modified[rn] = projs

    # Build a strategy that reflects the overrides for the two legs
    r1, r2 = start_race, start_race + 1
    ov1 = per_leg_overrides.get(r1)
    ov2 = per_leg_overrides.get(r2)
    # Use the max of override counts across legs for the strategy
    mod_strategy = MultiRaceStrategy(
        a_count=max(ov1.a_count if ov1 else strategy.a_count,
                    ov2.a_count if ov2 else strategy.a_count),
        b_count=max(ov1.b_count if ov1 else strategy.b_count,
                    ov2.b_count if ov2 else strategy.b_count),
        c_count=max(ov1.c_count if ov1 else strategy.c_count,
                    ov2.c_count if ov2 else strategy.c_count),
    )

    return build_daily_double_plan(
        modified, start_race, budget, settings, mod_strategy, race_quality,
    )


def build_multi_race_with_overrides(
    race_projections: Dict[int, List[Dict[str, Any]]],
    start_race: int,
    bet_type: str,
    budget: float,
    strategy: MultiRaceStrategy,
    per_leg_overrides: Dict[int, CountOverrides],
    settings: "BetSettings",
    race_quality: Optional[Dict[int, float]] = None,
) -> "MultiRacePlan":
    """Build multi-race plan (P3/P4/P5/P6) with per-leg overrides."""
    modified = {}
    for rn, projs in race_projections.items():
        if rn in per_leg_overrides:
            ov = per_leg_overrides[rn]
            modified[rn] = apply_overrides(projs, ov)
        else:
            modified[rn] = projs

    # Strategy from max of all leg overrides
    a_counts = [per_leg_overrides[rn].a_count for rn in per_leg_overrides if rn in per_leg_overrides]
    b_counts = [per_leg_overrides[rn].b_count for rn in per_leg_overrides if rn in per_leg_overrides]
    c_counts = [per_leg_overrides[rn].c_count for rn in per_leg_overrides if rn in per_leg_overrides]
    mod_strategy = MultiRaceStrategy(
        a_count=max(a_counts) if a_counts else strategy.a_count,
        b_count=max(b_counts) if b_counts else strategy.b_count,
        c_count=max(c_counts) if c_counts else strategy.c_count,
    )

    return build_multi_race_plan(
        modified, start_race, bet_type, budget, mod_strategy, settings, race_quality,
    )


# ---------------------------------------------------------------------------
# Commander Export
# ---------------------------------------------------------------------------

def commander_slip_to_text(slip_entries: List[Dict]) -> str:
    """Format bet slip as printable text."""
    lines = ["=" * 50, "BET COMMANDER — TICKET SLIP", "=" * 50, ""]
    total = 0.0
    for i, entry in enumerate(slip_entries, 1):
        bt = entry.get("bet_type", "?")
        races = entry.get("races", [])
        race_str = "-".join(str(r) for r in races)
        cost = entry.get("total_cost", 0)
        total += cost
        lines.append(f"{i}. {bt}  R{race_str}  ${cost:.2f}")
        for t in entry.get("computed_tickets", []):
            if isinstance(t, Ticket):
                lines.append(f"   {t.rationale}")
            elif isinstance(t, dict):
                lines.append(f"   {t.get('rationale', '')}")
        lines.append("")
    lines.append("-" * 50)
    lines.append(f"TOTAL: ${total:.2f}")
    lines.append("=" * 50)
    return "\n".join(lines)


def commander_slip_to_json(slip_entries: List[Dict]) -> str:
    """Export bet slip as JSON string."""
    import json

    export = []
    for entry in slip_entries:
        tickets_out = []
        for t in entry.get("computed_tickets", []):
            if isinstance(t, Ticket):
                tickets_out.append({
                    "bet_type": t.bet_type,
                    "selections": t.selections,
                    "cost": t.cost,
                    "rationale": t.rationale,
                    "details": t.details,
                })
            elif isinstance(t, dict):
                tickets_out.append(t)
        export.append({
            "bet_type": entry.get("bet_type"),
            "races": entry.get("races", []),
            "base_wager": entry.get("base_wager", 2),
            "total_cost": entry.get("total_cost", 0),
            "tickets": tickets_out,
            "leg_overrides": {
                str(rn): {
                    "a_count": ov.a_count, "b_count": ov.b_count,
                    "c_count": ov.c_count,
                    "pinned_a": ov.pinned_a, "excluded": ov.excluded,
                }
                for rn, ov in entry.get("leg_overrides", {}).items()
                if isinstance(ov, CountOverrides)
            },
        })
    return json.dumps(export, indent=2)


def race_analysis_to_dict(a: "RaceAnalysis") -> Dict[str, Any]:
    """Serialize RaceAnalysis for API response."""
    return {
        "race_number": a.race_number,
        "grade": a.grade,
        "grade_reasons": a.grade_reasons,
        "edge_score": a.edge_score,
        "chaos_index": a.chaos_index,
        "has_true_single": a.has_true_single,
        "top_overlay": a.top_overlay,
        "figure_quality_pct": a.figure_quality_pct,
        "field_size": a.field_size,
        "top_horse": a.top_horse,
        "top_confidence": a.top_confidence,
        "top_cycle": a.top_cycle,
        "a_horses": a.a_horses,
        "b_horses": a.b_horses,
    }


def bet_recommendation_to_dict(r: "BetRecommendation") -> Dict[str, Any]:
    """Serialize BetRecommendation for API response."""
    return {
        "bet_type": r.bet_type,
        "races": r.races,
        "confidence": r.confidence,
        "reason_codes": r.reason_codes,
        "reason_text": r.reason_text,
        "key_horse": r.key_horse,
        "suggested_counts": {
            str(rn): {
                "a_count": ov.a_count, "b_count": ov.b_count,
                "c_count": ov.c_count,
            }
            for rn, ov in r.suggested_counts.items()
        },
    }
