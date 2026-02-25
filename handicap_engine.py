"""Deterministic handicapping engine for Ragozin-style analysis.

Cycle-based projection types (per Ragozin Sheets Edge permanent rules)
----------------------------------------------------------------------
PAIRED        – Last two figures at/near best, within PAIRED_TOL.
                Project last figure ±0-2.  Highest confidence.
REBOUND       – Top → worse → improving back within REBOUND_WINDOW starts.
                High confidence — resilience signal.
IMPROVING     – Two or more consecutive improvements, last move < BIG_NEW_TOP.
                Project dampened continuation.
NEW_TOP_BOUNCE – Last race is a big new top (>= BIG_NEW_TOP better than all prior).
                 Expect regression: project +BOUNCE_LOW to +BOUNCE_HIGH worse.
TAIL_OFF      – Two or more consecutive deteriorations.
                Downgrade — project continued decline.
NEUTRAL       – No clear cycle signal.  Project around recent figure ±2.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math

# ======================================================================
# Ragozin Sheets Edge — tunable constants
# (do not change unless user says)
# ======================================================================
PAIRED_TOL = 1.0        # max gap between last two figs for PAIRED
BIG_NEW_TOP = 5.0       # improvement threshold for bounce risk
BOUNCE_LOW = 2.0        # min regression projected after big new top
BOUNCE_HIGH = 6.0       # max regression projected after big new top
REBOUND_WINDOW = 3      # max starts to look back for rebound pattern
LAYOFF_LONG = 60        # days — long layoff threshold
FIRST_TIME_SURFACE_PENALTY = "medium"  # penalty tier for unknown surface

# --- Age-based development pattern constants ---
DEV_IDEAL_MAX_STEP = 2.0   # max improvement per year for "ideal" development
DEV_BIG_JUMP = 3.0         # improvement threshold that triggers bounce risk flag
DEV_UPGRADE_BONUS = 3.0    # raw_score bonus for strong development
DEV_DOWNGRADE_PENALTY = 2.0  # raw_score penalty for stalled/declining development

# --- Sheets-first cycle priority (higher = better cycle) ---
CYCLE_PRIORITY = {
    "PAIRED": 6,            # includes PAIRED_TOP (same projection_type)
    "REBOUND": 4,
    "IMPROVING": 3,
    "NEUTRAL": 2,
    "TAIL_OFF": 1,
    "NEW_TOP_BOUNCE": 0,    # risk flag, never treated as positive
}
TIE_WINDOW = 1.5            # proj_mid gap within which bias_score breaks ties


@dataclass
class FigureEntry:
    value: float
    surface: str = ""
    flags: List[str] = field(default_factory=list)


@dataclass
class BiasInput:
    e_pct: float = 25.0
    ep_pct: float = 25.0
    p_pct: float = 25.0
    s_pct: float = 25.0
    speed_favoring: bool = False

    def normalized_weights(self) -> Dict[str, float]:
        total = max(self.e_pct + self.ep_pct + self.p_pct + self.s_pct, 1.0)
        return {
            "E": self.e_pct / total,
            "EP": self.ep_pct / total,
            "P": self.p_pct / total,
            "S": self.s_pct / total,
        }


@dataclass
class HorseInput:
    name: str
    post: str
    style: str
    figures: List[FigureEntry]
    notes: Dict[str, str] = field(default_factory=dict)
    drf_overrides: Dict[str, str] = field(default_factory=dict)
    scratched: bool = False
    figure_source: str = "ragozin"  # "ragozin" (lower=better) or "brisnet" (higher=better)
    age: int = 0
    seasonal_bests: Dict[int, float] = field(default_factory=dict)  # {age: best_figure}


@dataclass
class HorseProjection:
    name: str
    post: str
    style: str
    projected_low: float
    projected_high: float
    confidence: float
    tags: List[str]
    raw_score: float
    bias_score: float
    summary: str
    projection_type: str = "NEUTRAL"
    new_top_setup: bool = False
    new_top_setup_type: str = ""
    new_top_confidence: int = 0
    new_top_explanation: str = ""
    bounce_risk: bool = False
    tossed: bool = False
    toss_reasons: List[str] = field(default_factory=list)
    dev_pattern: str = ""           # STRONG_DEV, STALLED, DECLINING, BIG_JUMP, ""
    dev_explanation: str = ""
    proj_mid: float = 0.0
    spread: float = 0.0
    cycle_priority: int = 2
    sheets_rank: int = 0            # 1-based position in sheets-first order
    tie_break_used: bool = False
    explain: List[tuple] = field(default_factory=list)


@dataclass
class AuditResult:
    """AUDIT section output per the Ragozin Sheets Edge permanent rules."""
    cycle_best: str = ""          # horse name ranked #1 by cycle-based scoring
    raw_best: str = ""            # horse name with best raw figure (last race)
    cycle_vs_raw_match: bool = False  # True if same horse
    bounce_candidates: List[str] = field(default_factory=list)  # names
    first_time_surface: List[Dict] = field(default_factory=list)  # [{name, penalty}]
    toss_list: List[Dict] = field(default_factory=list)  # [{name, reasons}]


class HandicappingEngine:
    """Encapsulates deterministic Sheets-first projections."""

    def __init__(self):
        pass

    def analyze_race(
        self,
        horses: List[HorseInput],
        bias: BiasInput,
        scratches: Optional[List[str]] = None,
    ) -> List[HorseProjection]:
        scratches = scratches or []
        results: List[HorseProjection] = []
        bias_weights = bias.normalized_weights()

        for horse in horses:
            if horse.scratched or horse.name in scratches:
                continue
            projection = self._project_horse(horse, bias_weights, bias.speed_favoring)
            # TOSS evaluation
            tossed, toss_reasons = self._evaluate_toss(horse, projection, bias)
            projection.tossed = tossed
            projection.toss_reasons = toss_reasons
            results.append(projection)

        # ==============================================================
        # Sheets-first ranking: cycle_priority → proj_mid → confidence
        # Bias/style only breaks ties within TIE_WINDOW.
        # ==============================================================
        results = self._sheets_first_sort(results)
        return results

    @staticmethod
    def _sheets_first_sort(results: List["HorseProjection"]) -> List["HorseProjection"]:
        """Sort by cycle_priority DESC, raw_score DESC (normalized), confidence DESC.

        Then apply tie-break pass: within TIE_WINDOW of proj_mid AND same
        cycle_priority, reorder by bias_score DESC.
        """
        if not results:
            return results

        # Primary sort: sheets-first (raw_score already normalizes direction)
        results.sort(key=lambda hp: (-hp.cycle_priority, -hp.raw_score, -hp.confidence))

        # Assign initial sheets_rank
        for i, hp in enumerate(results):
            hp.sheets_rank = i + 1

        # Tie-break pass: bubble-swap adjacent horses within tie window
        changed = True
        while changed:
            changed = False
            for i in range(len(results) - 1):
                a, b = results[i], results[i + 1]
                if (a.cycle_priority == b.cycle_priority
                        and abs(a.proj_mid - b.proj_mid) <= TIE_WINDOW
                        and b.bias_score > a.bias_score):
                    results[i], results[i + 1] = b, a
                    a.tie_break_used = True
                    b.tie_break_used = True
                    changed = True

        # Reassign final sheets_rank and build explain
        for i, hp in enumerate(results):
            hp.sheets_rank = i + 1
            hp.explain = [
                ("cycle_priority", hp.cycle_priority),
                ("proj_mid", round(hp.proj_mid, 1)),
                ("confidence", hp.confidence),
                ("tie_break_used", hp.tie_break_used),
                ("bias_score", round(hp.bias_score, 1)),
                ("sheets_rank", hp.sheets_rank),
                ("notes", list(hp.tags)),
            ]

        return results

    def analyze_race_with_audit(
        self,
        horses: List[HorseInput],
        bias: BiasInput,
        scratches: Optional[List[str]] = None,
    ) -> tuple:
        """Return ``(projections, audit)`` — full analysis + AUDIT section."""
        projections = self.analyze_race(horses, bias, scratches)
        audit = self._generate_audit(horses, projections)
        return projections, audit

    @staticmethod
    def _generate_audit(
        horses: List["HorseInput"],
        projections: List["HorseProjection"],
    ) -> "AuditResult":
        """Build AUDIT per permanent rules.

        1. Cycle-best vs raw-best comparison
        2. All bounce candidates
        3. First-time-surface horses with penalties
        4. Toss list with reasons
        """
        audit = AuditResult()
        if not projections:
            return audit

        # --- 1. Cycle-best vs raw-best ---
        # Cycle-best = #1 by bias_score (already sorted)
        audit.cycle_best = projections[0].name

        # Raw-best = best recent figure (regardless of cycle)
        # Find the corresponding horse input for each projection
        horse_map = {h.name: h for h in horses}
        best_raw_name = ""
        best_raw_val = None
        for p in projections:
            h = horse_map.get(p.name)
            if not h or not h.figures:
                continue
            recent = h.figures[0].value
            is_brisnet = h.figure_source == "brisnet"
            if best_raw_val is None:
                best_raw_val = recent
                best_raw_name = p.name
            elif is_brisnet and recent > best_raw_val:
                best_raw_val = recent
                best_raw_name = p.name
            elif not is_brisnet and recent < best_raw_val:
                best_raw_val = recent
                best_raw_name = p.name

        audit.raw_best = best_raw_name
        audit.cycle_vs_raw_match = (audit.cycle_best == audit.raw_best)

        # --- 2. Bounce candidates ---
        audit.bounce_candidates = [
            p.name for p in projections
            if p.bounce_risk or p.projection_type == "NEW_TOP_BOUNCE"
        ]

        # --- 3. First-time-surface horses ---
        for p in projections:
            if "SURFACE_UNKNOWN" in p.tags:
                audit.first_time_surface.append({
                    "name": p.name,
                    "penalty": FIRST_TIME_SURFACE_PENALTY,
                    "conf_adj": "-15%",
                })
            elif "SURFACE_ADAPT" in p.tags:
                audit.first_time_surface.append({
                    "name": p.name,
                    "penalty": "light",
                    "conf_adj": "-5%",
                })

        # --- 4. Toss list ---
        audit.toss_list = [
            {"name": p.name, "reasons": p.toss_reasons}
            for p in projections if p.tossed
        ]

        return audit

    # ------------------------------------------------------------------
    # Core projection
    # ------------------------------------------------------------------

    def _project_horse(
        self,
        horse: HorseInput,
        bias_weights: Dict[str, float],
        speed_favoring: bool,
    ) -> HorseProjection:
        is_brisnet = horse.figure_source == "brisnet"
        figures = [f.value for f in horse.figures if f.value > 0]
        fig_flags = horse.figures[0].flags if horse.figures else []
        tags: List[str] = []

        if not figures:
            figures = [70.0] if is_brisnet else [24.0]

        recent = figures[0]

        # ==============================================================
        # 1. Classify cycle pattern
        # ==============================================================
        projection_type, cycle_tags, cycle_meta = self._classify_cycle(
            figures, is_brisnet,
        )
        tags.extend(cycle_tags)

        # ==============================================================
        # 1b. New Top Setup detection (4yo, Ragozin only)
        # ==============================================================
        is_setup, setup_type, setup_conf, setup_expl = self._detect_new_top_setup(
            horse.figures, horse.age, is_brisnet, projection_type, tags,
        )
        bounce_risk = self._detect_bounce_risk(figures, is_brisnet)
        if is_setup:
            tags.append(f"NEW_TOP_{setup_type}")
        if bounce_risk:
            tags.append("BOUNCE_RISK")

        # Additional context tags (these don't change projection_type)
        if any(flag.upper() in {"PACE", "T"} for flag in fig_flags):
            tags.append("FORGIVE")

        lasix = horse.notes.get("lasix", "").lower()
        if lasix:
            tags.append(f"LASIX_{lasix.upper()}")

        surface_switch = horse.notes.get("surface_switch", "").lower() in {"new", "first"}
        if surface_switch:
            tags.append("SURFACE_SWITCH")

        # ==============================================================
        # 2. Compute projection range from cycle type
        # ==============================================================
        if projection_type == "PAIRED":
            # Paired top / plateau: expect last figure ±0-2
            # Ragozin (lower=better): low=recent, high=recent+2
            # BRISNET (higher=better): low=recent-2, high=recent
            if is_brisnet:
                projected_low = recent - 2
                projected_high = recent
            else:
                projected_low = recent
                projected_high = recent + 2
            mid = recent
            confidence = 0.70

        elif projection_type == "REBOUND":
            # Top → worse → recovering. High priority — resilience.
            # Project near recovery figure, slight upside toward pre-bounce.
            pre_bounce = cycle_meta.get("pre_bounce", recent)
            matched = cycle_meta.get("matched_top", False)
            if matched:
                # Matched/beat the pre-bounce top — project repeat
                if is_brisnet:
                    projected_low = recent - 1
                    projected_high = recent + 1
                else:
                    projected_low = recent - 1
                    projected_high = recent + 1
                mid = recent
                confidence = 0.70
            else:
                # Still recovering — project between recovery and pre-bounce
                if is_brisnet:
                    projected_low = recent - 1
                    projected_high = recent + 2
                    mid = recent + 0.5
                else:
                    projected_low = recent - 2
                    projected_high = recent + 1
                    mid = recent - 0.5
                confidence = 0.65

        elif projection_type == "NEW_TOP_BOUNCE":
            # Big new top (>= BIG_NEW_TOP) → expect regression
            # Ragozin: bounce = figure goes UP (worse)
            # BRISNET: bounce = figure goes DOWN (worse)
            if is_brisnet:
                projected_low = recent - BOUNCE_HIGH
                projected_high = recent - BOUNCE_LOW
                mid = recent - (BOUNCE_LOW + BOUNCE_HIGH) / 2
            else:
                projected_low = recent + BOUNCE_LOW
                projected_high = recent + BOUNCE_HIGH
                mid = recent + (BOUNCE_LOW + BOUNCE_HIGH) / 2
            confidence = 0.45

        elif projection_type == "IMPROVING":
            # Consecutive improvements → project dampened continuation
            avg_step = cycle_meta.get("avg_step", 1.0)
            proj_step = avg_step * 0.7  # dampen to ~70%
            if is_brisnet:
                projected_low = recent
                projected_high = recent + proj_step * 2
                mid = recent + proj_step
            else:
                projected_low = recent - proj_step * 2
                projected_high = recent
                mid = recent - proj_step
            confidence = 0.55

        elif projection_type == "TAIL_OFF":
            # Deteriorating form → project continued decline (dampened)
            avg_decline = cycle_meta.get("avg_decline", 1.0)
            proj_decline = avg_decline * 0.5  # dampen to ~50%
            if is_brisnet:
                projected_low = recent - proj_decline * 2
                projected_high = recent
                mid = recent - proj_decline
            else:
                projected_low = recent
                projected_high = recent + proj_decline * 2
                mid = recent + proj_decline
            confidence = 0.40

        else:  # NEUTRAL
            mid = recent
            spread = 2.5
            projected_low = mid - spread / 2
            projected_high = mid + spread / 2
            confidence = 0.50

        # ==============================================================
        # 3. Modifiers (secondary to cycle type — never override figures)
        # ==============================================================
        if surface_switch:
            confidence -= 0.10
            # Widen range
            spread_now = projected_high - projected_low
            if spread_now < 3.0:
                extra = (3.0 - spread_now) / 2
                projected_low -= extra
                projected_high += extra

        # --- Surface experience modifier ---
        surf_exp = self._surface_experience(horse.figures)
        if surf_exp == "NO_HISTORY":
            # Single-surface horse, never raced on today's surface — big unknown
            tags.append("SURFACE_UNKNOWN")
            confidence -= 0.15
            spread_now = projected_high - projected_low
            if spread_now < 4.0:
                extra = (4.0 - spread_now) / 2
                projected_low -= extra
                projected_high += extra
        elif surf_exp == "SURFACE_ADAPTABLE":
            # Proven on 2+ surfaces but not today's — lighter risk
            tags.append("SURFACE_ADAPT")
            confidence -= 0.05
            spread_now = projected_high - projected_low
            if spread_now < 3.0:
                extra = (3.0 - spread_now) / 2
                projected_low -= extra
                projected_high += extra
        elif surf_exp == "MULTI_PROVEN":
            # Consistent figures across 2+ surfaces including today's — reliable
            tags.append("MULTI_SURFACE")
            confidence += 0.05
            spread_now = projected_high - projected_low
            if spread_now > 2.0:
                shrink = min(0.5, (spread_now - 2.0) / 2)
                projected_low += shrink
                projected_high -= shrink

        if lasix == "first":
            # Slight improvement expected
            mid += -0.3 if not is_brisnet else 0.3
        elif lasix == "second":
            mid += -0.2 if not is_brisnet else 0.2

        confidence = max(0.25, min(confidence, 0.90))
        projected_low = max(0.0, projected_low)

        # ==============================================================
        # 4. Age-based development pattern
        # ==============================================================
        dev_pattern, dev_tags, dev_adj, dev_expl = self._classify_development(
            horse.seasonal_bests, horse.age, is_brisnet,
        )
        tags.extend(dev_tags)

        # ==============================================================
        # 5. Scoring (higher raw_score = better horse, always)
        # ==============================================================
        raw_score = mid if is_brisnet else (100.0 - mid)
        raw_score += dev_adj  # development upgrade/downgrade

        style_key = self._base_style(horse.style)
        bias_bonus = bias_weights.get(style_key, 0.0) * 10.0
        if speed_favoring and style_key in {"E", "EP"}:
            bias_bonus += 1.5
        if speed_favoring and style_key in {"P", "S"}:
            bias_bonus -= 1.0

        bias_score = raw_score + bias_bonus

        # ==============================================================
        # 5. Summary text
        # ==============================================================
        type_labels = {
            "PAIRED": "PAIRED (expect same figure)",
            "REBOUND": "REBOUND (recovery, upside)",
            "IMPROVING": "IMPROVING (forward move)",
            "NEW_TOP_BOUNCE": "NEW_TOP_BOUNCE (regression likely)",
            "TAIL_OFF": "TAIL_OFF (declining form)",
            "NEUTRAL": "NEUTRAL",
        }
        summary_parts = [
            f"{type_labels.get(projection_type, projection_type)}: "
            f"{projected_low:.1f}\u2013{projected_high:.1f}",
            f"Conf {confidence:.0%}",
        ]
        if tags:
            summary_parts.append(", ".join(tags))
        if is_setup:
            stars = "\u2b50\u2b50" if setup_conf >= 75 else "\u2b50"
            summary_parts.append(f"{stars} NEW_TOP({setup_type} {setup_conf}%)")
        if bounce_risk and not is_setup:
            summary_parts.append("\u26a0\ufe0f BOUNCE_RISK")
        if dev_pattern:
            summary_parts.append(f"DEV:{dev_pattern}")
        summary = " | ".join(summary_parts)

        return HorseProjection(
            name=horse.name,
            post=horse.post,
            style=horse.style,
            projected_low=projected_low,
            projected_high=projected_high,
            confidence=confidence,
            tags=tags,
            raw_score=raw_score,
            bias_score=bias_score,
            summary=summary,
            projection_type=projection_type,
            new_top_setup=is_setup,
            new_top_setup_type=setup_type,
            new_top_confidence=setup_conf,
            new_top_explanation=setup_expl,
            bounce_risk=bounce_risk,
            dev_pattern=dev_pattern,
            dev_explanation=dev_expl,
            proj_mid=(projected_low + projected_high) / 2,
            spread=projected_high - projected_low,
            cycle_priority=CYCLE_PRIORITY.get(projection_type, 2),
        )

    # ------------------------------------------------------------------
    # Cycle classifier
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_cycle(
        figures: List[float],
        is_brisnet: bool,
    ) -> tuple:
        """Return ``(projection_type, tags, meta_dict)``.

        Priority order (per Ragozin Sheets Edge permanent rules):
          1. NEW_TOP_BOUNCE – big new top (>= BIG_NEW_TOP better than all prior)
          2. PAIRED         – last two within PAIRED_TOL, at/near best
          3. REBOUND        – top → worse → improving back within REBOUND_WINDOW
          4. IMPROVING      – 2+ consecutive improvements, last move < BIG_NEW_TOP
          5. PAIRED         – regular paired or plateau (stable, not at top)
          6. TAIL_OFF       – 2+ consecutive deteriorations
          7. NEUTRAL        – fallback
        """
        tags: List[str] = []
        meta: Dict = {}

        if len(figures) < 2:
            return ("NEUTRAL", tags, meta)

        recent = figures[0]
        prev = figures[1]

        # --- helper: "better" comparison ---
        # For Ragozin lower=better; for BRISNET higher=better.
        if is_brisnet:
            prior_best = max(figures[1:])
            improvement_over_best = recent - prior_best  # positive = better
        else:
            prior_best = min(figures[1:])
            improvement_over_best = prior_best - recent  # positive = better

        # --- 1. Big new top (>= BIG_NEW_TOP improvement) → NEW_TOP_BOUNCE ---
        big_new_top = improvement_over_best >= BIG_NEW_TOP
        if big_new_top:
            tags.append("NEW_TOP_BOUNCE")
            return ("NEW_TOP_BOUNCE", tags, meta)

        # --- 2. Paired at top detection ---
        paired = abs(recent - prev) <= PAIRED_TOL

        if paired:
            if is_brisnet:
                best = max(figures)
                paired_top = (best - recent) <= 2.0 and (best - prev) <= 2.0
            else:
                best = min(figures)
                paired_top = (recent - best) <= 2.0 and (prev - best) <= 2.0

            if paired_top:
                tags.append("PAIRED_TOP")
                return ("PAIRED", tags, meta)

        # --- 3. REBOUND: top → worse → improving back ---
        # No age gating — fires for any horse with the pattern.
        if len(figures) >= 3:
            # Look within REBOUND_WINDOW recent starts for the pattern
            window = min(len(figures), REBOUND_WINDOW + 1)
            for start in range(1, window - 1):
                pre_bounce_fig = figures[start + 1]
                bounced_fig = figures[start]
                recovery_fig = figures[0]

                # Was the pre-bounce figure near-top?
                if is_brisnet:
                    all_best = max(figures)
                    was_near_top = (all_best - pre_bounce_fig) <= 2.0
                    did_bounce = pre_bounce_fig - bounced_fig >= 3.0
                    recovered = recovery_fig - bounced_fig >= 2.0
                    near_pre = recovery_fig >= pre_bounce_fig - 3.0
                else:
                    all_best = min(figures)
                    was_near_top = (pre_bounce_fig - all_best) <= 2.0
                    did_bounce = bounced_fig - pre_bounce_fig >= 3.0
                    recovered = bounced_fig - recovery_fig >= 2.0
                    near_pre = recovery_fig <= pre_bounce_fig + 3.0

                if was_near_top and did_bounce and recovered and near_pre:
                    tags.append("REBOUND")
                    meta["pre_bounce"] = pre_bounce_fig
                    meta["bounced"] = bounced_fig
                    meta["recovery"] = recovery_fig
                    # Matched or beat pre-bounce top?
                    if is_brisnet:
                        meta["matched_top"] = recovery_fig >= pre_bounce_fig
                    else:
                        meta["matched_top"] = recovery_fig <= pre_bounce_fig
                    return ("REBOUND", tags, meta)

        # --- 4. Consecutive improvements (2+) → IMPROVING ---
        consec_improve = 0
        for i in range(len(figures) - 1):
            if is_brisnet:
                improving = figures[i] > figures[i + 1]
            else:
                improving = figures[i] < figures[i + 1]
            if improving:
                consec_improve += 1
            else:
                break

        if consec_improve >= 2:
            steps = [abs(figures[i] - figures[i + 1]) for i in range(consec_improve)]
            avg_step = sum(steps) / len(steps) if steps else 1.0
            tags.append("IMPROVING")
            meta["avg_step"] = avg_step
            meta["consec"] = consec_improve
            return ("IMPROVING", tags, meta)

        # --- 5. Regular paired / plateau → PAIRED ---
        plateau_slice = figures[:4]
        plateau = (
            len(plateau_slice) >= 3
            and (max(plateau_slice) - min(plateau_slice)) <= 2.0
        )

        if paired:
            tags.append("PAIRED")
            return ("PAIRED", tags, meta)

        if plateau:
            tags.append("PLATEAU")
            return ("PAIRED", tags, meta)

        # --- 6. TAIL_OFF: 2+ consecutive deteriorations ---
        consec_decline = 0
        for i in range(len(figures) - 1):
            if is_brisnet:
                declining = figures[i] < figures[i + 1]
            else:
                declining = figures[i] > figures[i + 1]
            if declining:
                consec_decline += 1
            else:
                break

        if consec_decline >= 2:
            steps = [abs(figures[i] - figures[i + 1]) for i in range(consec_decline)]
            avg_step = sum(steps) / len(steps) if steps else 1.0
            tags.append("TAIL_OFF")
            meta["avg_decline"] = avg_step
            meta["consec_decline"] = consec_decline
            return ("TAIL_OFF", tags, meta)

        # --- 7. Moderate new top (2-4 better, not big enough for bounce) ---
        if improvement_over_best >= 2.0:
            tags.append("NEW_TOP")
            return ("NEUTRAL", tags, meta)

        # --- 8. No clear pattern ---
        return ("NEUTRAL", tags, meta)

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Surface experience classifier
    # ------------------------------------------------------------------

    @staticmethod
    def _surface_experience(figures: List["FigureEntry"]) -> str:
        """Classify surface experience from figure entries.

        Returns:
          ``"NO_HISTORY"``    – today's surface never seen in prior races
          ``"MULTI_PROVEN"``  – ran well on 2+ distinct surfaces (range ≤ 3)
          ``""``              – single-surface history or insufficient data
        """
        if len(figures) < 2:
            return ""

        current = (figures[0].surface or "").upper().strip()
        if not current:
            return ""  # no surface info available

        prior_surfaces: Dict[str, List[float]] = {}
        for f in figures[1:]:
            s = (f.surface or "").upper().strip()
            if s and f.value > 0:
                prior_surfaces.setdefault(s, []).append(f.value)

        if not prior_surfaces:
            return ""

        multi = len(prior_surfaces) >= 2
        bests = [min(vals) for vals in prior_surfaces.values()]
        consistent = (max(bests) - min(bests) <= 3.0) if multi else False

        # Check if horse has NEVER raced on today's surface
        if current not in prior_surfaces:
            if multi and consistent:
                # Proven on 2+ surfaces → adaptable, lighter risk
                return "SURFACE_ADAPTABLE"
            return "NO_HISTORY"

        # Today's surface IS in prior history
        if multi and consistent:
            return "MULTI_PROVEN"

        return ""

    # ------------------------------------------------------------------
    # Bounce risk detector
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_bounce_risk(figures: List[float], is_brisnet: bool) -> bool:
        """True if last race was a BIG_NEW_TOP (>= BIG_NEW_TOP better than all prior)."""
        if len(figures) < 2:
            return False
        if is_brisnet:
            prior_best = max(figures[1:])
            return (figures[0] - prior_best) >= BIG_NEW_TOP
        else:
            prior_best = min(figures[1:])
            return (prior_best - figures[0]) >= BIG_NEW_TOP

    # ------------------------------------------------------------------
    # New Top Setup detector (4yo, Ragozin only)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_new_top_setup(
        figures: List["FigureEntry"],
        age: int,
        is_brisnet: bool,
        cycle_type: str,
        cycle_tags: List[str],
    ) -> tuple:
        """Detect New Top Setup patterns for 4-year-olds.

        Returns ``(is_setup, setup_type, confidence, explanation)``.
        Only fires for Ragozin data (lower = better) when age == 4.
        """
        _NONE = (False, "", 0, "")

        # Guard clauses
        if is_brisnet:
            return _NONE
        if age != 4:
            return _NONE

        vals = [f.value for f in figures if f.value > 0]
        if len(vals) < 2:
            return _NONE

        recent = vals[0]
        prior_best = min(vals[1:])  # lower = better for Ragozin
        improvement_over_best = prior_best - recent  # positive = better

        # Exclusion: already BIG_NEW_TOP — bounce_risk handles it
        if improvement_over_best >= BIG_NEW_TOP:
            return _NONE

        # Exclusion: PAIRED_TOP — stable repeat, not a setup
        if "PAIRED_TOP" in cycle_tags:
            return _NONE

        fig_str = "\u2192".join(str(int(v)) for v in vals[:5])

        # --- Pattern 1: REBOUND ---
        # Top -> bounce -> meaningful recovery
        if len(vals) >= 3:
            pre_bounce = vals[2]
            bounced = vals[1]
            # Was pre_bounce a good figure? (within 2 of overall best)
            overall_best = min(vals)
            was_top = (pre_bounce - overall_best) <= 2 if len(vals) > 2 else True
            # Bounce: went at least 3 worse
            did_bounce = bounced > pre_bounce + 3
            # Recovery: improved at least 2 from bounce
            recovered = recent < bounced - 2
            # Still within range of pre-bounce best
            near_top = recent <= pre_bounce + 3
            if was_top and did_bounce and recovered and near_top:
                conf = 70
                if recent <= pre_bounce:
                    conf = 80  # matched or beat pre-bounce top
                return (True, "REBOUND", conf,
                        f"Bounce recovery: {fig_str}")

        # --- Pattern 2: THIRD_START_BACK ---
        # Layoff proxy: total races <= 3 AND 2+ consecutive improvements
        total = len(vals)
        if total <= 3 and total >= 2:
            consec = 0
            for i in range(total - 1):
                if vals[i] < vals[i + 1]:  # lower = better
                    consec += 1
                else:
                    break
            if consec >= min(2, total - 1):
                steps = [vals[i + 1] - vals[i] for i in range(consec)]
                avg_step = sum(steps) / len(steps) if steps else 1.0
                conf = 65
                if avg_step >= 3:
                    conf = 75
                return (True, "THIRD_START_BACK", conf,
                        f"{total} starts, improving: {fig_str}")

        # --- Pattern 3: HIDDEN_FORM ---
        # Flat figs (range <= 2 over 3+) + trouble/pace excuse flags
        if len(vals) >= 3:
            check_slice = vals[:4] if len(vals) >= 4 else vals
            fig_range = max(check_slice) - min(check_slice)
            if fig_range <= 2.0:
                excuse_flags = {"PACE", "WIDE", "BOUNCE", "GROUND_SAVE"}
                excuse_count = 0
                for f in figures[:3]:
                    if any(fl.upper() in excuse_flags for fl in f.flags):
                        excuse_count += 1
                if excuse_count >= 1:
                    conf = 60
                    if excuse_count >= 2:
                        conf = 70
                    return (True, "HIDDEN_FORM", conf,
                            f"Flat figs (range {fig_range:.0f}) with excuses: {fig_str}")

        # --- Pattern 4: UNLOCK ---
        # Current surface matches a prior surface where horse ran notably better
        if len(figures) >= 2:
            cur_surf = figures[0].surface.upper() if figures[0].surface else ""
            if cur_surf:
                same_figs = [f.value for f in figures[1:]
                             if f.surface and f.surface.upper() == cur_surf and f.value > 0]
                diff_figs = [f.value for f in figures[1:]
                             if f.surface and f.surface.upper() != cur_surf and f.value > 0]
                if same_figs and diff_figs:
                    best_same = min(same_figs)
                    best_diff = min(diff_figs)
                    gap = best_diff - best_same
                    # Horse is notably better on current surface
                    if gap >= 3:
                        last_surf = figures[1].surface.upper() if figures[1].surface else ""
                        if last_surf and last_surf != cur_surf:
                            conf = 65
                            if gap >= 5:
                                conf = 75
                            return (True, "UNLOCK", conf,
                                    f"Back to {cur_surf}: best {best_same:.0f} "
                                    f"vs {best_diff:.0f} off-surface")

        # --- Pattern 5: NEAR_TOP_POP ---
        # Hovering 1-2 of prior best, not paired, not declining
        if len(vals) >= 2:
            gap_to_best = recent - prior_best  # Ragozin: positive = worse
            if 0 < gap_to_best <= 2:
                not_declining = recent <= vals[1]  # same or better than prev
                not_paired = abs(recent - vals[1]) > 1.0
                if not_declining and not_paired:
                    conf = 60
                    if gap_to_best <= 1:
                        conf = 70
                    return (True, "NEAR_TOP_POP", conf,
                            f"Within {gap_to_best:.0f} of best ({prior_best:.0f}): {fig_str}")

        return _NONE

    # ------------------------------------------------------------------
    # Age-based development pattern classifier
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_development(
        seasonal_bests: Dict[int, float],
        age: int,
        is_brisnet: bool,
    ) -> tuple:
        """Evaluate year-over-year development from seasonal best figures.

        Core philosophy: horses should improve with age. Small, incremental
        forward moves (1-2 points/year) = elite pattern. Stalled or declining
        year-over-year = weak. Big jumps (>3-4 points) = bounce risk.

        Args:
            seasonal_bests: {age: best_figure} e.g. {2: 12.0, 3: 10.5, 4: 9.0}
            age: current age
            is_brisnet: True if higher=better

        Returns:
            (pattern, tags, score_adj, explanation) where pattern is one of:
            STRONG_DEV, STALLED, DECLINING, BIG_JUMP, or ""
        """
        _NONE = ("", [], 0.0, "")

        if not seasonal_bests or age < 3:
            return _NONE

        # Build sorted sequence of (age, best) pairs
        ages_sorted = sorted(seasonal_bests.keys())
        if len(ages_sorted) < 2:
            return _NONE

        # Calculate year-over-year changes
        transitions = []
        for i in range(len(ages_sorted) - 1):
            a1, a2 = ages_sorted[i], ages_sorted[i + 1]
            f1, f2 = seasonal_bests[a1], seasonal_bests[a2]
            if is_brisnet:
                delta = f2 - f1  # positive = improved
            else:
                delta = f1 - f2  # positive = improved (lower = better)
            transitions.append((a1, a2, delta))

        if not transitions:
            return _NONE

        tags = []
        all_improved = all(d > 0 for _, _, d in transitions)
        all_stalled = all(abs(d) < 0.5 for _, _, d in transitions)
        any_declined = any(d < -0.5 for _, _, d in transitions)
        any_big_jump = any(d > DEV_BIG_JUMP for _, _, d in transitions)
        all_controlled = all(0 < d <= DEV_IDEAL_MAX_STEP for _, _, d in transitions)

        # Format figures for explanation
        fig_parts = [f"{a}yo:{seasonal_bests[a]:.0f}" for a in ages_sorted]
        fig_str = " -> ".join(fig_parts)

        # --- STRONG_DEV: all years improved, each step <= 2 points ---
        if all_improved and all_controlled and len(transitions) >= 1:
            tags.append("STRONG_DEV")
            score_adj = DEV_UPGRADE_BONUS
            if len(transitions) >= 2:
                score_adj = DEV_UPGRADE_BONUS + 1.0  # extra for 3+ seasons
            return ("STRONG_DEV", tags, score_adj,
                    f"Ideal development: {fig_str}")

        # --- BIG_JUMP: improved but too much at once (bounce risk) ---
        if any_big_jump:
            big = [(a1, a2, d) for a1, a2, d in transitions if d > DEV_BIG_JUMP]
            tags.append("DEV_BIG_JUMP")
            jump_desc = ", ".join(f"{a1}yo->{a2}yo: {d:.1f}pts" for a1, a2, d in big)
            return ("BIG_JUMP", tags, -1.0,
                    f"Big jump risk ({jump_desc}): {fig_str}")

        # --- IMPROVED but large steps (>2, <=3): good but not ideal ---
        if all_improved and not all_controlled:
            tags.append("DEV_IMPROVED")
            return ("STRONG_DEV", tags, DEV_UPGRADE_BONUS * 0.5,
                    f"Improving but large steps: {fig_str}")

        # --- STALLED: no meaningful change year-over-year ---
        if all_stalled:
            tags.append("DEV_STALLED")
            return ("STALLED", tags, -DEV_DOWNGRADE_PENALTY,
                    f"No forward development: {fig_str}")

        # --- DECLINING: got worse year-over-year ---
        if any_declined and not all_improved:
            declined = [(a1, a2, d) for a1, a2, d in transitions if d < -0.5]
            tags.append("DEV_DECLINING")
            penalty = DEV_DOWNGRADE_PENALTY
            if all(d < -0.5 for _, _, d in transitions):
                penalty = DEV_DOWNGRADE_PENALTY + 1.0  # every year worse
            return ("DECLINING", tags, -penalty,
                    f"Year-over-year decline: {fig_str}")

        return _NONE

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # TOSS module
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_toss(
        horse: "HorseInput",
        projection: "HorseProjection",
        bias: "BiasInput",
    ) -> tuple:
        """Evaluate TOSS criteria.  Toss if ANY TWO factors are true.

        Returns ``(tossed: bool, reasons: List[str])``.

        Factors:
          (a) TAIL_OFF or BOUNCE candidate (NEW_TOP_BOUNCE / BOUNCE_RISK)
          (b) Poor condition fit — surface unknown/new with no pattern excuse
          (c) Pace mismatch — running style doesn't fit bias AND not best cycle
          (d) Repeated failures at today's setup with no cycle improvement
        """
        reasons: List[str] = []
        tags = projection.tags

        # (a) tail-off or bounce candidate
        if projection.projection_type in ("TAIL_OFF", "NEW_TOP_BOUNCE"):
            reasons.append(f"(a) {projection.projection_type}: deteriorating or bounce candidate")
        elif projection.bounce_risk:
            reasons.append("(a) BOUNCE_RISK: big new top, regression expected")

        # (b) poor condition fit — surface unknown with no excuse
        has_surface_excuse = (
            projection.new_top_setup
            or "REBOUND" in tags
            or "IMPROVING" in tags
        )
        if "SURFACE_UNKNOWN" in tags and not has_surface_excuse:
            reasons.append("(b) SURFACE_UNKNOWN: no prior races on this surface, no pattern excuse")
        elif "SURFACE_SWITCH" in tags and "SURFACE_UNKNOWN" in tags:
            reasons.append("(b) SURFACE_SWITCH + UNKNOWN: poor condition fit")

        # (c) pace mismatch — style doesn't fit bias profile
        style = horse.style.upper() if horse.style else "P"
        is_speed = style.startswith("E")
        is_closer = style.startswith("S") or style.startswith("P")

        speed_bias = bias.e_pct + bias.ep_pct
        close_bias = bias.p_pct + bias.s_pct

        best_cycle = projection.projection_type in ("PAIRED", "REBOUND")

        pace_mismatch = False
        if bias.speed_favoring and is_closer and not best_cycle:
            pace_mismatch = True
            reasons.append("(c) Pace mismatch: closer in speed-favoring profile, not best cycle")
        elif not bias.speed_favoring and speed_bias < 30 and is_speed and not best_cycle:
            pace_mismatch = True
            reasons.append("(c) Pace mismatch: speed horse in closer-favoring profile, not best cycle")

        # (d) repeated failures at today's setup
        # Check notes for failure indicators
        repeat_fail = horse.notes.get("repeat_fail", "").lower() in {"yes", "true", "1"}
        improving_cycle = projection.projection_type in ("IMPROVING", "REBOUND")
        if repeat_fail and not improving_cycle:
            reasons.append("(d) Repeated failures at this setup with no cycle improvement")

        tossed = len(reasons) >= 2
        return (tossed, reasons)

    # ------------------------------------------------------------------

    @staticmethod
    def _base_style(style: str) -> str:
        if not style:
            return "P"
        style = style.upper()
        if style.startswith("EP"):
            return "EP"
        if style.startswith("E"):
            return "E"
        if style.startswith("S"):
            return "S"
        return "P"
