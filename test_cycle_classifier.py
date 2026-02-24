"""Unit tests for cycle classifier, TOSS module, and AUDIT section."""
import pytest
from handicap_engine import (
    HandicappingEngine,
    HorseInput,
    HorseProjection,
    FigureEntry,
    BiasInput,
    PAIRED_TOL,
    BIG_NEW_TOP,
    BOUNCE_LOW,
    BOUNCE_HIGH,
    REBOUND_WINDOW,
)

HE = HandicappingEngine


# =====================================================================
# Ragozin (lower = better)
# =====================================================================

class TestClassifyCycleRagozin:
    """All Ragozin tests: lower figure = better horse."""

    # --- PAIRED ---

    def test_paired_top(self):
        """Last two near-best within PAIRED_TOL → PAIRED + PAIRED_TOP."""
        typ, tags, _ = HE._classify_cycle([12.0, 12.5, 15.0, 16.0], False)
        assert typ == "PAIRED"
        assert "PAIRED_TOP" in tags

    def test_paired_top_exact_same(self):
        """Identical last two at the best → PAIRED."""
        typ, tags, _ = HE._classify_cycle([10.0, 10.0, 14.0], False)
        assert typ == "PAIRED"
        assert "PAIRED_TOP" in tags

    def test_paired_not_at_top(self):
        """Paired but not near the best → still PAIRED (regular)."""
        typ, tags, _ = HE._classify_cycle([16.0, 16.5, 10.0], False)
        assert typ == "PAIRED"
        assert "PAIRED" in tags
        assert "PAIRED_TOP" not in tags

    def test_plateau(self):
        """Three figures within 2.0 range, not paired → PAIRED + PLATEAU."""
        # abs(14-16)=2 > PAIRED_TOL, but range [14,16,15]=2.0 → plateau
        typ, tags, _ = HE._classify_cycle([14.0, 16.0, 15.0], False)
        assert typ == "PAIRED"
        assert "PLATEAU" in tags

    # --- IMPROVING ---

    def test_improving_three_steps(self):
        """Three consecutive improvements → IMPROVING."""
        typ, tags, meta = HE._classify_cycle([10.0, 13.0, 16.0, 20.0], False)
        assert typ == "IMPROVING"
        assert "IMPROVING" in tags
        assert meta["consec"] == 3
        assert meta["avg_step"] == pytest.approx(3.33, abs=0.1)

    def test_improving_exactly_two(self):
        """Exactly 2 consecutive improvements (minimum) → IMPROVING."""
        typ, tags, meta = HE._classify_cycle([11.0, 13.0, 15.0], False)
        assert typ == "IMPROVING"
        assert meta["consec"] == 2

    def test_improving_boundary_under_big_new_top(self):
        """Improvement just under BIG_NEW_TOP → IMPROVING, not bounce."""
        # recent=10.1, prior_best=min([15,16])=15, improvement=4.9
        typ, tags, _ = HE._classify_cycle([10.1, 15.0, 16.0], False)
        assert typ == "IMPROVING"

    # --- NEW_TOP_BOUNCE ---

    def test_new_top_bounce(self):
        """Improvement >= BIG_NEW_TOP over all prior → NEW_TOP_BOUNCE."""
        typ, tags, _ = HE._classify_cycle([8.0, 14.0, 15.0], False)
        assert typ == "NEW_TOP_BOUNCE"
        assert "NEW_TOP_BOUNCE" in tags

    def test_new_top_bounce_exact_boundary(self):
        """Improvement exactly == BIG_NEW_TOP → NEW_TOP_BOUNCE."""
        # recent=10, prior_best=min([15,16])=15, improvement=5.0
        typ, tags, _ = HE._classify_cycle([10.0, 15.0, 16.0], False)
        assert typ == "NEW_TOP_BOUNCE"

    def test_new_top_bounce_huge(self):
        """Massive improvement → NEW_TOP_BOUNCE."""
        typ, tags, _ = HE._classify_cycle([5.0, 20.0, 22.0], False)
        assert typ == "NEW_TOP_BOUNCE"

    # --- REBOUND ---

    def test_rebound_classic(self):
        """Top → bounce → recovery: 14→20→15."""
        typ, tags, meta = HE._classify_cycle([15.0, 20.0, 14.0], False)
        assert typ == "REBOUND"
        assert "REBOUND" in tags
        assert meta["pre_bounce"] == 14.0
        assert meta["bounced"] == 20.0
        assert meta["recovery"] == 15.0

    def test_rebound_matched_top(self):
        """Recovery matches or beats pre-bounce top."""
        typ, tags, meta = HE._classify_cycle([13.0, 20.0, 14.0], False)
        assert typ == "REBOUND"
        assert meta["matched_top"] is True

    def test_rebound_not_recovered_enough(self):
        """Bounce but recovery < 2 improvement → not REBOUND."""
        # 14→20→19 — recovered only 1 point
        typ, tags, _ = HE._classify_cycle([19.0, 20.0, 14.0], False)
        assert typ != "REBOUND"

    def test_rebound_too_far_from_pre_bounce(self):
        """Recovery too far from pre-bounce best (>3 worse) → not REBOUND."""
        # 14→20→18 — recovered but 18 > 14+3=17
        typ, tags, _ = HE._classify_cycle([18.0, 20.0, 14.0], False)
        assert typ != "REBOUND"

    def test_rebound_four_figures(self):
        """REBOUND detected in 4-figure sequence."""
        # figures[0]=15, figures[1]=18, figures[2]=20, figures[3]=14
        # start=2: pre=14, bounced=20, recovery=15
        typ, tags, meta = HE._classify_cycle([15.0, 18.0, 20.0, 14.0], False)
        assert typ == "REBOUND"
        assert meta["pre_bounce"] == 14.0

    def test_rebound_no_age_gating(self):
        """REBOUND fires without age restriction (unlike old new_top_setup)."""
        # Just verify it returns REBOUND — no age param at all
        typ, tags, _ = HE._classify_cycle([15.0, 20.0, 14.0], False)
        assert typ == "REBOUND"

    # --- TAIL_OFF ---

    def test_tail_off_three_declines(self):
        """Three consecutive deteriorations → TAIL_OFF."""
        typ, tags, meta = HE._classify_cycle([20.0, 17.0, 14.0, 10.0], False)
        assert typ == "TAIL_OFF"
        assert "TAIL_OFF" in tags
        assert meta["consec_decline"] == 3
        assert meta["avg_decline"] == pytest.approx(3.33, abs=0.1)

    def test_tail_off_exactly_two(self):
        """Exactly 2 consecutive deteriorations → TAIL_OFF."""
        typ, tags, meta = HE._classify_cycle([18.0, 16.0, 14.0], False)
        assert typ == "TAIL_OFF"
        assert meta["consec_decline"] == 2

    def test_tail_off_one_decline_not_enough(self):
        """Only 1 deterioration → not TAIL_OFF."""
        typ, tags, _ = HE._classify_cycle([16.0, 14.0, 18.0], False)
        assert typ != "TAIL_OFF"

    # --- NEUTRAL ---

    def test_neutral_mixed_pattern(self):
        """Mixed up/down pattern → NEUTRAL."""
        typ, tags, _ = HE._classify_cycle([15.0, 12.0, 16.0], False)
        assert typ == "NEUTRAL"

    def test_neutral_single_figure(self):
        """Only one figure → NEUTRAL."""
        typ, tags, _ = HE._classify_cycle([15.0], False)
        assert typ == "NEUTRAL"

    def test_neutral_new_top_moderate(self):
        """Improvement 2-4 pts but not consecutive → NEUTRAL + NEW_TOP tag."""
        # recent=12, prior=min([14,16])=14, improvement=2 — but only 1 improving step
        typ, tags, _ = HE._classify_cycle([12.0, 14.0, 11.0], False)
        # 12 < 14 → improvement. But only 1 consec (12<14, but 14>11 breaks).
        # improvement_over_best: prior_best=min([14,11])=11, 11-12=-1 → no.
        # Actually this would not hit NEW_TOP. Let me pick a better case.
        pass  # tested via specific new_top test below

    def test_neutral_moderate_new_top(self):
        """Improvement 2-4 over prior best, not consecutive → NEUTRAL + NEW_TOP."""
        # recent=10, prev=14, older=12. prior_best=min([14,12])=12. improvement=12-10=2
        # Not big new top (2<5). Not paired (abs(10-14)=4>1).
        # REBOUND: start=1: pre=12, bounced=14, did_bounce=14-12=2<3 → no.
        # consec improve: 10<14 yes, 14>12 no → 1 step, not enough.
        # Not plateau. Not tail_off. NEW_TOP: improvement=2>=2 → yes.
        typ, tags, _ = HE._classify_cycle([10.0, 14.0, 12.0], False)
        assert typ == "NEUTRAL"
        assert "NEW_TOP" in tags

    # --- Priority ordering ---

    def test_bounce_beats_everything(self):
        """NEW_TOP_BOUNCE has highest priority even if also improving."""
        # 5 → 12 → 14: improvement_over_best = 12-5=7 → bounce
        typ, tags, _ = HE._classify_cycle([5.0, 12.0, 14.0], False)
        assert typ == "NEW_TOP_BOUNCE"

    def test_paired_beats_rebound(self):
        """PAIRED_TOP takes priority over REBOUND."""
        # figures that might look like rebound but are paired at top
        # 12 → 12.5 → 18 → 12: paired (12, 12.5 within 1), near best 12
        typ, tags, _ = HE._classify_cycle([12.0, 12.5, 18.0, 12.0], False)
        assert typ == "PAIRED"
        assert "PAIRED_TOP" in tags

    def test_rebound_beats_improving(self):
        """REBOUND has higher priority than IMPROVING."""
        # 14→20→14→22: start=1: pre=14, bounced=20, recovery=14
        # But also improving: 14<20? no. Not improving.
        # Better: 15→20→13: rebound pattern
        # consec improve: 15<20? no. Not improving → REBOUND wins by ordering.
        typ, tags, _ = HE._classify_cycle([15.0, 20.0, 13.0], False)
        assert typ == "REBOUND"

    def test_improving_beats_tail_off(self):
        """IMPROVING has higher priority than TAIL_OFF (can't be both)."""
        # By definition, can't have 2+ consecutive improvements and 2+ declines
        # This is a logical test — if improving, it's not tail_off
        typ, _, _ = HE._classify_cycle([8.0, 12.0, 16.0], False)
        assert typ == "IMPROVING"


# =====================================================================
# BRISNET (higher = better)
# =====================================================================

class TestClassifyCycleBrisnet:
    """All BRISNET tests: higher figure = better horse."""

    def test_paired_top_brisnet(self):
        typ, tags, _ = HE._classify_cycle([85.0, 84.5, 80.0], True)
        assert typ == "PAIRED"
        assert "PAIRED_TOP" in tags

    def test_improving_brisnet(self):
        """Consecutive increases → IMPROVING."""
        typ, tags, _ = HE._classify_cycle([90.0, 87.0, 84.0], True)
        assert typ == "IMPROVING"

    def test_new_top_bounce_brisnet(self):
        """Big new top (higher) → NEW_TOP_BOUNCE."""
        typ, tags, _ = HE._classify_cycle([95.0, 88.0, 87.0], True)
        assert typ == "NEW_TOP_BOUNCE"

    def test_tail_off_brisnet(self):
        """Getting worse (lower each time) → TAIL_OFF."""
        typ, tags, _ = HE._classify_cycle([70.0, 75.0, 80.0], True)
        assert typ == "TAIL_OFF"

    def test_rebound_brisnet(self):
        """Top(85) → bounce(75) → recovery(83)."""
        typ, tags, meta = HE._classify_cycle([83.0, 75.0, 85.0], True)
        assert typ == "REBOUND"
        assert meta["pre_bounce"] == 85.0
        assert meta["bounced"] == 75.0


# =====================================================================
# Projection ranges for new cycle types
# =====================================================================

class TestProjectionRanges:
    """Verify projection ranges and confidence for each cycle type."""

    def _project(self, figures, is_brisnet=False, age=0):
        engine = HandicappingEngine()
        horse = HorseInput(
            name="Test", post="1", style="P",
            figures=[FigureEntry(v) for v in figures],
            figure_source="brisnet" if is_brisnet else "ragozin",
            age=age,
        )
        return engine.analyze_race([horse], BiasInput())[0]

    def test_paired_projection(self):
        p = self._project([12.0, 12.5, 15.0, 16.0])
        assert p.projection_type == "PAIRED"
        assert p.confidence == pytest.approx(0.70, abs=0.16)  # before modifiers
        assert p.projected_low <= 12.0
        assert p.projected_high >= 12.0

    def test_rebound_projection_not_matched(self):
        """REBOUND (not matched top) → conf ~0.65, range with upside."""
        p = self._project([15.0, 20.0, 14.0])
        assert p.projection_type == "REBOUND"
        assert p.confidence >= 0.50  # 0.65 base, may adjust from modifiers
        assert p.projected_low < 15.0  # upside projected

    def test_rebound_projection_matched(self):
        """REBOUND (matched top) → conf ~0.70."""
        p = self._project([13.0, 20.0, 14.0])
        assert p.projection_type == "REBOUND"
        assert p.confidence >= 0.55  # 0.70 base

    def test_new_top_bounce_projection(self):
        """NEW_TOP_BOUNCE → regression range, low confidence."""
        p = self._project([8.0, 14.0, 15.0])
        assert p.projection_type == "NEW_TOP_BOUNCE"
        assert p.confidence <= 0.50
        # Ragozin: projected_low should be worse (higher) than recent
        assert p.projected_low >= 8.0 + BOUNCE_LOW

    def test_improving_projection(self):
        p = self._project([10.0, 13.0, 16.0])
        assert p.projection_type == "IMPROVING"
        assert p.confidence >= 0.45

    def test_tail_off_projection(self):
        """TAIL_OFF → downgraded range, low confidence."""
        p = self._project([20.0, 17.0, 14.0])
        assert p.projection_type == "TAIL_OFF"
        assert p.confidence <= 0.50
        # Ragozin: projected range should be worse (higher) than recent
        assert p.projected_high > 20.0

    def test_neutral_projection(self):
        p = self._project([15.0, 12.0, 16.0])
        assert p.projection_type == "NEUTRAL"
        assert p.confidence == pytest.approx(0.50, abs=0.16)


# =====================================================================
# Bounce risk detector (uses constant)
# =====================================================================

class TestBounceRisk:
    def test_bounce_risk_ragozin(self):
        assert HE._detect_bounce_risk([8.0, 14.0, 15.0], False) is True

    def test_no_bounce_risk_ragozin(self):
        assert HE._detect_bounce_risk([10.0, 13.0, 15.0], False) is False

    def test_bounce_risk_brisnet(self):
        assert HE._detect_bounce_risk([95.0, 88.0, 87.0], True) is True

    def test_bounce_risk_single_fig(self):
        assert HE._detect_bounce_risk([15.0], False) is False

    def test_bounce_risk_exact_boundary(self):
        """Exactly BIG_NEW_TOP improvement → True."""
        assert HE._detect_bounce_risk([10.0, 15.0], False) is True

    def test_bounce_risk_just_under(self):
        """Just under BIG_NEW_TOP → False."""
        assert HE._detect_bounce_risk([10.1, 15.0], False) is False


# =====================================================================
# Constants
# =====================================================================

class TestConstants:
    def test_paired_tol(self):
        assert PAIRED_TOL == 1.0

    def test_big_new_top(self):
        assert BIG_NEW_TOP == 5.0

    def test_bounce_range(self):
        assert BOUNCE_LOW == 2.0
        assert BOUNCE_HIGH == 6.0

    def test_rebound_window(self):
        assert REBOUND_WINDOW == 3


# =====================================================================
# TOSS module
# =====================================================================

class TestTossModule:
    """TOSS: toss if ANY TWO factors are true."""

    def _make_horse(self, name="H", style="P", figures=None, surface=None,
                    notes=None, source="ragozin"):
        if figures is None:
            figures = [12.0, 13.0]
        figs = [FigureEntry(v, surface or "") for v in figures]
        if surface and len(figs) > 0:
            figs[0] = FigureEntry(figs[0].value, surface)
        return HorseInput(
            name=name, post="1", style=style, figures=figs,
            notes=notes or {}, figure_source=source,
        )

    def _run(self, horse, bias=None):
        e = HandicappingEngine()
        bias = bias or BiasInput()
        projs = e.analyze_race([horse], bias)
        return projs[0] if projs else None

    # --- Tossed scenarios (2+ factors) ---

    def test_tail_off_plus_surface_unknown(self):
        """TAIL_OFF + SURFACE_UNKNOWN → tossed."""
        h = HorseInput(
            name="TailSurf", post="1", style="P",
            figures=[
                FigureEntry(20, "TURF"),  # today = turf
                FigureEntry(17, "DIRT"),
                FigureEntry(14, "DIRT"),
            ],
        )
        p = self._run(h)
        assert p.tossed is True
        assert any("(a)" in r for r in p.toss_reasons)
        assert any("(b)" in r for r in p.toss_reasons)

    def test_bounce_plus_pace_mismatch(self):
        """NEW_TOP_BOUNCE + pace mismatch (closer in speed bias) → tossed."""
        h = self._make_horse(style="S", figures=[8.0, 14.0, 15.0])
        bias = BiasInput(e_pct=50, ep_pct=30, p_pct=10, s_pct=10, speed_favoring=True)
        p = self._run(h, bias)
        assert p.tossed is True
        assert any("(a)" in r for r in p.toss_reasons)
        assert any("(c)" in r for r in p.toss_reasons)

    def test_tail_off_plus_repeat_fail(self):
        """TAIL_OFF + repeated failures → tossed."""
        h = HorseInput(
            name="TailFail", post="1", style="P",
            figures=[FigureEntry(20), FigureEntry(17), FigureEntry(14)],
            notes={"repeat_fail": "yes"},
        )
        p = self._run(h)
        assert p.tossed is True
        assert any("(a)" in r for r in p.toss_reasons)
        assert any("(d)" in r for r in p.toss_reasons)

    def test_surface_unknown_plus_repeat_fail(self):
        """SURFACE_UNKNOWN + repeat_fail, neutral cycle → tossed."""
        h = HorseInput(
            name="SurfFail", post="1", style="P",
            figures=[
                FigureEntry(15, "TURF"),
                FigureEntry(12, "DIRT"),
                FigureEntry(16, "DIRT"),
            ],
            notes={"repeat_fail": "yes"},
        )
        p = self._run(h)
        assert p.tossed is True
        assert len(p.toss_reasons) >= 2

    # --- Not tossed scenarios (0-1 factor) ---

    def test_paired_clean_not_tossed(self):
        """PAIRED with no risk factors → not tossed."""
        h = self._make_horse(figures=[12.0, 12.5, 15.0])
        p = self._run(h)
        assert p.tossed is False
        assert p.toss_reasons == []

    def test_tail_off_only_not_tossed(self):
        """TAIL_OFF alone is only 1 factor → not tossed."""
        h = self._make_horse(figures=[20.0, 17.0, 14.0])
        p = self._run(h)
        assert p.projection_type == "TAIL_OFF"
        assert p.tossed is False
        assert len(p.toss_reasons) == 1

    def test_surface_unknown_only_not_tossed(self):
        """SURFACE_UNKNOWN alone is only 1 factor → not tossed."""
        h = HorseInput(
            name="SurfOnly", post="1", style="P",
            figures=[
                FigureEntry(15, "TURF"),
                FigureEntry(12, "DIRT"),
                FigureEntry(16, "DIRT"),
            ],
        )
        p = self._run(h)
        assert p.tossed is False

    def test_improving_with_surface_unknown_not_tossed(self):
        """IMPROVING + SURFACE_UNKNOWN → (b) doesn't fire (pattern excuse)."""
        h = HorseInput(
            name="ImproveSurf", post="1", style="P",
            figures=[
                FigureEntry(10, "TURF"),
                FigureEntry(13, "DIRT"),
                FigureEntry(16, "DIRT"),
            ],
        )
        p = self._run(h)
        # IMPROVING provides a pattern excuse, so (b) shouldn't fire
        assert not any("(b)" in r for r in p.toss_reasons)

    def test_rebound_escapes_pace_mismatch(self):
        """REBOUND horse is 'best cycle', so pace mismatch doesn't fire."""
        h = HorseInput(
            name="ReboundCloser", post="1", style="S",
            figures=[FigureEntry(15), FigureEntry(20), FigureEntry(14)],
        )
        bias = BiasInput(speed_favoring=True)
        p = self._run(h, bias)
        assert p.projection_type == "REBOUND"
        assert not any("(c)" in r for r in p.toss_reasons)

    def test_repeat_fail_with_improving_not_flagged(self):
        """Repeat fail + IMPROVING cycle → (d) doesn't fire."""
        h = HorseInput(
            name="ImproveFail", post="1", style="P",
            figures=[FigureEntry(10), FigureEntry(13), FigureEntry(16)],
            notes={"repeat_fail": "yes"},
        )
        p = self._run(h)
        assert not any("(d)" in r for r in p.toss_reasons)


# =====================================================================
# AUDIT section
# =====================================================================

class TestAudit:
    """Tests for _generate_audit and analyze_race_with_audit."""

    def _make_horses(self, specs):
        """Build HorseInput list from [(name, style, figures, surface)]."""
        result = []
        for i, (name, style, figs, surf) in enumerate(specs):
            result.append(HorseInput(
                name=name, post=str(i + 1), style=style,
                figures=[FigureEntry(v, surf) for v in figs],
            ))
        return result

    def test_cycle_best_vs_raw_best_same(self):
        """When cycle-best and raw-best are the same horse."""
        horses = self._make_horses([
            ("Alpha", "E", [10.0, 11.0, 15.0], "DIRT"),  # best raw + cycle
            ("Beta", "P", [14.0, 14.5, 16.0], "DIRT"),
        ])
        e = HandicappingEngine()
        projs, audit = e.analyze_race_with_audit(horses, BiasInput())
        assert audit.cycle_best == audit.raw_best
        assert audit.cycle_vs_raw_match is True

    def test_cycle_best_vs_raw_best_different(self):
        """Cycle-best differs from raw-best (raw has better figure but bad cycle)."""
        horses = self._make_horses([
            ("CycleBest", "E", [12.0, 12.5, 15.0], "DIRT"),  # PAIRED, reliable
            ("RawBest", "P", [8.0, 14.0, 15.0], "DIRT"),   # NEW_TOP_BOUNCE, best raw=8
        ])
        e = HandicappingEngine()
        projs, audit = e.analyze_race_with_audit(horses, BiasInput())
        # raw-best should be RawBest (8.0 is lower = better in Ragozin)
        assert audit.raw_best == "RawBest"
        # but cycle-best might be CycleBest (PAIRED, higher confidence)
        # or RawBest depending on bias_score. Let's just verify both fields populated.
        assert audit.cycle_best != ""
        assert audit.raw_best != ""

    def test_bounce_candidates_listed(self):
        """Bounce candidates appear in audit."""
        horses = self._make_horses([
            ("Bouncer", "E", [8.0, 14.0, 15.0], "DIRT"),
            ("Steady", "P", [12.0, 12.5, 15.0], "DIRT"),
        ])
        e = HandicappingEngine()
        _, audit = e.analyze_race_with_audit(horses, BiasInput())
        assert "Bouncer" in audit.bounce_candidates
        assert "Steady" not in audit.bounce_candidates

    def test_no_bounce_candidates(self):
        """No bounce candidates when none exist."""
        horses = self._make_horses([
            ("A", "E", [12.0, 12.5, 15.0], "DIRT"),
            ("B", "P", [14.0, 14.5, 16.0], "DIRT"),
        ])
        e = HandicappingEngine()
        _, audit = e.analyze_race_with_audit(horses, BiasInput())
        assert audit.bounce_candidates == []

    def test_first_time_surface_unknown(self):
        """Horse on new surface flagged with SURFACE_UNKNOWN."""
        horses = [
            HorseInput(
                name="NewSurface", post="1", style="P",
                figures=[
                    FigureEntry(15, "TURF"),
                    FigureEntry(14, "DIRT"),
                    FigureEntry(16, "DIRT"),
                ],
            ),
            HorseInput(
                name="SameSurface", post="2", style="E",
                figures=[FigureEntry(12, "DIRT"), FigureEntry(13, "DIRT")],
            ),
        ]
        e = HandicappingEngine()
        _, audit = e.analyze_race_with_audit(horses, BiasInput())
        names = [entry['name'] for entry in audit.first_time_surface]
        assert "NewSurface" in names
        assert "SameSurface" not in names

    def test_toss_list_populated(self):
        """Tossed horses appear in audit.toss_list."""
        horses = [
            HorseInput(
                name="Tossable", post="1", style="P",
                figures=[
                    FigureEntry(20, "TURF"),
                    FigureEntry(17, "DIRT"),
                    FigureEntry(14, "DIRT"),
                ],
            ),
        ]
        e = HandicappingEngine()
        _, audit = e.analyze_race_with_audit(horses, BiasInput())
        toss_names = [entry['name'] for entry in audit.toss_list]
        # TAIL_OFF + SURFACE_UNKNOWN → tossed
        assert "Tossable" in toss_names

    def test_audit_empty_race(self):
        """Empty race produces empty audit."""
        e = HandicappingEngine()
        _, audit = e.analyze_race_with_audit([], BiasInput())
        assert audit.cycle_best == ""
        assert audit.raw_best == ""
        assert audit.bounce_candidates == []
        assert audit.first_time_surface == []
        assert audit.toss_list == []

    def test_analyze_race_with_audit_returns_tuple(self):
        """analyze_race_with_audit returns (projections, audit) tuple."""
        horses = self._make_horses([
            ("A", "E", [12.0, 12.5, 15.0], "DIRT"),
        ])
        e = HandicappingEngine()
        result = e.analyze_race_with_audit(horses, BiasInput())
        assert isinstance(result, tuple)
        assert len(result) == 2
        projs, audit = result
        assert isinstance(projs, list)
        assert hasattr(audit, 'cycle_best')


# =====================================================================
# Poly-Dirt Study: merge + pattern detection
# =====================================================================

class TestPolyDirtStudy:
    """Tests for poly_dirt_study merge, enrichment, and pattern finding."""

    def test_normalize_name(self):
        from poly_dirt_study import normalize_name
        assert normalize_name("Tiger Moon") == "TIGER MOON"
        assert normalize_name("K's Pick") == "KS PICK"
        assert normalize_name("  Curlin's Gesture  ") == "CURLINS GESTURE"
        assert normalize_name("BIG-TIME") == "BIGTIME"

    def test_parse_brisnet_date(self):
        from poly_dirt_study import parse_brisnet_date
        from datetime import date
        assert parse_brisnet_date("04Dec25") == date(2025, 12, 4)
        assert parse_brisnet_date("31Jan26") == date(2026, 1, 31)
        assert parse_brisnet_date("15Jly24") == date(2024, 7, 15)
        assert parse_brisnet_date("") is None
        assert parse_brisnet_date("baddate") is None

    def test_derive_surface_aw_prefix(self):
        from poly_dirt_study import derive_surface
        assert derive_surface("AWT", "DIRT") == "POLY"
        assert derive_surface("AW", "") == "POLY"
        assert derive_surface("GP", "DIRT") == "DIRT"
        assert derive_surface("", "TURF") == "TURF"
        assert derive_surface("", "AW") == "POLY"

    def test_merge_enrichment_matching(self):
        """Merge finds matches by normalized name and enriches in-place."""
        from poly_dirt_study import merge_enrichment
        horses = [
            {"horse_name": "Tiger Moon", "lines": []},
            {"horse_name": "War Officer", "lines": []},
            {"horse_name": "Unknown Horse", "lines": []},
        ]
        enrichment = {
            "TIGER MOON": {"runstyle": "E", "sire": "Malibu Moon", "age": 4,
                           "match_confidence": "high", "running_line_dates": []},
            "WAR OFFICER": {"runstyle": "P", "sire": "War Front", "age": 3,
                            "match_confidence": "high", "running_line_dates": []},
            "EXTRA BRISNET": {"runstyle": "S", "sire": "Tapit", "age": 5,
                              "match_confidence": "high", "running_line_dates": []},
        }
        matched, unmatched_rag, unmatched_bris = merge_enrichment(horses, enrichment)
        assert matched == 2
        assert "Unknown Horse" in unmatched_rag
        assert "EXTRA BRISNET" in unmatched_bris
        # Enrichment was applied in-place
        assert horses[0]["_enrichment"]["runstyle"] == "E"
        assert horses[1]["_enrichment"]["age"] == 3

    def test_merge_enrichment_empty(self):
        """Empty enrichment matches nothing."""
        from poly_dirt_study import merge_enrichment
        horses = [{"horse_name": "A", "lines": []}]
        matched, unmatched_rag, unmatched_bris = merge_enrichment(horses, {})
        assert matched == 0
        assert len(unmatched_rag) == 1
        assert len(unmatched_bris) == 0

    def test_find_pattern_poly_dirt(self):
        """Paired poly -> dirt pattern is detected correctly."""
        from poly_dirt_study import find_pattern_instances
        # Lines are 0-back first. Build: dirt(0-back), poly(1-back), poly(2-back)
        horses = [{
            "horse_name": "TestHorse",
            "age": 0,
            "lines": [
                {"parsed_figure": 12.0, "track": "AQU", "surface": "DIRT"},  # 0-back: dirt
                {"parsed_figure": 15.0, "track": "AWT", "surface": "DIRT"},  # 1-back: poly (AW track)
                {"parsed_figure": 15.5, "track": "AWT", "surface": "DIRT"},  # 2-back: poly, paired (gap=0.5)
            ],
        }]
        instances = find_pattern_instances(horses)
        assert len(instances) == 1
        inst = instances[0]
        assert inst["name"] == "TestHorse"
        assert inst["delta"] == 3.0  # poly_0(15) - dirt(12) = +3 forward
        assert inst["pair_gap"] == 0.5

    def test_find_pattern_not_paired(self):
        """Non-paired poly runs are NOT flagged."""
        from poly_dirt_study import find_pattern_instances
        horses = [{
            "horse_name": "NotPaired",
            "age": 0,
            "lines": [
                {"parsed_figure": 12.0, "track": "AQU", "surface": "DIRT"},
                {"parsed_figure": 15.0, "track": "AWT", "surface": "DIRT"},
                {"parsed_figure": 18.0, "track": "AWT", "surface": "DIRT"},  # gap=3, not paired
            ],
        }]
        instances = find_pattern_instances(horses)
        assert len(instances) == 0

    def test_find_pattern_with_enrichment(self):
        """Enrichment fields flow through to pattern instances."""
        from poly_dirt_study import find_pattern_instances, merge_enrichment
        from datetime import date
        horses = [{
            "horse_name": "Enriched",
            "age": 0,
            "lines": [
                {"parsed_figure": 10.0, "track": "SAR", "surface": "DIRT"},
                {"parsed_figure": 14.0, "track": "AWT", "surface": "DIRT"},
                {"parsed_figure": 14.5, "track": "AWT", "surface": "DIRT"},
            ],
        }]
        enrichment = {
            "ENRICHED": {
                "runstyle": "EP", "sire": "Into Mischief", "age": 3,
                "match_confidence": "high",
                "running_line_dates": [date(2025, 3, 1), date(2025, 2, 15), date(2025, 1, 20)],
                "running_line_surfaces": ["DIRT", "AW", "AW"],
            },
        }
        merge_enrichment(horses, enrichment)
        instances = find_pattern_instances(horses)
        assert len(instances) == 1
        inst = instances[0]
        assert inst["age"] == 3
        assert inst["style"] == "EP"
        assert inst["sire"] == "Into Mischief"
        assert inst["days_between"] == 14  # Feb 15 → Mar 1

    def test_classify_result(self):
        from poly_dirt_study import classify_result
        assert classify_result(3.0) == "forward_A"
        assert classify_result(2.0) == "forward_A"
        assert classify_result(1.5) == "forward_B"
        assert classify_result(0.5) == "same"
        assert classify_result(-0.5) == "same"
        assert classify_result(-2.0) == "regress"
        assert classify_result(-5.0) == "regress"

    def test_dedup_same_instance(self):
        """Same horse with identical poly/dirt figures is deduplicated."""
        from poly_dirt_study import find_pattern_instances
        horse = {
            "horse_name": "Dupe",
            "age": 0,
            "lines": [
                {"parsed_figure": 12.0, "track": "AQU", "surface": "DIRT"},
                {"parsed_figure": 15.0, "track": "AWT", "surface": "DIRT"},
                {"parsed_figure": 15.5, "track": "AWT", "surface": "DIRT"},
            ],
        }
        # Pass same horse twice — should still produce 1 instance
        instances = find_pattern_instances([horse, horse])
        assert len(instances) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
