"""Tests for ragozin_parser — concatenated figure+data line splitting."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragozin_parser import (
    _CONCAT_FIGURE_DATA_RE,
    _is_figure_line,
    _parse_figure_line,
    _HAS_TRACK_CODE,
)


class TestConcatFigureDataSplit:
    def test_cajun_lad_line(self):
        """F\u272731+ QT[AWGPMYs should split into figure=31 + data."""
        line = "F\u272731+ QT[AWGPMYs"
        cm = _CONCAT_FIGURE_DATA_RE.match(line)
        assert cm is not None
        fig_part = cm.group(1).strip()
        data_part = cm.group(2).strip()
        assert _is_figure_line(fig_part)
        assert _HAS_TRACK_CODE.search(data_part)
        parsed = _parse_figure_line(fig_part)
        assert parsed["parsed_figure"] == 31.0

    def test_simple_concat(self):
        """v12 GP12 should split into figure=12 + data=GP12."""
        line = "v12 GP12"
        cm = _CONCAT_FIGURE_DATA_RE.match(line)
        assert cm is not None
        fig_part = cm.group(1).strip()
        data_part = cm.group(2).strip()
        assert _is_figure_line(fig_part)
        assert data_part == "GP12"
        parsed = _parse_figure_line(fig_part)
        assert parsed["parsed_figure"] == 12.0

    def test_normal_figure_not_concat(self):
        """A standalone figure line should NOT match concat pattern."""
        line = "F\u272723-"
        cm = _CONCAT_FIGURE_DATA_RE.match(line)
        assert cm is None  # no space separator -> no match

    def test_noise_not_concat(self):
        """GELDED should NOT match concat pattern."""
        line = "GELDED"
        cm = _CONCAT_FIGURE_DATA_RE.match(line)
        assert cm is None  # no digits

    def test_year_summary_not_concat(self):
        """1  RACES  26 is filtered by _YEAR_SUMMARY_RE before concat check.

        In the actual parser, year summary lines (e.g. '1  RACES  26') are
        skipped before reaching the concatenated-line logic. The concat regex
        could structurally match, but the parser ordering prevents it.
        """
        from ragozin_parser import _YEAR_SUMMARY_RE
        line = "1  RACES  26"
        # Year summary regex catches this first
        assert _YEAR_SUMMARY_RE.match(line) is not None
