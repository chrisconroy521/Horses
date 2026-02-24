"""Unit tests for age extraction and surface parsing fixes."""
import pytest
from datetime import date
from ragozin_parser import _foal_year_to_age, _parse_data_line


# =====================================================================
# Age: foal-year to racing age
# =====================================================================

class TestFoalYearToAge:
    def test_foal_year_23_in_2026(self):
        """Foal year 23 (born 2023) = 3yo in 2026."""
        age = _foal_year_to_age(23)
        expected = date.today().year - 2023
        assert age == expected

    def test_foal_year_22(self):
        """Foal year 22 (born 2022) = 4yo in 2026."""
        age = _foal_year_to_age(22)
        expected = date.today().year - 2022
        assert age == expected

    def test_foal_year_21(self):
        """Foal year 21 (born 2021) = 5yo in 2026."""
        age = _foal_year_to_age(21)
        expected = date.today().year - 2021
        assert age == expected

    def test_foal_year_18(self):
        """Foal year 18 (born 2018) = 8yo in 2026."""
        age = _foal_year_to_age(18)
        expected = date.today().year - 2018
        assert age == expected

    def test_small_value_is_age(self):
        """Values < 10 are treated as the actual age."""
        assert _foal_year_to_age(4) == 4
        assert _foal_year_to_age(3) == 3
        assert _foal_year_to_age(7) == 7

    def test_boundary_10_is_foal_year(self):
        """Value 10 is treated as foal year 2010."""
        age = _foal_year_to_age(10)
        expected = date.today().year - 2010
        assert age == expected


# =====================================================================
# Surface: POLY/AW detection in data lines
# =====================================================================

class TestParseSurface:
    def test_aw_prefix_in_track_code(self):
        """'AWGP30' — AW in track code → POLY."""
        result = _parse_data_line("AWGP30")
        assert result['surface'] == 'POLY'

    def test_aw_embedded_in_track_code(self):
        """'wYQAWGP15' — AW inside track code → POLY."""
        result = _parse_data_line("wYQAWGP15")
        assert result['surface'] == 'POLY'

    def test_aw_in_prefix(self):
        """'awGP 6f' — aw in lowercase prefix → POLY."""
        result = _parse_data_line("awGP 6f")
        assert result['surface'] == 'POLY'

    def test_turf_surface(self):
        """'t GP 1m' — t in prefix → TURF."""
        result = _parse_data_line("t GP 1m")
        assert result['surface'] == 'TURF'

    def test_turf_in_complex_line(self):
        """Turf detection with messy prefix."""
        result = _parse_data_line("s$T[fMSMTWAW")
        # 'AW' in track code takes priority
        assert result['surface'] == 'POLY'

    def test_slop_surface(self):
        """'s CD 6f' — s in prefix → SLOP."""
        result = _parse_data_line("s CD 6f")
        assert result['surface'] == 'SLOP'

    def test_dirt_default(self):
        """'GP 6f 1:11' — no surface indicator → DIRT."""
        result = _parse_data_line("GP 6f 1:11")
        assert result['surface'] == 'DIRT'

    def test_dirt_with_numbers_only(self):
        """'MSKD28' — plain track code, no surface prefix → DIRT."""
        result = _parse_data_line("MSKD28")
        assert result['surface'] == 'DIRT'

    def test_no_track_code(self):
        """No uppercase letters → defaults to DIRT."""
        result = _parse_data_line("123 456")
        assert result['surface'] == 'DIRT'
        assert result['track'] == ''


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
