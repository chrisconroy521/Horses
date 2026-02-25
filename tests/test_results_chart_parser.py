"""Tests for Equibase results chart PDF parser."""
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from results_chart_parser import ResultsChartParser, chart_to_rows
from ingest_results import preview_pdf


class TestChartPdfExtraction:
    """Test raw text extraction from fixture PDF."""

    def test_extract_text_non_empty(self, fixture_chart_pdf):
        parser = ResultsChartParser()
        raw = parser._extract_text(fixture_chart_pdf)
        assert raw, "Extracted text should be non-empty"
        assert len(raw) > 100, "Extracted text should be >100 chars"

    def test_extracted_text_contains_markers(self, fixture_chart_pdf):
        parser = ResultsChartParser()
        raw = parser._extract_text(fixture_chart_pdf)
        assert "FIRST RACE" in raw, "Should contain FIRST RACE header"
        assert "BOLD OPTION" in raw, "Should contain horse name BOLD OPTION"
        assert "$2 Mutuel Prices" in raw, "Should contain payoff header"


class TestChartPdfParsing:
    """Test structured parsing from fixture PDF."""

    def test_parse_produces_rows(self, fixture_chart_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_chart_pdf)
        rows = chart_to_rows(result)
        assert len(rows) >= 5, f"Expected >=5 rows from 7-entry fixture, got {len(rows)}"

    def test_parse_race_count(self, fixture_chart_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_chart_pdf)
        assert len(result.card.races) == 2, (
            f"Expected 2 races, got {len(result.card.races)}"
        )

    def test_parse_fields_populated(self, fixture_chart_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_chart_pdf)
        rows = chart_to_rows(result)

        # Find BOLD OPTION entry
        bold = [r for r in rows if "BOLD OPTION" in r.get("horse_name", "")]
        assert bold, "BOLD OPTION should be in parsed rows"

        entry = bold[0]
        assert entry["finish_pos"] == 1, f"BOLD OPTION finish should be 1, got {entry['finish_pos']}"
        assert entry["odds"] == 3.70, f"BOLD OPTION odds should be 3.70, got {entry['odds']}"
        assert entry["win_payoff"] == 9.40, f"BOLD OPTION win payoff should be 9.40, got {entry['win_payoff']}"

    def test_parse_confidence_above_threshold(self, fixture_chart_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_chart_pdf)
        assert result.confidence > 0.5, (
            f"Confidence should be >0.5 for well-formed fixture, got {result.confidence}"
        )

    def test_missing_fields_list(self, fixture_chart_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_chart_pdf)
        assert isinstance(result.missing_fields, list)

    def test_program_field_populated(self, fixture_chart_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_chart_pdf)
        rows = chart_to_rows(result)
        for r in rows:
            assert "program" in r, "Each row should have a program field"


class TestCoupledEntries:
    """Test parsing of coupled entries (1/1A)."""

    def test_coupled_entry_detected(self, fixture_coupled_entry_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_coupled_entry_pdf)
        rows = chart_to_rows(result)

        coupled = [r for r in rows if r.get("coupled_with")]
        assert len(coupled) >= 1, "Should detect at least one coupled entry"

    def test_coupled_program_1a(self, fixture_coupled_entry_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_coupled_entry_pdf)
        rows = chart_to_rows(result)

        entry_1a = [r for r in rows if r.get("program") == "1A"
                     or r.get("program") == "1a"]
        assert entry_1a, "Should have entry with program '1A'"
        assert entry_1a[0]["coupled_with"] == "1", "1A should be coupled with 1"

    def test_coupled_payoffs_assigned(self, fixture_coupled_entry_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_coupled_entry_pdf)
        rows = chart_to_rows(result)

        winner = [r for r in rows if r.get("program") in ("1", "1") and r.get("finish_pos") == 1]
        assert winner, "Should have a winner with program 1"
        assert winner[0]["win_payoff"] == 10.60, (
            f"Winner win payoff should be 10.60, got {winner[0]['win_payoff']}"
        )

    def test_exotic_payoffs_not_mixed_into_wps(self, fixture_coupled_entry_pdf):
        """Exacta/Trifecta lines should not create spurious payoff entries."""
        parser = ResultsChartParser()
        result = parser.parse(fixture_coupled_entry_pdf)
        rows = chart_to_rows(result)

        # No entry should have a payoff of 42.60 (the Exacta payout)
        for r in rows:
            if r.get("win_payoff"):
                assert r["win_payoff"] != 42.60, "Exacta payout leaked into WPS"


class TestScratches:
    """Test parsing of scratched entries."""

    def test_scratched_entries_detected(self, fixture_scratches_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_scratches_pdf)
        rows = chart_to_rows(result)

        scratched = [r for r in rows if r.get("scratched")]
        assert len(scratched) >= 1, (
            f"Should detect at least 1 scratched entry, got {len(scratched)}"
        )

    def test_active_entries_correct(self, fixture_scratches_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_scratches_pdf)
        rows = chart_to_rows(result)

        active = [r for r in rows if not r.get("scratched")]
        assert len(active) == 4, f"Should have 4 active entries, got {len(active)}"

    def test_scratches_excluded_from_confidence(self, fixture_scratches_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_scratches_pdf)
        # Confidence should still be reasonable despite scratches
        assert result.confidence > 0.3, (
            f"Confidence should be >0.3 with scratches, got {result.confidence}"
        )


class TestAltPayoffLabels:
    """Test handling of exotic payoff labels."""

    def test_exotic_lines_skipped(self, fixture_alt_payoff_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_alt_payoff_pdf)
        rows = chart_to_rows(result)

        # No WPS entry should have exotic amounts
        for r in rows:
            if r.get("win_payoff"):
                assert r["win_payoff"] < 100, (
                    f"Exotic payoff leaked: {r['horse_name']} win={r['win_payoff']}"
                )

    def test_wps_payoffs_correct(self, fixture_alt_payoff_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_alt_payoff_pdf)
        rows = chart_to_rows(result)

        winner = [r for r in rows if r.get("finish_pos") == 1]
        assert winner, "Should have a winner"
        assert winner[0]["win_payoff"] == 14.80, (
            f"Winner win payoff should be 14.80, got {winner[0]['win_payoff']}"
        )

    def test_five_entries_parsed(self, fixture_alt_payoff_pdf):
        parser = ResultsChartParser()
        result = parser.parse(fixture_alt_payoff_pdf)
        rows = chart_to_rows(result)
        assert len(rows) == 5, f"Expected 5 entries, got {len(rows)}"


class TestDryRunPreview:
    """Test the preview_pdf dry-run function."""

    def test_preview_returns_all_fields(self, fixture_chart_pdf):
        preview = preview_pdf(fixture_chart_pdf)
        assert "races" in preview
        assert "entries" in preview
        assert "sample_rows" in preview
        assert "confidence" in preview
        assert "missing_fields" in preview
        assert "track" in preview
        assert "race_date" in preview

    def test_preview_sample_capped(self, fixture_chart_pdf):
        preview = preview_pdf(fixture_chart_pdf)
        assert len(preview["sample_rows"]) <= 10

    def test_preview_track_detected(self, fixture_chart_pdf):
        preview = preview_pdf(fixture_chart_pdf)
        assert "Gulfstream" in preview["track"]

    def test_preview_date_detected(self, fixture_chart_pdf):
        preview = preview_pdf(fixture_chart_pdf)
        assert preview["race_date"] == "02/26/2026"

    def test_preview_scratches_counted(self, fixture_scratches_pdf):
        preview = preview_pdf(fixture_scratches_pdf)
        assert preview["scratches"] >= 1, "Should count scratched entries"
        assert preview["entries"] == 4, "Active entries should be 4"
