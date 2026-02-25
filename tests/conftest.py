"""Shared fixtures for results chart parser tests."""
import pytest
from pathlib import Path

import fitz  # PyMuPDF


def _write_lines(page, lines, start_y=50, x=50, fontsize=9, bold=False):
    """Helper: write list of text lines to a page, return final y."""
    fname = "helv" if not bold else "hebo"
    y = start_y
    for line in lines:
        fs = fontsize
        b = bold
        if isinstance(line, tuple):
            text, fs, b = line[0], line[1], line[2] if len(line) > 2 else False
            fname = "hebo" if b else "helv"
        else:
            text = line
            fname = "hebo" if bold else "helv"
        page.insert_text((x, y), text, fontsize=fs, fontname=fname)
        y += fs + 4
    return y


@pytest.fixture
def fixture_chart_pdf(tmp_path):
    """Generate a 2-race Equibase-style chart PDF for testing.

    Contains:
    - Gulfstream Park / February 26, 2026
    - FIRST RACE (4 entries, Dirt, 6 Furlongs)
    - SECOND RACE (3 entries, Turf, 1 1/16 Miles)
    - Starters tables with Pgm/Horse/Fin/Odds columns
    - $2 Mutuel Prices payoff sections
    """
    pdf_path = tmp_path / "test_chart.pdf"
    doc = fitz.open()  # new empty PDF

    # --- Page 1: Header + FIRST RACE ---
    page1 = doc.new_page(width=612, height=792)
    y = 50

    def write(page, text, ypos, fontsize=10, bold=False):
        fname = "helv" if not bold else "hebo"
        page.insert_text(
            (50, ypos), text, fontsize=fontsize, fontname=fname,
        )
        return ypos + fontsize + 4

    y = write(page1, "Copyright 2026 Equibase Company LLC.  All Rights Reserved.", y, 8)
    y = write(page1, "Gulfstream Park", y, 14, bold=True)
    y = write(page1, "February 26, 2026", y, 12)
    y += 10

    y = write(page1, "FIRST RACE", y, 12, bold=True)
    y = write(page1, "Dirt - 6 Furlongs", y, 10)
    y += 5
    y = write(page1, "Pgm  Horse                  Jockey          Wt   PP  Str  Fin   Odds", y, 9)
    y = write(page1, "---  --------------------   ------------    ---  --  ---  ---   ------", y, 8)

    # 4 entries
    entries_r1 = [
        ("1", "BOLD OPTION", "Ortiz I Jr", "122", "1", "1", "1", "3.70"),
        ("2", "STORM RUNNER", "Prat F", "118", "2", "3", "2", "5.20"),
        ("3", "DARK MAGIC", "Saez L", "120", "3", "2", "3", "8.40"),
        ("4", "FAST COPPER", "Castellano J", "117", "4", "4", "4", "12.60"),
    ]
    for pgm, name, jock, wt, pp, strp, fin, odds in entries_r1:
        line = f"  {pgm}   {name:<22s} {jock:<16s} {wt}  {pp:>2s}   {strp:>2s}   {fin:>2s}   {odds:>6s}"
        y = write(page1, line, y, 9)

    y += 15
    y = write(page1, "$2 Mutuel Prices", y, 10, bold=True)
    y = write(page1, "1 - BOLD OPTION                    9.40  5.20  3.60", y, 9)
    y = write(page1, "2 - STORM RUNNER                          6.80  4.40", y, 9)
    y = write(page1, "3 - DARK MAGIC                                  5.20", y, 9)

    # --- Page 2: SECOND RACE ---
    page2 = doc.new_page(width=612, height=792)
    y = 50

    y = write(page2, "SECOND RACE", y, 12, bold=True)
    y = write(page2, "Turf - 1 1/16 Miles", y, 10)
    y += 5
    y = write(page2, "Pgm  Horse                  Jockey          Wt   PP  Str  Fin   Odds", y, 9)
    y = write(page2, "---  --------------------   ------------    ---  --  ---  ---   ------", y, 8)

    entries_r2 = [
        ("1", "MORNING STAR", "Rosario J", "124", "1", "2", "1", "2.10"),
        ("2", "SILVER CREEK", "Gaffalione T", "121", "2", "1", "2", "4.50"),
        ("3", "RED PHANTOM", "Velazquez J", "119", "3", "3", "3", "9.80"),
    ]
    for pgm, name, jock, wt, pp, strp, fin, odds in entries_r2:
        line = f"  {pgm}   {name:<22s} {jock:<16s} {wt}  {pp:>2s}   {strp:>2s}   {fin:>2s}   {odds:>6s}"
        y = write(page2, line, y, 9)

    y += 15
    y = write(page2, "$2 Mutuel Prices", y, 10, bold=True)
    y = write(page2, "1 - MORNING STAR                   6.20  3.40  2.80", y, 9)
    y = write(page2, "2 - SILVER CREEK                          5.60  3.20", y, 9)
    y = write(page2, "3 - RED PHANTOM                                 4.60", y, 9)

    doc.save(str(pdf_path))
    doc.close()

    return str(pdf_path)


@pytest.fixture
def fixture_coupled_entry_pdf(tmp_path):
    """Chart with coupled entries (1/1A) in a single race.

    THIRD RACE at Churchill Downs, Dirt, 7 Furlongs.
    Entry 1 and 1A are coupled; 1A finishes 2nd.
    """
    pdf_path = tmp_path / "coupled_chart.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 50

    def w(text, ypos, fs=9, bold=False):
        page.insert_text((50, ypos), text, fontsize=fs,
                         fontname="hebo" if bold else "helv")
        return ypos + fs + 4

    y = w("Churchill Downs", y, 14, True)
    y = w("March 1, 2026", y, 12)
    y += 10
    y = w("THIRD RACE", y, 12, True)
    y = w("Dirt - 7 Furlongs", y, 10)
    y += 5
    y = w("Pgm  Horse                  Jockey          Wt   PP  Str  Fin   Odds", y, 9)
    y = w("---  --------------------   ------------    ---  --  ---  ---   ------", y, 8)

    entries = [
        ("1",  "ROYAL FLEET",   "Ortiz I Jr",    "122", "1", "2", "1", "4.30"),
        ("1A", "ROYAL GUARD",   "Prat F",        "120", "6", "1", "2", "4.30"),
        ("2",  "NIGHT PATROL",  "Saez L",        "118", "2", "3", "3", "7.10"),
        ("3",  "COPPER TRAIL",  "Castellano J",  "119", "3", "5", "4", "15.20"),
        ("4",  "FAST BREAK",    "Rosario J",     "121", "4", "4", "5", "6.50"),
        ("5",  "WIND CHASER",   "Gaffalione T",  "117", "5", "6", "6", "22.40"),
    ]
    for pgm, name, jock, wt, pp, strp, fin, odds in entries:
        line = f"  {pgm:<3s} {name:<22s} {jock:<16s} {wt}  {pp:>2s}   {strp:>2s}   {fin:>2s}   {odds:>6s}"
        y = w(line, y, 9)

    y += 15
    y = w("$2 Mutuel Prices", y, 10, True)
    y = w("1 - ROYAL FLEET                    10.60  5.80  3.40", y, 9)
    y = w("1A - ROYAL GUARD                          6.20  4.10", y, 9)
    y = w("2 - NIGHT PATROL                                5.80", y, 9)
    y += 10
    y = w("$2 Exacta  1-1A  paid  $42.60", y, 9)
    y = w("$1 Trifecta  1-1A-2  paid  $98.40", y, 9)

    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture
def fixture_scratches_pdf(tmp_path):
    """Chart with scratched entries.

    FOURTH RACE at Saratoga, Turf, 1 Mile.
    Entry 3 and 5 are scratched.
    """
    pdf_path = tmp_path / "scratches_chart.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 50

    def w(text, ypos, fs=9, bold=False):
        page.insert_text((50, ypos), text, fontsize=fs,
                         fontname="hebo" if bold else "helv")
        return ypos + fs + 4

    y = w("Saratoga", y, 14, True)
    y = w("August 15, 2026", y, 12)
    y += 10
    y = w("FOURTH RACE", y, 12, True)
    y = w("Turf - 1 Mile", y, 10)
    y += 5
    y = w("Pgm  Horse                  Jockey          Wt   PP  Str  Fin   Odds", y, 9)
    y = w("---  --------------------   ------------    ---  --  ---  ---   ------", y, 8)

    active = [
        ("1", "EMERALD ISLE",  "Ortiz I Jr",    "122", "1", "1", "1", "3.20"),
        ("2", "GOLDEN ARROW",  "Prat F",        "120", "2", "2", "2", "5.60"),
        ("4", "STORM FRONT",   "Saez L",        "119", "3", "3", "3", "8.90"),
        ("6", "RIVER DANCE",   "Castellano J",  "117", "4", "4", "4", "14.30"),
    ]
    for pgm, name, jock, wt, pp, strp, fin, odds in active:
        line = f"  {pgm}   {name:<22s} {jock:<16s} {wt}  {pp:>2s}   {strp:>2s}   {fin:>2s}   {odds:>6s}"
        y = w(line, y, 9)

    y += 10
    y = w("Scratched: 3 - LUCKY PENNY (trainer), 5 - BLUE HORIZON (vet)", y, 9)

    y += 15
    y = w("$2 Mutuel Prices", y, 10, True)
    y = w("1 - EMERALD ISLE                   8.40  4.60  3.20", y, 9)
    y = w("2 - GOLDEN ARROW                          7.20  4.80", y, 9)
    y = w("4 - STORM FRONT                                 6.40", y, 9)

    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture
def fixture_alt_payoff_pdf(tmp_path):
    """Chart with alternative/exotic payoff labels mixed in.

    FIFTH RACE at Del Mar, Dirt, 6 Furlongs.
    Has $2 Exacta, $1 Trifecta, Daily Double lines after WPS.
    """
    pdf_path = tmp_path / "alt_payoff_chart.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 50

    def w(text, ypos, fs=9, bold=False):
        page.insert_text((50, ypos), text, fontsize=fs,
                         fontname="hebo" if bold else "helv")
        return ypos + fs + 4

    y = w("Del Mar", y, 14, True)
    y = w("July 20, 2026", y, 12)
    y += 10
    y = w("FIFTH RACE", y, 12, True)
    y = w("Dirt - 6 Furlongs", y, 10)
    y += 5
    y = w("Pgm  Horse                  Jockey          Wt   PP  Str  Fin   Odds", y, 9)
    y = w("---  --------------------   ------------    ---  --  ---  ---   ------", y, 8)

    entries = [
        ("1", "PACIFIC DREAM",  "Smith M",       "122", "1", "3", "1", "6.40"),
        ("2", "SUNSET GLOW",    "Rispoli U",     "120", "2", "1", "2", "3.10"),
        ("3", "DESERT STAR",    "Prat F",        "118", "3", "2", "3", "9.70"),
        ("4", "OCEAN BREEZE",   "Velazquez J",   "121", "4", "4", "4", "11.50"),
        ("5", "CANYON WIND",    "Rosario J",     "119", "5", "5", "5", "18.20"),
    ]
    for pgm, name, jock, wt, pp, strp, fin, odds in entries:
        line = f"  {pgm}   {name:<22s} {jock:<16s} {wt}  {pp:>2s}   {strp:>2s}   {fin:>2s}   {odds:>6s}"
        y = w(line, y, 9)

    y += 15
    y = w("$2 Mutuel Prices", y, 10, True)
    y = w("1 - PACIFIC DREAM                  14.80  6.40  4.20", y, 9)
    y = w("2 - SUNSET GLOW                           4.60  3.40", y, 9)
    y = w("3 - DESERT STAR                                  6.80", y, 9)
    y += 10
    y = w("$2 Exacta  1-2  paid  $38.20", y, 9)
    y = w("$1 Trifecta  1-2-3  paid  $124.50", y, 9)
    y = w("$2 Superfecta  1-2-3-4  paid  $412.80", y, 9)
    y = w("$2 Daily Double  (3-1)  paid  $28.60", y, 9)

    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)
