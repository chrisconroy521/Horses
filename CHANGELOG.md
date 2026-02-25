# Changelog

## v0.6.0-odds-bets (2026-02-24)

### Features
- **Odds (ML) in ranked table**: Morning line odds column in Full ranked table and Best Bets table, populated from `odds_snapshots` (post-priority lookup)
- **Odds coverage metric**: "Odds coverage: X/Y horses" shown near ranked table, best bets, and bet builder tickets
- **NO_ODDS tagging**: Horses without odds data tagged with `NO_ODDS` in ranked table and best bets
- **Show Odds toggle**: Default-ON checkbox controls odds visibility in ranked table, best bets, and ticket headers
- **EXACTA per-horse odds**: Ticket headers render per-horse odds for KEY and SAVER structures (e.g., `EXACTA KEY Alpha @ 5/1 over Beta @ 8/1`)
- **WIN ticket raw odds**: WIN ticket headers prefer raw odds string (`5/1`) over decimal format
- **Kelly sizing with ML odds**: Bet builder uses Kelly criterion when morning line odds are available
- **BRISNET ML odds pipeline**: End-to-end parsing of morning line odds from BRISNET PPs, stored in `odds_snapshots`, injected into bet builder

### Infrastructure
- `parse_odds_decimal()` in `brisnet_parser.py` — converts all ML odds formats (fraction, dash, decimal, even/evs)
- `get_odds_snapshots_full()` in `persistence.py` — returns full snapshot rows with `odds_raw`
- `GET /odds/snapshots/{session_id}` endpoint for UI consumption
- 24 odds-specific tests covering parsing, persistence, bet builder, ranked table display, and NO_ODDS tagging
