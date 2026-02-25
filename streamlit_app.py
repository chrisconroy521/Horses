import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os
import re
from pathlib import Path
import io
import base64
from datetime import datetime

from handicap_engine import HandicappingEngine, HorseInput, FigureEntry, BiasInput, AuditResult

# Configure page
st.set_page_config(
    page_title="Racing Sheets Parser",
    page_icon="🐎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
_AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Auth-aware API helpers
# ---------------------------------------------------------------------------

def _auth_headers() -> dict:
    """Return Authorization header if token is present in session state."""
    token = st.session_state.get("auth_token")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def api_get(path: str, **kwargs):
    """GET request to the API with auth header injected."""
    headers = {**_auth_headers(), **kwargs.pop("headers", {})}
    return requests.get(f"{API_BASE_URL}{path}", headers=headers, **kwargs)


def api_post(path: str, **kwargs):
    """POST request to the API with auth header injected."""
    headers = {**_auth_headers(), **kwargs.pop("headers", {})}
    return requests.post(f"{API_BASE_URL}{path}", headers=headers, **kwargs)


def api_delete(path: str, **kwargs):
    """DELETE request to the API with auth header injected."""
    headers = {**_auth_headers(), **kwargs.pop("headers", {})}
    return requests.delete(f"{API_BASE_URL}{path}", headers=headers, **kwargs)


# --- Minimal CSS for layout consistency ---
st.markdown("""<style>
h3 { margin-top: 1.5rem; margin-bottom: 0.75rem; }
div[data-testid="stExpander"] { margin-bottom: 0.75rem; }

/* Mobile-first responsive layout */
@media (max-width: 768px) {
    section[data-testid="stSidebar"] { width: 0px; min-width: 0px; }
    section[data-testid="stSidebar"][aria-expanded="true"] { width: 260px; min-width: 260px; }
    div[data-testid="column"] { width: 100% !important; flex: 100% !important; min-width: 100% !important; }
    button, .stButton > button, .stDownloadButton > button {
        min-height: 48px !important; min-width: 48px !important;
        font-size: 1rem !important; padding: 0.5rem 1rem !important;
    }
    .block-container { padding: 1rem 0.75rem !important; }
    .stDataFrame { font-size: 0.85rem; }
}
@media (min-width: 769px) and (max-width: 1024px) {
    .block-container { padding: 1.5rem 1rem !important; }
    div[data-testid="column"] { min-width: 45% !important; }
}
</style>""", unsafe_allow_html=True)

def main():
    # --- Auth gate ---
    if _AUTH_ENABLED:
        params = st.query_params
        if "token" in params:
            st.session_state["auth_token"] = params["token"]
            st.query_params.clear()
            st.rerun()
        if "error" in params and params.get("error") == "unauthorized":
            st.error("Your email is not on the authorized access list.")
            st.stop()
        if "auth_token" not in st.session_state:
            st.title("Racing Sheets Parser")
            st.markdown("Sign in with your Google account to continue.")
            st.link_button("Sign in with Google", f"{API_BASE_URL}/auth/google", type="primary")
            st.stop()

    st.title("Racing Sheets Parser")
    st.markdown("Sheets-first handicapping: upload Ragozin Sheets for figure cycles, then add Racing Form/BRISNET for program numbers, pace/runstyle, and track bias.")
    
    # Sidebar
    st.sidebar.header("Navigation")
    pages = [
        "Bet Commander", "Daily Best WIN Bets", "Dual Mode Betting", "Dashboard",
        "Upload PDF", "Engine",
        "Results", "Results Inbox", "Bet Builder", "Calibration",
        "Database", "Horse Past Performance", "Horses Overview",
        "Individual Horse Analysis", "Race Analysis", "Statistics",
        "Manage Sheets", "API Status",
    ]

    # Apply programmatic navigation (e.g. "Open in Engine" button)
    if '_nav_target' in st.session_state:
        st.session_state['_page_selector'] = st.session_state.pop('_nav_target')

    page = st.sidebar.selectbox("Choose a page", pages, key='_page_selector')
    st.sidebar.markdown("---")

    if page == "Bet Commander":
        bet_commander_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "Upload PDF":
        upload_page()
    elif page == "Engine":
        engine_page()
    elif page == "Horse Past Performance":
        horse_past_performance_page()
    elif page == "Horses Overview":
        horses_overview_page()
    elif page == "Individual Horse Analysis":
        individual_horse_analysis_page()
    elif page == "Race Analysis":
        race_analysis_page()
    elif page == "Database":
        database_page()
    elif page == "Results":
        results_page()
    elif page == "Bet Builder":
        bet_builder_page()
    elif page == "Results Inbox":
        results_inbox_page()
    elif page == "Daily Best WIN Bets":
        daily_wins_page()
    elif page == "Dual Mode Betting":
        dual_mode_page()
    elif page == "Calibration":
        calibration_page()
    elif page == "Statistics":
        statistics_page()
    elif page == "Manage Sheets":
        manage_sheets_page()
    elif page == "API Status":
        api_status_page()

def _extract_figure(line: dict) -> float:
    """Extract a numeric Ragozin figure from a race-line dict. Returns 0.0 on failure."""
    pf = line.get('parsed_figure')
    if pf and isinstance(pf, (int, float)) and pf > 0:
        return float(pf)
    fig_str = line.get('fig', '')
    if fig_str:
        m = re.search(r'(\d+(?:\.\d+)?)', str(fig_str))
        if m:
            return float(m.group(1))
    return 0.0


def _horse_dict_to_input(horse: dict, post: str) -> HorseInput:
    """Convert a parsed horse dict to a HorseInput for the engine.

    Populates ``notes`` from enrichment (lasix detection from QuickPlay)
    and ``drf_overrides`` with runstyle_rating, prime_power, class_rating.
    """
    figures = []
    for line in horse.get('lines', []):
        val = _extract_figure(line)
        surface = line.get('surface', line.get('surface_type', ''))
        flags = line.get('flags', []) + line.get('post_symbols', [])
        figures.append(FigureEntry(value=val, surface=surface, flags=flags))
    # Use runstyle from BRISNET data if available, else default "P"
    style = horse.get('runstyle', 'P') or 'P'
    # Map BRISNET styles: E/P -> EP
    style = style.replace('/', '')
    fig_source = horse.get('figure_source', 'ragozin')

    # Build notes from enrichment / QuickPlay
    notes: dict = {}
    qp_all = ' '.join(horse.get('quickplay_positive', []) + horse.get('quickplay_negative', []))
    qp_lower = qp_all.lower()
    if 'lasix' in qp_lower or 'first time lasix' in qp_lower:
        if 'first time' in qp_lower or '1st time' in qp_lower:
            notes['lasix'] = 'first'
        else:
            notes['lasix'] = 'second'

    # Build drf_overrides from secondary enrichment
    drf_overrides: dict = {}
    if horse.get('runstyle_rating'):
        drf_overrides['runstyle_rating'] = str(horse['runstyle_rating'])
    if horse.get('prime_power') is not None:
        drf_overrides['prime_power'] = str(horse['prime_power'])
    if horse.get('class_rating') is not None:
        drf_overrides['class_rating'] = str(horse['class_rating'])

    return HorseInput(
        name=horse.get('horse_name', 'Unknown'),
        post=post,
        style=style,
        figures=figures,
        figure_source=fig_source,
        notes=notes,
        drf_overrides=drf_overrides,
        age=horse.get('age', 0),
    )


def _run_race_projections(engine, race_horses, bias, scratches, race_num=0):
    """Build HorseInput list from race_horses and run engine.

    Returns ``(projections, audit)`` tuple.
    """
    horse_inputs = []
    for i, h in enumerate(race_horses):
        post = str(i + 1)
        hi = _horse_dict_to_input(h, post)
        horse_inputs.append(hi)
    return engine.analyze_race_with_audit(horse_inputs, bias, scratches=scratches)


def _best_bet_score(p):
    """Composite score: bias_score + confidence*10 - spread_penalty."""
    spread = p.projected_high - p.projected_low
    return p.bias_score + (p.confidence * 10) - spread


def _quality_badge(pct_missing: float) -> str:
    """Return a colored Streamlit badge for figure quality."""
    quality = 1.0 - pct_missing
    if quality >= 0.90:
        return f":green[OK {quality:.0%}]"
    elif quality >= 0.80:
        return f":orange[WARN {quality:.0%}]"
    else:
        return f":red[BLOCK {quality:.0%}]"


def _normalize_name_ui(name: str) -> str:
    """Uppercase, strip punctuation, collapse whitespace (mirror persistence logic)."""
    import re as _re
    n = (name or "").upper().strip()
    n = _re.sub(r"[\u2018\u2019\u201C\u201D'\-.,()\"']", "", n)
    n = _re.sub(r"\s+", " ", n)
    return n


def _lookup_odds(race_number, horse_name, post, odds_data, odds_by_post):
    """Lookup ML odds: priority (race_number, post), fallback (race_number, normalized_name)."""
    if post is not None:
        try:
            key_post = (int(race_number), int(post))
            if key_post in odds_by_post:
                return odds_by_post[key_post]
        except (ValueError, TypeError):
            pass
    norm = _normalize_name_ui(horse_name)
    key_name = (int(race_number), norm)
    return odds_data.get(key_name)


def _fmt_odds_display(odds_entry):
    """Format odds for table display: prefer raw string, fallback decimal."""
    if not odds_entry:
        return "\u2014"
    raw = odds_entry.get("odds_raw") or ""
    dec = odds_entry.get("odds_decimal")
    if raw:
        return raw
    if dec is not None:
        return f"{dec:.1f}-1"
    return "\u2014"


# --- Reusable layout helpers ---

def render_section_header(title: str, subtitle: str = None):
    """Standardized section header."""
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def render_metrics_row(metrics: dict):
    """Render a dict of {label: value} as a metrics row."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


def render_styled_table(df, caption: str = None, highlight_fn=None):
    """Render a styled dataframe with optional caption and row highlighting."""
    if caption:
        st.caption(caption)
    if highlight_fn:
        styled = df.style.apply(highlight_fn, axis=1)
    else:
        styled = df.style
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _persist_predictions(session_id, track, race_date, race_number, projections):
    """Save engine projections to the DB via the API."""
    try:
        payload = {
            "session_id": session_id,
            "track": track,
            "race_date": race_date,
            "race_number": race_number,
            "projections": [
                {
                    "name": p.name,
                    "projection_type": p.projection_type,
                    "bias_score": p.bias_score,
                    "raw_score": p.raw_score,
                    "confidence": p.confidence,
                    "projected_low": p.projected_low,
                    "projected_high": p.projected_high,
                    "tags": p.tags,
                    "new_top_setup": p.new_top_setup,
                    "bounce_risk": p.bounce_risk,
                    "tossed": p.tossed,
                    "proj_mid": p.proj_mid,
                    "spread": p.spread,
                    "cycle_priority": p.cycle_priority,
                    "sheets_rank": p.sheets_rank,
                    "tie_break_used": p.tie_break_used,
                    "explain": p.explain,
                }
                for p in projections
            ],
        }
        api_post(f"/predictions/save", json=payload, timeout=10)
    except Exception:
        pass  # best-effort; don't block the UI


def _mark_engine_run(session_id: str, race_number: int,
                     projections=None, audit=None, bias=None):
    """Record engine outputs for a session/race as the canonical source of truth.

    Stores full projections, audit, and bias per race so that downstream pages
    (Bet Builder, Dual Mode, Big Race Mode) can read from session_state instead
    of re-fetching from the API.
    """
    store = st.session_state.setdefault("engine_outputs_by_session", {})
    entry = store.setdefault(session_id, {
        "races_run": set(), "race_outputs": {}, "timestamp": None,
    })
    entry["races_run"].add(race_number)
    if projections is not None:
        entry.setdefault("race_outputs", {})[race_number] = {
            "projections": projections,
            "audit": audit,
            "bias": bias,
        }
    entry["timestamp"] = datetime.now().isoformat()
    st.session_state["current_session_id"] = session_id


def _run_engine_full_card(session_id: str, session_meta: dict):
    """Run the engine on ALL races for a session, persist, and update session_state.

    Returns True if at least one race produced projections, False otherwise.
    """
    track = session_meta.get('track') or session_meta.get('track_name', '')
    date = session_meta.get('date') or session_meta.get('race_date', '')
    is_session = 'session_id' in session_meta
    has_secondary = session_meta.get('has_secondary', False)

    # Load horse data
    try:
        if is_session and has_secondary:
            h_resp = requests.get(
                f"{API_BASE_URL}/sessions/{session_id}/races",
                params={"source": "merged"}, timeout=30,
            )
            all_horses = h_resp.json().get("race_data", {}).get("horses", []) if h_resp.ok else []
        else:
            h_resp = api_get(f"/races/{session_id}/horses", timeout=30)
            all_horses = h_resp.json().get("horses", []) if h_resp.ok else []
    except Exception:
        all_horses = []

    if not all_horses:
        return False

    # Group by race
    race_groups: dict = {}
    for h in all_horses:
        rn = h.get('race_number', 0) or 0
        race_groups.setdefault(rn, []).append(h)

    engine = HandicappingEngine()
    bias = BiasInput()  # default bias (neutral)
    any_produced = False
    progress = st.progress(0)
    race_keys = sorted(race_groups.keys())

    for idx, rn in enumerate(race_keys):
        rh = race_groups[rn]
        if not rh:
            progress.progress((idx + 1) / len(race_keys))
            continue
        projs, audit = _run_race_projections(engine, rh, bias, [], rn)
        if projs:
            _persist_predictions(session_id, track, date, rn, projs)
            _mark_engine_run(session_id, rn, projections=projs, audit=audit, bias=bias)
            any_produced = True
        progress.progress((idx + 1) / len(race_keys))

    progress.empty()
    return any_produced


# ---------------------------------------------------------------------------
# Shared display helpers (used by Engine + Dashboard)
# ---------------------------------------------------------------------------

def _render_top_picks(display_projections, race_number, key_prefix="eng"):
    """Render top 3 pick cards for a race."""
    top3 = display_projections[:3]
    st.subheader(f"Top 3 Picks \u2014 Race {race_number}")
    for rank, p in enumerate(top3, 1):
        col1, col2, col3 = st.columns([1, 2, 3])
        badge = ""
        if p.new_top_setup:
            badge = " \u2b50\u2b50" if p.new_top_confidence >= 75 else " \u2b50"
        if p.bounce_risk:
            badge += " \u26a0\ufe0f"
        with col1:
            st.metric(label=f"#{rank}", value=p.name)
        with col2:
            st.markdown(
                f"**Post {p.post}** | {p.style} | {p.projection_type}  \n"
                f"Fig {p.projected_low:.1f}\u2013{p.projected_high:.1f} | "
                f"Conf {p.confidence:.0%}"
            )
        with col3:
            if badge:
                st.markdown(f"**Tags:** {badge.strip()}")
            st.caption(p.summary)
            if p.new_top_setup:
                st.caption(
                    f"New Top Setup: {p.new_top_setup_type} "
                    f"({p.new_top_confidence}%) \u2014 {p.new_top_explanation}"
                )


def _render_ranked_table(display_projections, race_horses, show_odds,
                         odds_data, odds_by_post, race_num, key_prefix="eng"):
    """Render the full ranked table with optional odds column."""
    is_brisnet_data = any(
        h.get('figure_source') == 'brisnet' or h.get('quickplay_positive')
        for h in race_horses
    )

    st.subheader("Full Ranked Table")
    if show_odds:
        odds_found = sum(
            1 for p in display_projections
            if _lookup_odds(race_num, p.name, p.post, odds_data, odds_by_post)
            and _lookup_odds(race_num, p.name, p.post, odds_data, odds_by_post).get("odds_decimal") is not None
        )
        st.caption(f"Odds coverage: {odds_found}/{len(display_projections)} horses")

    rows = []
    for p in display_projections:
        rank_str = f"{p.sheets_rank}" if hasattr(p, 'sheets_rank') and p.sheets_rank else ""
        if hasattr(p, 'tie_break_used') and p.tie_break_used:
            rank_str += "*"
        row = {
            'Rank': rank_str,
            'Horse': p.name,
            'Post': p.post,
            'Style': p.style,
            'Cycle': p.projection_type,
            'Setup': '',
            'Proj Mid': f"{p.proj_mid:.1f}" if hasattr(p, 'proj_mid') else '',
            'Proj Low': f"{p.projected_low:.1f}",
            'Proj High': f"{p.projected_high:.1f}",
            'Confidence': f"{p.confidence:.0%}",
            'Tags': ', '.join(p.tags) if p.tags else '-',
            'Tie-Break': f"{p.bias_score:.1f}",
            'Summary': p.summary,
        }
        if p.new_top_setup:
            stars = "\u2b50\u2b50" if p.new_top_confidence >= 75 else "\u2b50"
            row['Setup'] = f"{stars} {p.new_top_setup_type} ({p.new_top_confidence}%)"
        elif p.bounce_risk:
            row['Setup'] = "\u26a0\ufe0f BOUNCE RISK"
        if is_brisnet_data:
            matching = [h for h in race_horses if h.get('horse_name') == p.name]
            if matching:
                row['Prime Power'] = matching[0].get('prime_power', '')
        if show_odds:
            o = _lookup_odds(race_num, p.name, p.post, odds_data, odds_by_post)
            row['Odds (ML)'] = _fmt_odds_display(o)
            if not o or o.get("odds_decimal") is None:
                existing = row.get('Tags', '-')
                row['Tags'] = f"{existing}, NO_ODDS" if existing != '-' else "NO_ODDS"
        rows.append(row)

    render_styled_table(pd.DataFrame(rows))


def _render_analysis_expanders(sorted_by_bias, audit, race_horses, key_prefix="eng"):
    """Render QuickPlay, enrichment, setups, toss, audit, and charts expanders."""
    is_brisnet_data = any(
        h.get('figure_source') == 'brisnet' or h.get('quickplay_positive')
        for h in race_horses
    )

    # --- QuickPlay Comments (BRISNET only) ---
    if is_brisnet_data:
        with st.expander("QuickPlay Comments", expanded=False):
            for h in race_horses:
                name = h.get('horse_name', 'Unknown')
                qp_pos = h.get('quickplay_positive', [])
                qp_neg = h.get('quickplay_negative', [])
                pp = h.get('prime_power', '')
                if qp_pos or qp_neg:
                    st.markdown(f"**{name}** (Prime Power: {pp})")
                    for comment in qp_pos:
                        st.markdown(f"  :green[+ {comment}]")
                    for comment in qp_neg:
                        st.markdown(f"  :red[- {comment}]")
                    st.markdown("---")

    # --- Enrichment Details ---
    has_enrichment = any(h.get('enrichment_source') == 'both' for h in race_horses)
    if has_enrichment:
        with st.expander("Enrichment Details", expanded=False):
            for h in race_horses:
                if h.get('enrichment_source') != 'both':
                    continue
                name = h.get('horse_name', 'Unknown')
                st.markdown(f"**{name}**")
                trainer = h.get('trainer', '')
                trainer_stats = h.get('trainer_stats', '')
                jockey = h.get('jockey', '')
                jockey_stats = h.get('jockey_stats', '')
                life_rec = h.get('life_record', '')
                life_earn = h.get('life_earnings', '')
                st.markdown(
                    f"Trainer: {trainer} ({trainer_stats}) | "
                    f"Jockey: {jockey} ({jockey_stats}) | "
                    f"Life: {h.get('life_starts', 0)} starts, {life_rec} | "
                    f"Earnings: {'$' + str(life_earn) if life_earn else 'N/A'}"
                )
                workouts = h.get('workouts', [])
                if workouts:
                    wo_strs = [
                        f"{w.get('date','')} {w.get('track','')} "
                        f"{w.get('distance','')} {w.get('time','')}"
                        for w in workouts[:3]
                    ]
                    st.caption("Workouts: " + " | ".join(wo_strs))

    # --- New Top Setups & Bounce Risk ---
    new_top_horses = [p for p in sorted_by_bias if p.new_top_setup]
    bounce_risk_horses = [p for p in sorted_by_bias if p.bounce_risk and not p.new_top_setup]
    if new_top_horses or bounce_risk_horses:
        with st.expander("New Top Setups & Bounce Risk", expanded=True):
            if new_top_horses:
                st.markdown("**New Top Setup Candidates:**")
                for p in new_top_horses:
                    stars = "\u2b50\u2b50" if p.new_top_confidence >= 75 else "\u2b50"
                    st.markdown(
                        f"- **{p.name}** {stars} | "
                        f"Type: {p.new_top_setup_type} | "
                        f"Confidence: {p.new_top_confidence}% | "
                        f"{p.new_top_explanation}"
                    )
            if bounce_risk_horses:
                st.markdown("**Bounce Risk (ran big new top last out):**")
                for p in bounce_risk_horses:
                    st.markdown(
                        f"- **{p.name}** \u26a0\ufe0f | "
                        f"Cycle: {p.projection_type} | "
                        f"Fig: {p.projected_low:.1f}\u2013{p.projected_high:.1f}"
                    )

    # --- TOSS List ---
    tossed_horses = [p for p in sorted_by_bias if p.tossed]
    if tossed_horses:
        with st.expander("TOSS List (eliminate)", expanded=True):
            for p in tossed_horses:
                reasons_str = "; ".join(p.toss_reasons)
                st.markdown(f"- **{p.name}** \u2014 {reasons_str}")

    # --- AUDIT ---
    if audit:
        with st.expander("AUDIT", expanded=True):
            st.markdown(
                f"**Cycle-best:** {audit.cycle_best} | "
                f"**Raw-best:** {audit.raw_best}"
            )
            if audit.cycle_vs_raw_match:
                st.success("Cycle-best and raw-best are the same horse.")
            else:
                st.info(f"Cycle-best ({audit.cycle_best}) differs from raw-best ({audit.raw_best}).")

            if audit.bounce_candidates:
                st.markdown("**Bounce candidates:** " + ", ".join(audit.bounce_candidates))

            if audit.first_time_surface:
                st.markdown("**First-time-surface horses:**")
                for entry in audit.first_time_surface:
                    st.markdown(
                        f"- **{entry['name']}** \u2014 penalty: {entry['penalty']}, "
                        f"conf adj: {entry['conf_adj']}"
                    )

            if audit.toss_list:
                st.markdown("**Tossed:**")
                for entry in audit.toss_list:
                    reasons_str = "; ".join(entry['reasons'])
                    st.markdown(f"- **{entry['name']}** \u2014 {reasons_str}")

    # --- Charts ---
    with st.expander("Charts", expanded=False):
        fig = px.bar(
            x=[p.name for p in sorted_by_bias],
            y=[p.confidence for p in sorted_by_bias],
            labels={'x': 'Horse', 'y': 'Confidence'},
            title="Projection Confidence",
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        for p in sorted_by_bias:
            fig2.add_trace(go.Bar(
                name=p.name,
                x=[p.name],
                y=[p.projected_high - p.projected_low],
                base=[p.projected_low],
                text=[f"{p.projected_low:.1f}-{p.projected_high:.1f}"],
                textposition='outside',
            ))
        fig2.update_layout(
            title="Projected Figure Range",
            yaxis_title="Speed Figure",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Ranking Explain ---
    if sorted_by_bias and hasattr(sorted_by_bias[0], 'explain') and sorted_by_bias[0].explain:
        with st.expander("Ranking Explain", expanded=False):
            for p in sorted_by_bias:
                rank_label = f"#{p.sheets_rank}" if hasattr(p, 'sheets_rank') else ""
                tie_marker = " (tie-break)" if hasattr(p, 'tie_break_used') and p.tie_break_used else ""
                st.markdown(f"**{rank_label} {p.name}**{tie_marker}")
                if hasattr(p, 'explain') and p.explain:
                    explain_rows = []
                    items = p.explain.items() if isinstance(p.explain, dict) else p.explain
                    for key, val in items:
                        explain_rows.append({"Factor": key, "Value": str(val)})
                    st.dataframe(pd.DataFrame(explain_rows), use_container_width=True, hide_index=True)
                st.markdown("---")


def _render_best_bets_table(best_bets, odds_data, odds_by_post, key_prefix="eng"):
    """Render best bets table + top 3 cards."""
    top_n = min(10, len(best_bets))
    st.caption(f"Showing top {top_n} of {len(best_bets)} entries")

    bb_show_odds = st.checkbox("Show Odds", value=True, key=f"bb_show_odds_{key_prefix}")
    if bb_show_odds:
        bb_odds_found = sum(
            1 for _rn, _p in best_bets[:top_n]
            if _lookup_odds(_rn, _p.name, _p.post, odds_data, odds_by_post)
            and _lookup_odds(_rn, _p.name, _p.post, odds_data, odds_by_post).get("odds_decimal") is not None
        )
        st.caption(f"Odds coverage: {bb_odds_found}/{top_n} horses")

    rows = []
    for rn, p in best_bets[:top_n]:
        score = _best_bet_score(p)
        row = {
            'Race': rn,
            'Horse': p.name,
            'Post': p.post,
            'Style': p.style,
            'Cycle': p.projection_type,
            'Setup': '',
            'Proj': f"{p.projected_low:.1f}\u2013{p.projected_high:.1f}",
            'Confidence': f"{p.confidence:.0%}",
            'Bias Score': f"{p.bias_score:.1f}",
            'Best Bet Score': f"{score:.1f}",
            'Tags': ', '.join(p.tags) if p.tags else '-',
            'Summary': p.summary,
        }
        if p.new_top_setup:
            stars = "\u2b50\u2b50" if p.new_top_confidence >= 75 else "\u2b50"
            row['Setup'] = f"{stars} {p.new_top_setup_type} ({p.new_top_confidence}%)"
        elif p.bounce_risk:
            row['Setup'] = "\u26a0\ufe0f BOUNCE RISK"
        if bb_show_odds:
            o = _lookup_odds(rn, p.name, p.post, odds_data, odds_by_post)
            row['Odds (ML)'] = _fmt_odds_display(o)
            if not o or o.get("odds_decimal") is None:
                existing = row.get('Tags', '-')
                row['Tags'] = f"{existing}, NO_ODDS" if existing != '-' else "NO_ODDS"
        rows.append(row)

    df = pd.DataFrame(rows)
    styled = df.style
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Highlight top 3
    st.subheader("Top 3 Best Bets")
    for rank, (rn, p) in enumerate(best_bets[:3], 1):
        score = _best_bet_score(p)
        col1, col2, col3 = st.columns([1, 2, 3])
        with col1:
            badge = ""
            if p.new_top_setup:
                badge = " \u2b50\u2b50" if p.new_top_confidence >= 75 else " \u2b50"
            if p.bounce_risk:
                badge += " \u26a0\ufe0f"
            st.metric(f"#{rank}", f"R{rn} {p.name}{badge}")
        with col2:
            st.markdown(
                f"**Cycle** {p.projection_type} | "
                f"**Fig** {p.projected_low:.1f}\u2013{p.projected_high:.1f} | "
                f"**Conf** {p.confidence:.0%} | "
                f"**Score** {score:.1f}"
            )
            if p.new_top_setup:
                st.caption(
                    f"New Top Setup (4yo): {p.new_top_setup_type} "
                    f"({p.new_top_confidence}%) \u2014 {p.new_top_explanation}"
                )
        with col3:
            tags_str = ", ".join(p.tags) if p.tags else "none"
            st.markdown(f"Tags: {tags_str}")
            st.caption(p.summary)



# ---------------------------------------------------------------------------
# Bet Commander
# ---------------------------------------------------------------------------

def bet_commander_page():
    """Bet Commander — API-driven card analysis, recommendations, and ticket building.

    Flow: Streamlit UI → FastAPI endpoints → DB predictions
    Uses: /bets/recommend, /bets/daily-double, /bets/multi-race, /bets/single-race
    """
    from bet_builder import commander_slip_to_text, commander_slip_to_json

    st.header("Bet Commander")
    st.caption("Full-card betting: auto-recommendations (DD/EX/TRI/P3), adjustable A/B/C counts, export.")

    # --- Session selector ---
    try:
        sr = api_get("/sessions", timeout=10)
        sessions = sr.json().get("sessions", []) if sr.ok else []
    except Exception:
        sessions = []

    if not sessions:
        st.info("No sessions found. Upload and run the engine first.")
        return

    def _cmd_label(x):
        name = x.get('primary_pdf_filename') or x.get('original_filename', 'unknown')
        trk = x.get('track') or x.get('track_name', '')
        dt = x.get('date') or x.get('race_date', '')
        return f"{name} | {trk} | {dt}"

    cmd_sel = st.selectbox("Session:", options=sessions, format_func=_cmd_label, key="cmd_session")
    if not cmd_sel:
        return

    cmd_sid = cmd_sel.get('session_id') or cmd_sel.get('id')
    cmd_track = cmd_sel.get('track') or cmd_sel.get('track_name', '')
    cmd_date = cmd_sel.get('date') or cmd_sel.get('race_date', '')

    # Track session changes — clear slip when session changes
    if st.session_state.get("commander_session_id") != cmd_sid:
        st.session_state["commander_slip"] = []
        st.session_state["commander_rec_data"] = None
        st.session_state["commander_session_id"] = cmd_sid

    # --- Settings row ---
    st.divider()
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        bankroll = st.number_input("Bankroll ($)", min_value=50, max_value=50000,
                                   value=1000, step=50, key="cmd_bankroll")
    with sc2:
        base_wager = st.number_input("Base Wager ($)", min_value=1, max_value=20,
                                     value=2, step=1, key="cmd_base_wager")
    with sc3:
        total_budget = st.number_input("Total Budget ($)", min_value=10, max_value=5000,
                                       value=200, step=10, key="cmd_total_budget")

    # --- Per-bet-type A/B/C counts ---
    st.markdown("**Counts per bet type**")
    ct1, ct2, ct3, ct4 = st.columns(4)
    with ct1:
        st.caption("Daily Double")
        dd_a = st.number_input("DD A", min_value=1, max_value=3, value=1, key="cmd_dd_a")
        dd_b = st.number_input("DD B", min_value=1, max_value=4, value=2, key="cmd_dd_b")
        dd_c = st.number_input("DD C", min_value=0, max_value=3, value=0, key="cmd_dd_c")
    with ct2:
        st.caption("Exacta")
        ex_a = st.number_input("EX A", min_value=1, max_value=2, value=1, key="cmd_ex_a")
        ex_b = st.number_input("EX B", min_value=1, max_value=4, value=3, key="cmd_ex_b")
    with ct3:
        st.caption("Trifecta")
        tri_a = st.number_input("TRI 1st", min_value=1, max_value=2, value=1, key="cmd_tri_a")
        tri_b = st.number_input("TRI 2nd", min_value=1, max_value=4, value=2, key="cmd_tri_b")
        tri_c = st.number_input("TRI 3rd", min_value=0, max_value=4, value=2, key="cmd_tri_c")
    with ct4:
        st.caption("Pick 3")
        p3_a = st.number_input("P3 A", min_value=1, max_value=3, value=1, key="cmd_p3_a")
        p3_b = st.number_input("P3 B", min_value=1, max_value=4, value=3, key="cmd_p3_b")
        p3_c = st.number_input("P3 C", min_value=0, max_value=6, value=5, key="cmd_p3_c")

    # =====================================================================
    # [A] RECOMMEND PLAYS — calls POST /bets/recommend
    # =====================================================================
    st.divider()
    if st.button("Recommend Plays", type="primary", key="cmd_recommend"):
        payload = {
            "session_id": cmd_sid, "track": cmd_track, "race_date": cmd_date,
            "bankroll": bankroll,
        }
        try:
            resp = api_post("/bets/recommend", json=payload, timeout=30)
            if resp.ok:
                st.session_state["commander_rec_data"] = resp.json()
            elif resp.status_code == 404:
                st.session_state["commander_rec_data"] = None
                st.warning("No predictions found in the database for this session.")
                st.markdown("**Run the engine first**, then come back here.")
                if st.button("Run Engine + Save Predictions", key="cmd_run_engine_save"):
                    with st.spinner("Running engine on full card..."):
                        ok = _run_engine_full_card(cmd_sid, cmd_sel)
                    if ok:
                        st.success("Engine complete and predictions saved. Click **Recommend Plays** again.")
                        st.rerun()
                    else:
                        st.error("Engine produced no projections. Check session data.")
                return
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    rec_data = st.session_state.get("commander_rec_data")
    if not rec_data:
        st.info("Click **Recommend Plays** to analyze the card and get recommendations.")
        return

    analyses = rec_data.get("analyses", {})
    recommendations = rec_data.get("recommendations", [])

    # --- Card Summary ---
    st.subheader("Card Summary")

    race_count = len(analyses)
    singles_count = sum(1 for a in analyses.values() if a.get("has_true_single"))
    chaos_count = sum(1 for a in analyses.values() if a.get("chaos_index", 0) > 0.55)
    odds_coverage = sum(1 for a in analyses.values() if a.get("top_overlay") is not None) / max(race_count, 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Races", race_count)
    m2.metric("True Singles", singles_count)
    m3.metric("Chaos Races", chaos_count)
    m4.metric("Odds Coverage", f"{odds_coverage:.0%}")

    card_rows = []
    for rn in sorted(analyses.keys(), key=lambda x: int(x)):
        a = analyses[rn]
        overlay = a.get("top_overlay")
        card_rows.append({
            "Race": int(rn),
            "Grade": a.get("grade", "?"),
            "Edge": f"{a.get('edge_score', 0):.0f}",
            "Chaos": f"{a.get('chaos_index', 0):.2f}",
            "Single": "Y" if a.get("has_true_single") else "",
            "Top Horse": a.get("top_horse", ""),
            "Cycle": a.get("top_cycle", ""),
            "Conf": f"{a.get('top_confidence', 0):.0%}",
            "Overlay": f"{overlay:.2f}" if overlay else "-",
            "Quality": f"{a.get('figure_quality_pct', 0):.0%}",
            "Field": a.get("field_size", 0),
        })
    if card_rows:
        st.dataframe(pd.DataFrame(card_rows), use_container_width=True, hide_index=True)

    # =====================================================================
    # [B] RECOMMENDED PLAYS
    # =====================================================================
    st.subheader("Recommended Plays")

    if not recommendations:
        st.info("No plays recommended for this card. Build custom tickets below.")
    else:
        rec_rows = []
        for rec in recommendations:
            race_str = "-".join(str(r) for r in rec.get("races", []))
            rec_rows.append({
                "Type": rec.get("bet_type", ""),
                "Race(s)": f"R{race_str}",
                "Conf": f"{rec.get('confidence', 0):.0%}",
                "Key Horse": rec.get("key_horse", "") or "",
                "Reason": rec.get("reason_text", ""),
            })
        st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

        # Add to Slip buttons
        add_cols = st.columns(min(len(recommendations) + 1, 6))
        for i, rec in enumerate(recommendations):
            if i < len(add_cols) - 1:
                with add_cols[i]:
                    race_str = "-".join(str(r) for r in rec.get("races", []))
                    if st.button(f"Add {rec['bet_type']} R{race_str}", key=f"cmd_add_{i}"):
                        import uuid
                        new_entry = {
                            "id": str(uuid.uuid4())[:8],
                            "bet_type": rec["bet_type"],
                            "races": rec["races"],
                            "base_wager": base_wager,
                            "suggested_counts": rec.get("suggested_counts", {}),
                            "computed_tickets": [],
                            "total_cost": 0,
                            "built": False,
                        }
                        st.session_state.setdefault("commander_slip", []).append(new_entry)
                        st.rerun()
        with add_cols[-1]:
            if st.button("Add All", key="cmd_add_all"):
                import uuid
                for rec in recommendations:
                    new_entry = {
                        "id": str(uuid.uuid4())[:8],
                        "bet_type": rec["bet_type"],
                        "races": rec["races"],
                        "base_wager": base_wager,
                        "suggested_counts": rec.get("suggested_counts", {}),
                        "computed_tickets": [],
                        "total_cost": 0,
                        "built": False,
                    }
                    st.session_state.setdefault("commander_slip", []).append(new_entry)
                st.rerun()

    # =====================================================================
    # [C] BET SLIP — per-leg A/B/C overrides + Build Tickets via API
    # =====================================================================
    st.divider()
    st.subheader("Bet Slip")

    slip = st.session_state.get("commander_slip", [])

    if not slip:
        st.info("Slip is empty. Add plays from recommendations above.")
    else:
        total_cost = sum(e.get("total_cost", 0) for e in slip)
        budget_pct = total_cost / bankroll if bankroll > 0 else 0

        bc1, bc2 = st.columns([3, 1])
        with bc1:
            st.progress(min(budget_pct, 1.0))
        with bc2:
            st.markdown(f"**${total_cost:.2f}** / ${bankroll:.0f}")

        if total_cost > bankroll:
            st.error(f"Over budget by ${total_cost - bankroll:.2f}!")

        to_remove = []
        for idx, entry in enumerate(slip):
            eid = entry.get("id", str(idx))
            bt = entry.get("bet_type", "?")
            races = entry.get("races", [])
            race_str = "-".join(str(r) for r in races)
            entry_cost = entry.get("total_cost", 0)
            built = entry.get("built", False)
            status = f"${entry_cost:.2f}" if built else "not built"

            with st.expander(f"{bt} R{race_str} — {status}", expanded=not built):
                # Per-leg A/B/C override controls
                suggested = entry.get("suggested_counts", {})
                for rn in races:
                    rn_key = str(rn)
                    sug = suggested.get(rn_key, {})
                    # Use bet-type defaults, overridden by suggested
                    if bt == "DD":
                        def_a, def_b, def_c = dd_a, dd_b, dd_c
                    elif bt == "EXACTA":
                        def_a, def_b, def_c = ex_a, ex_b, 0
                    elif bt == "TRIFECTA":
                        def_a, def_b, def_c = tri_a, tri_b, tri_c
                    elif bt == "PICK3":
                        def_a, def_b, def_c = p3_a, p3_b, p3_c
                    elif bt == "WIN":
                        def_a, def_b, def_c = 1, 0, 0
                    else:
                        def_a, def_b, def_c = 1, 3, 5

                    st.markdown(f"**Race {rn}**")
                    lc1, lc2, lc3 = st.columns(3)
                    with lc1:
                        entry.setdefault("_a", {})[rn] = st.number_input(
                            "A", min_value=1, max_value=5,
                            value=sug.get("a_count", def_a),
                            key=f"cmd_a_{eid}_{rn}")
                    with lc2:
                        entry.setdefault("_b", {})[rn] = st.number_input(
                            "B", min_value=0, max_value=6,
                            value=sug.get("b_count", def_b),
                            key=f"cmd_b_{eid}_{rn}")
                    with lc3:
                        entry.setdefault("_c", {})[rn] = st.number_input(
                            "C", min_value=0, max_value=6,
                            value=sug.get("c_count", def_c),
                            key=f"cmd_c_{eid}_{rn}")

                # Build Tickets button — calls API
                if st.button("Build Tickets", type="primary", key=f"cmd_build_{eid}"):
                    _commander_build_via_api(
                        entry, cmd_sid, cmd_track, cmd_date, base_wager,
                    )
                    st.rerun()

                # Show built tickets
                for t in entry.get("computed_tickets", []):
                    rat = t.get("rationale", "") if isinstance(t, dict) else ""
                    st.markdown(f"- {rat}")

                if st.button("Remove", key=f"cmd_remove_{eid}"):
                    to_remove.append(idx)

        if to_remove:
            for ri in sorted(to_remove, reverse=True):
                slip.pop(ri)
            st.session_state["commander_slip"] = slip
            st.rerun()

    # =====================================================================
    # [D] OUTPUT — Build All + Export + Save
    # =====================================================================
    st.divider()
    st.subheader("Output")

    slip = st.session_state.get("commander_slip", [])
    total_cost = sum(e.get("total_cost", 0) for e in slip)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        if slip and st.button("Build All Tickets", type="primary", key="cmd_build_all"):
            for entry in slip:
                if not entry.get("built"):
                    _commander_build_via_api(
                        entry, cmd_sid, cmd_track, cmd_date, base_wager,
                    )
            st.session_state["commander_slip"] = slip
            st.rerun()
    with d2:
        if slip:
            text_export = commander_slip_to_text(slip)
            st.download_button("Download Text", text_export, "bet_commander.txt",
                               "text/plain", key="cmd_dl_txt")
    with d3:
        if slip:
            json_export = commander_slip_to_json(slip)
            st.download_button("Download JSON", json_export, "bet_commander.json",
                               "application/json", key="cmd_dl_json")
    with d4:
        if slip and st.button("Save to Database", key="cmd_save_db"):
            serialized = []
            for entry in slip:
                ser = {
                    "bet_type": entry.get("bet_type"),
                    "races": entry.get("races", []),
                    "base_wager": entry.get("base_wager", 2),
                    "total_cost": entry.get("total_cost", 0),
                    "tickets": entry.get("computed_tickets", []),
                }
                serialized.append(ser)
            payload = {
                "session_id": cmd_sid, "track": cmd_track, "race_date": cmd_date,
                "slip_entries": serialized, "total_cost": total_cost, "bankroll": bankroll,
            }
            try:
                resp = api_post("/bets/commander-save", json=payload, timeout=15)
                if resp.ok:
                    st.success(f"Saved! Plan ID: {resp.json().get('plan_id')}")
                else:
                    st.error(f"Save failed: {resp.text}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    if slip:
        st.caption(f"Slip: {len(slip)} plays | Total: ${total_cost:.2f} | Bankroll: ${bankroll:.0f}")


def _commander_build_via_api(entry, sid, track, race_date, base_wager):
    """Build tickets for a slip entry by calling the appropriate API endpoint.

    Mutates entry in place: sets computed_tickets, total_cost, built.
    """
    bt = entry.get("bet_type", "")
    races = entry.get("races", [])
    a_counts = entry.get("_a", {})
    b_counts = entry.get("_b", {})
    c_counts = entry.get("_c", {})

    tickets = []
    error = None

    try:
        if bt in ("WIN", "EXACTA", "TRIFECTA") and len(races) == 1:
            rn = races[0]
            payload = {
                "session_id": sid, "track": track, "race_date": race_date,
                "race_number": rn, "bet_type": bt,
                "budget": base_wager * 20,
                "a_count": a_counts.get(rn, 1),
                "b_count": b_counts.get(rn, 2),
                "c_count": c_counts.get(rn, 0),
                "save": False,
            }
            resp = api_post("/bets/single-race", json=payload, timeout=15)
            if resp.ok:
                plan = resp.json().get("plan", {})
                tickets = plan.get("tickets", [])
            else:
                error = resp.text

        elif bt == "DD" and len(races) == 2:
            # Use the first leg's counts as the strategy
            payload = {
                "session_id": sid, "track": track, "race_date": race_date,
                "start_race": races[0],
                "budget": base_wager * 24,
                "a_count": a_counts.get(races[0], 1),
                "b_count": b_counts.get(races[0], 2),
                "c_count": c_counts.get(races[0], 0),
                "save": False,
            }
            resp = api_post("/bets/daily-double", json=payload, timeout=15)
            if resp.ok:
                plan = resp.json().get("plan", {})
                # Convert DD tickets to common format
                for t in plan.get("tickets", []):
                    tickets.append({
                        "bet_type": "DD",
                        "selections": t.get("leg1", []) + t.get("leg2", []),
                        "cost": t.get("cost", 0),
                        "rationale": t.get("reason", ""),
                        "details": {"leg1": t.get("leg1"), "leg2": t.get("leg2")},
                    })
                if plan.get("passed"):
                    tickets = [{"bet_type": "DD", "selections": [], "cost": 0,
                                "rationale": f"PASS — {plan.get('pass_reason', '')}",
                                "details": {}}]
            else:
                error = resp.text

        elif bt in ("PICK3", "PICK6"):
            payload = {
                "session_id": sid, "track": track, "race_date": race_date,
                "start_race": races[0], "bet_type": bt,
                "budget": base_wager * len(races) * 16,
                "a_count": a_counts.get(races[0], 1),
                "b_count": b_counts.get(races[0], 3),
                "c_count": c_counts.get(races[0], 5),
                "save": False,
            }
            resp = api_post("/bets/multi-race", json=payload, timeout=15)
            if resp.ok:
                plan = resp.json().get("plan", {})
                legs = plan.get("legs", [])
                cost = plan.get("cost", 0)
                combos = plan.get("combinations", 0)
                leg_summary = " / ".join(
                    f"R{lg.get('race_number')}({lg.get('horse_count', 0)})"
                    for lg in legs
                )
                tickets.append({
                    "bet_type": bt,
                    "selections": [h for lg in legs for h in lg.get("horses", [])],
                    "cost": cost,
                    "rationale": f"{bt} R{races[0]}-R{races[-1]}: {leg_summary} = {combos} combos ${cost:.0f}",
                    "details": {"legs": legs},
                })
            else:
                error = resp.text

    except Exception as e:
        error = str(e)

    if error:
        tickets = [{"bet_type": bt, "selections": [], "cost": 0,
                     "rationale": f"ERROR: {error}", "details": {}}]

    entry["computed_tickets"] = tickets
    entry["total_cost"] = sum(t.get("cost", 0) for t in tickets)
    entry["built"] = True


def dashboard_page():
    st.header("Dashboard")
    st.caption("Unified workflow: upload \u2192 analyze \u2192 bet \u2192 evaluate.")

    # ===== Section A: Upload =====
    with st.expander("Upload", expanded=False):
        ucol1, ucol2 = st.columns(2)
        with ucol1:
            st.markdown("**Primary (Ragozin / BRISNET)**")
            primary_file = st.file_uploader(
                "Upload primary PDF", type=["pdf"], key="dash_primary_upload")
            if primary_file and st.button("Upload Primary", key="dash_btn_primary"):
                with st.spinner("Uploading primary..."):
                    try:
                        files = {"file": (primary_file.name, primary_file.getvalue(), "application/pdf")}
                        resp = api_post(f"/upload_primary", files=files, timeout=60)
                        if resp.ok:
                            data = resp.json()
                            sid = data.get("session_id", "")
                            st.success(f"Session: {sid}")
                            st.session_state["dash_session_id"] = sid
                        else:
                            st.error(f"Upload error: {resp.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with ucol2:
            st.markdown("**Secondary (add to existing session)**")
            sec_file = st.file_uploader(
                "Upload secondary PDF", type=["pdf"], key="dash_sec_upload")
            dash_sec_sid = st.text_input("Session ID:", key="dash_sec_sid",
                                          value=st.session_state.get("dash_session_id", ""))
            if sec_file and dash_sec_sid and st.button("Upload Secondary", key="dash_btn_sec"):
                with st.spinner("Uploading secondary..."):
                    try:
                        files = {"file": (sec_file.name, sec_file.getvalue(), "application/pdf")}
                        resp = requests.post(
                            f"{API_BASE_URL}/upload_secondary",
                            files=files,
                            params={"session_id": dash_sec_sid},
                            timeout=60,
                        )
                        if resp.ok:
                            st.success("Secondary merged.")
                        else:
                            st.error(f"Upload error: {resp.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Quick link to Engine
        if st.session_state.get("dash_session_id"):
            if st.button("Open in Engine", key="dash_open_engine"):
                st.session_state["active_session_id"] = st.session_state["dash_session_id"]
                st.session_state["_nav_target"] = "Engine"
                st.rerun()

    # ===== Section B: Daily Best WIN Bets =====
    with st.expander("Daily Best WIN Bets", expanded=True):
        from datetime import date as _date
        dw_date = st.date_input("Race Date", value=_date.today(), key="dash_dw_date")
        dw_date_str = dw_date.strftime("%m/%d/%Y") if dw_date else ""

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            dw_bankroll = st.number_input("Bankroll ($)", value=1000, min_value=100, step=100, key="dash_dw_bankroll")
        with dc2:
            dw_min_overlay = st.number_input("Min overlay", value=1.10, min_value=1.0, max_value=2.0,
                                              step=0.05, key="dash_dw_min_overlay")
        with dc3:
            dw_max_bets = st.number_input("Max bets", value=10, min_value=5, max_value=15, step=1, key="dash_dw_max_bets")

        if st.button("Generate", type="primary", key="dash_btn_daily_wins"):
            if not dw_date_str:
                st.warning("Select a race date.")
            else:
                with st.spinner("Building daily WIN bets..."):
                    payload = {
                        "race_date": dw_date_str,
                        "bankroll": dw_bankroll,
                        "risk_profile": "standard",
                        "max_risk_per_day_pct": 6.0,
                        "min_confidence": 0.65,
                        "min_odds_a": 2.0,
                        "paper_mode": True,
                        "max_bets": dw_max_bets,
                        "min_overlay": dw_min_overlay,
                        "save": True,
                    }
                    try:
                        resp = api_post(f"/bets/daily-wins", json=payload, timeout=30)
                        if resp.ok:
                            st.session_state["dash_dw_result"] = resp.json()
                        elif resp.status_code == 404:
                            st.warning("No predictions found. Run the engine on sessions first.")
                        else:
                            st.error(f"Error: {resp.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        dw_result = st.session_state.get("dash_dw_result")
        if dw_result:
            candidates = dw_result.get("candidates", [])
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Risk", f"${dw_result.get('total_risk', 0):.0f}")
            m2.metric("Bets", len(candidates))
            m3.metric("Tracks", len(dw_result.get("tracks", [])))

            if candidates:
                rows = []
                for c in candidates:
                    rows.append({
                        "Track": c.get("track", ""),
                        "Race": c.get("race_number", ""),
                        "Horse": c.get("horse_name", ""),
                        "Cycle": c.get("projection_type", ""),
                        "Conf": f"{c.get('confidence', 0):.0%}",
                        "Odds": f"{c['odds_decimal']:.1f}-1" if c.get("odds_decimal") else "-",
                        "FairOdds": f"{c['fair_odds']:.1f}-1" if c.get("fair_odds") else "-",
                        "Overlay": f"{c['overlay']:.2f}x" if c.get("overlay") else "-",
                        "Stake": f"${c.get('stake', 0):.0f}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # CSV export
                csv_data = "track,race,horse,cycle,confidence,odds,fair_odds,overlay,stake\n"
                for c in candidates:
                    csv_data += (
                        f"{c.get('track','')},{c.get('race_number','')},{c.get('horse_name','')}"
                        f",{c.get('projection_type','')},{c.get('confidence',0):.2f}"
                        f",{c.get('odds_decimal','')},{c.get('fair_odds','')}"
                        f",{c.get('overlay','')},{c.get('stake',0):.0f}\n"
                    )
                st.download_button("Export CSV", csv_data, "daily_wins.csv", "text/csv", key="dash_dw_csv")
            else:
                st.info("No qualifying bets found.")

    # ===== Section C: Engine Drilldown =====
    with st.expander("Engine Drilldown", expanded=True):
        engine = HandicappingEngine()

        # Session selector
        try:
            sess_resp = api_get(f"/races", timeout=15)
            races = sess_resp.json().get("races", []) if sess_resp.ok else []
        except Exception:
            races = []

        if not races:
            st.info("No parsed sessions. Upload a PDF first.")
        else:
            def _dash_session_label(x):
                name = x.get('primary_pdf_filename') or x.get('original_filename', 'unknown')
                track = x.get('track') or x.get('track_name', '')
                date = x.get('date') or x.get('analysis_date', x.get('race_date', ''))
                return f"{name} | {track} | {date}"

            dash_sel = st.selectbox("Session:", options=races,
                                     format_func=_dash_session_label, key="dash_sess")
            if dash_sel:
                dash_sid = dash_sel.get('session_id') or dash_sel.get('id')
                is_session = 'session_id' in dash_sel

                # Fetch horses
                try:
                    if is_session and dash_sel.get('has_secondary'):
                        h_resp = requests.get(
                            f"{API_BASE_URL}/sessions/{dash_sid}/races",
                            params={"source": "merged"}, timeout=30)
                        dash_horses = h_resp.json().get("race_data", {}).get("horses", []) if h_resp.ok else []
                    else:
                        h_resp = api_get(f"/races/{dash_sid}/horses", timeout=30)
                        dash_horses = h_resp.json().get("horses", []) if h_resp.ok else []
                except Exception:
                    dash_horses = []

                if not dash_horses:
                    st.warning("No horse data in this session.")
                else:
                    # Fetch odds
                    _d_odds_key = f"_odds_{dash_sid}"
                    if _d_odds_key not in st.session_state:
                        _d_odds = {}
                        _d_odds_by_post = {}
                        try:
                            _o_resp = api_get(f"/odds/snapshots/{dash_sid}", timeout=10)
                            if _o_resp.ok:
                                for snap in _o_resp.json().get("snapshots", []):
                                    entry = {"odds_raw": snap.get("odds_raw", ""), "odds_decimal": snap.get("odds_decimal")}
                                    _d_odds[(snap["race_number"], snap["normalized_name"])] = entry
                                    if snap.get("post") is not None:
                                        _d_odds_by_post[(snap["race_number"], snap["post"])] = entry
                        except Exception:
                            pass
                        st.session_state[_d_odds_key] = (_d_odds, _d_odds_by_post)
                    _d_odds_data, _d_odds_by_post = st.session_state[_d_odds_key]

                    # Group by race
                    dash_race_groups = {}
                    for h in dash_horses:
                        rn = h.get('race_number', 0) or 0
                        dash_race_groups.setdefault(rn, []).append(h)

                    # Figure quality
                    _d_race_quality = {}
                    for rn, horses_in_race in dash_race_groups.items():
                        missing = 0
                        first_timers = 0
                        for h in horses_in_race:
                            lines = h.get('lines', [])
                            figs = [_extract_figure(ln) for ln in lines]
                            max_fig = max(figs) if figs else 0
                            is_stub = (len(lines) == 1 and not lines[0].get('raw_text', ''))
                            if is_stub:
                                first_timers += 1
                            elif not figs or max_fig == 0:
                                missing += 1
                        scoreable = len(horses_in_race) - first_timers
                        _d_race_quality[rn] = missing / scoreable if scoreable > 0 else 0

                    # Race summary
                    race_summary = []
                    for rn in sorted(dash_race_groups.keys()):
                        q = _d_race_quality.get(rn)
                        badge = f" {_quality_badge(q)}" if q is not None else ""
                        race_summary.append(f"R{rn}: {len(dash_race_groups[rn])}{badge}")
                    st.markdown(" | ".join(race_summary))

                    # Bias sliders
                    bc1, bc2, bc3, bc4, bc5 = st.columns(5)
                    with bc1:
                        d_e = st.slider("E %", 0, 100, 25, key="dash_e")
                    with bc2:
                        d_ep = st.slider("EP %", 0, 100, 25, key="dash_ep")
                    with bc3:
                        d_p = st.slider("P %", 0, 100, 25, key="dash_p")
                    with bc4:
                        d_s = st.slider("S %", 0, 100, 25, key="dash_s")
                    with bc5:
                        d_sf = st.checkbox("Speed favoring", key="dash_sf")

                    dash_bias = BiasInput(
                        e_pct=float(d_e), ep_pct=float(d_ep),
                        p_pct=float(d_p), s_pct=float(d_s),
                        speed_favoring=d_sf,
                    )

                    # Race selector
                    dash_race_options = sorted(dash_race_groups.keys())
                    dash_race = st.selectbox(
                        "Race:", options=dash_race_options,
                        format_func=lambda rn: f"Race {rn} ({len(dash_race_groups[rn])} horses)" if rn else "All",
                        key="dash_race_sel",
                    )

                    race_horses = dash_race_groups[dash_race]

                    if st.button("Run Projections", type="primary", key="dash_run_proj"):
                        projs, audit = _run_race_projections(engine, race_horses, dash_bias, [], dash_race)
                        if not projs:
                            st.warning("No projections produced.")
                        else:
                            st.session_state["dash_projections"] = projs
                            st.session_state["dash_proj_race"] = dash_race
                            st.session_state["dash_audit"] = audit
                            st.session_state["dash_last_computed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            # Persist
                            s_track = dash_sel.get('track') or dash_sel.get('track_name', '')
                            s_date = dash_sel.get('date') or dash_sel.get('race_date', '')
                            _persist_predictions(dash_sid, s_track, s_date, dash_race, projs)
                            _mark_engine_run(dash_sid, dash_race, projections=projs, audit=audit)

                    # Display projections
                    d_projs = st.session_state.get("dash_projections")
                    d_proj_race = st.session_state.get("dash_proj_race")
                    d_audit = st.session_state.get("dash_audit")
                    d_last = st.session_state.get("dash_last_computed")

                    if d_projs and d_proj_race == dash_race:
                        if d_last:
                            st.caption(f"Last computed at: {d_last}")
                        sorted_by_bias = sorted(d_projs, key=lambda p: p.bias_score, reverse=True)
                        _render_top_picks(sorted_by_bias, dash_race, key_prefix="dash")
                        show_odds = st.checkbox("Show Odds", value=True, key=f"dash_show_odds_{dash_race}")
                        _render_ranked_table(sorted_by_bias, race_horses, show_odds,
                                             _d_odds_data, _d_odds_by_post, dash_race,
                                             key_prefix="dash")
                        _render_analysis_expanders(sorted_by_bias, d_audit, race_horses,
                                                   key_prefix="dash")
                    else:
                        st.info("Click **Run Projections** to analyze this race.")

    # ===== Section D: Bet Builder =====
    with st.expander("Bet Builder", expanded=False):
        # Need a session with best bets
        try:
            sess_resp2 = api_get(f"/races", timeout=15)
            races2 = sess_resp2.json().get("races", []) if sess_resp2.ok else []
        except Exception:
            races2 = []

        if not races2:
            st.info("No sessions available. Upload a PDF first.")
        else:
            def _bb_label(x):
                name = x.get('primary_pdf_filename') or x.get('original_filename', 'unknown')
                track = x.get('track') or x.get('track_name', '')
                return f"{name} | {track}"

            bb_sel = st.selectbox("Session:", options=races2, format_func=_bb_label, key="dash_bb_sess")
            if bb_sel:
                bb_sid = bb_sel.get('session_id') or bb_sel.get('id')
                bb_track = bb_sel.get('track') or bb_sel.get('track_name', '')
                bb_date = bb_sel.get('date') or bb_sel.get('race_date', '')

                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    bb_bankroll = st.number_input("Bankroll ($)", value=1000, min_value=100, step=100, key="dash_bb_bankroll")
                with bc2:
                    bb_risk = st.selectbox("Risk Profile", ["conservative", "standard", "aggressive"],
                                           index=1, key="dash_bb_risk")
                with bc3:
                    bb_paper = st.checkbox("Paper mode", value=True, key="dash_bb_paper")

                if st.button("Generate Plan", type="primary", key="dash_bb_generate"):
                    payload = {
                        "session_id": bb_sid,
                        "track": bb_track,
                        "race_date": bb_date,
                        "bankroll": bb_bankroll,
                        "risk_profile": bb_risk,
                        "paper_mode": bb_paper,
                        "save": True,
                    }
                    try:
                        resp = api_post(f"/bets/build", json=payload, timeout=30)
                        if resp.ok:
                            st.session_state["dash_bb_plan"] = resp.json()
                        else:
                            st.error(f"Error: {resp.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")

                plan_data = st.session_state.get("dash_bb_plan")
                if plan_data:
                    plan = plan_data.get("plan", {})
                    race_plans = plan.get("race_plans", [])
                    diag = plan.get("diagnostics", {})

                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric("Total Risk", f"${plan.get('total_risk', 0):.0f}")
                    pm2.metric("Tickets", diag.get("total_tickets", 0))
                    pm3.metric("Races", len(race_plans))
                    pm4.metric("Passed", diag.get("total_passed", 0))

                    if plan_data.get("plan_id"):
                        st.caption(f"Plan saved (ID {plan_data['plan_id']})")

                    for rp in race_plans:
                        if rp.get("passed"):
                            continue
                        tickets = rp.get("tickets", [])
                        ticket_lines = []
                        for t in tickets:
                            sels = " / ".join(t.get("selections", []))
                            ticket_lines.append(f"{t['bet_type']} {sels} — ${t['cost']:.0f}")
                        with st.expander(
                            f"R{rp['race_number']} Grade {rp['grade']} (${rp['total_cost']:.0f})",
                            expanded=True,
                        ):
                            for tl in ticket_lines:
                                st.markdown(f"- {tl}")

    # ===== Section D2: Bet Commander =====
    with st.expander("Bet Commander", expanded=False):
        st.markdown("Full-card betting workflow with recommender, per-leg overrides, and export.")
        if st.button("Open Bet Commander", key="dash_open_commander"):
            st.session_state['_nav_target'] = "Bet Commander"
            st.rerun()

    # ===== Section E: Results & ROI =====
    with st.expander("Results & ROI", expanded=False):
        res_file = st.file_uploader("Upload results PDF", type=["pdf"], key="dash_results_upload")
        if res_file and st.button("Upload Results", key="dash_btn_results"):
            with st.spinner("Uploading results..."):
                try:
                    files = {"file": (res_file.name, res_file.getvalue(), "application/pdf")}
                    resp = api_post(f"/results/upload", files=files, timeout=60)
                    if resp.ok:
                        st.success("Results uploaded.")
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

        # ROI summary
        try:
            roi_resp = api_get(f"/predictions/roi-detailed",
                                     params={"min_n": 3}, timeout=15)
            if roi_resp.ok:
                roi = roi_resp.json()
                summary = roi.get("summary", {})
                if summary.get("total_predictions", 0) > 0:
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Total Predictions", summary.get("total_predictions", 0))
                    r2.metric("With Results", summary.get("with_results", 0))
                    r3.metric("Correct Wins", summary.get("correct_wins", 0))
                    flat_roi = summary.get("flat_roi_pct")
                    r4.metric("Flat $2 ROI", f"{flat_roi:.1f}%" if flat_roi is not None else "N/A")
                else:
                    st.info("No prediction data with results yet.")
            else:
                st.info("No ROI data available.")
        except Exception:
            st.info("Could not fetch ROI data.")

    # ===== Section F: Database =====
    with st.expander("Database", expanded=False):
        try:
            db_resp = api_get(f"/db/stats", timeout=10)
            if db_resp.ok:
                stats = db_resp.json()
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Sheets Horses", stats.get("sheets_horses", 0))
                d2.metric("BRISNET Horses", stats.get("brisnet_horses", 0))
                d3.metric("Reconciled Pairs", stats.get("reconciled_pairs", 0))
                d4.metric("Coverage", f"{stats.get('coverage_pct', 0):.1f}%")
            else:
                st.info("Could not fetch DB stats.")
        except Exception:
            st.info("Could not connect to API.")


def engine_page():
    st.header("Handicapping Engine")
    st.info("Picks are determined by Ragozin cycle patterns first. Use Secondary only to break ties (bias/pace/program #).")

    engine = HandicappingEngine()

    # --- Load last parsed session button ---
    if st.button("Load last parsed session"):
        try:
            resp = api_get(f"/races", timeout=15)
            if resp.status_code == 200:
                races = resp.json().get("races", [])
                if races:
                    latest = races[-1]
                    rid = latest.get('session_id') or latest.get('id')
                    st.session_state['active_session_id'] = rid
                else:
                    st.warning("No parsed sessions found. Upload a PDF first.")
            else:
                st.error(f"API error: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Is the backend running on http://localhost:8000?")
        except Exception as e:
            st.error(f"Error: {e}")

    # --- Session selector ---
    try:
        resp = api_get(f"/races", timeout=15)
        if resp.status_code != 200:
            st.error("Could not fetch sessions from API.")
            return
        races = resp.json().get("races", [])
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is the backend running on http://localhost:8000?")
        return
    except Exception as e:
        st.error(f"Error: {e}")
        return

    if not races:
        st.info("No parsed sessions yet. Upload a PDF to get started.")
        return

    default_idx = 0
    if 'active_session_id' in st.session_state:
        target_id = st.session_state['active_session_id']
        for i, r in enumerate(races):
            rid = r.get('session_id') or r.get('id')
            if rid == target_id:
                default_idx = i
                break

    def _session_label(x):
        name = x.get('primary_pdf_filename') or x.get('original_filename', 'unknown')
        track = x.get('track') or x.get('track_name', '')
        date = x.get('date') or x.get('analysis_date', x.get('race_date', ''))
        tag = "Primary+Secondary" if x.get('has_secondary') else "Primary Only"
        return f"{name} | {track} | {date} | {tag}"

    selected = st.selectbox(
        "Session:",
        options=races,
        format_func=_session_label,
        index=default_idx,
    )

    if not selected:
        return

    # Determine the ID and whether to use session endpoint
    sel_id = selected.get('session_id') or selected.get('id')
    is_session = 'session_id' in selected

    # Fetch horse data — use merged endpoint for sessions with secondary
    try:
        if is_session and selected.get('has_secondary'):
            h_resp = requests.get(
                f"{API_BASE_URL}/sessions/{sel_id}/races",
                params={"source": "merged"}, timeout=30
            )
            if h_resp.status_code == 200:
                all_horses = h_resp.json().get("race_data", {}).get("horses", [])
            else:
                all_horses = []
        else:
            h_resp = api_get(f"/races/{sel_id}/horses", timeout=30)
            if h_resp.status_code != 200:
                st.error("Could not load horse data for this session.")
                return
            all_horses = h_resp.json().get("horses", [])
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")
        return
    except Exception as e:
        st.error(f"Error loading horses: {e}")
        return

    if not all_horses:
        st.warning("Session contains no horse data.")
        return

    # --- Fetch ML odds for this session (cached in session_state) ---
    _odds_cache_key = f"_odds_{sel_id}"
    if _odds_cache_key not in st.session_state:
        _odds_data = {}
        _odds_by_post = {}
        try:
            _o_resp = api_get(f"/odds/snapshots/{sel_id}", timeout=10)
            if _o_resp.ok:
                for snap in _o_resp.json().get("snapshots", []):
                    entry = {"odds_raw": snap.get("odds_raw", ""), "odds_decimal": snap.get("odds_decimal")}
                    key_name = (snap["race_number"], snap["normalized_name"])
                    _odds_data[key_name] = entry
                    if snap.get("post") is not None:
                        key_post = (snap["race_number"], snap["post"])
                        _odds_by_post[key_post] = entry
        except Exception:
            pass
        st.session_state[_odds_cache_key] = (_odds_data, _odds_by_post)
    _odds_data, _odds_by_post = st.session_state[_odds_cache_key]

    # --- Status panel ---
    source = selected.get('parser_used', 'unknown')
    total_lines = sum(len(h.get('lines', [])) for h in all_horses)
    st.caption(
        f"Loaded lines: {total_lines}, horses: {len(all_horses)} (source: {source})"
    )

    # --- Session DB Stats ---
    try:
        ss_resp = api_get(f"/sessions/{sel_id}/stats", timeout=10)
        if ss_resp.status_code == 200:
            ss = ss_resp.json()
            db_resp = api_get(f"/db/stats", timeout=10)
            gs = db_resp.json() if db_resp.status_code == 200 else {}
            with st.expander("DB Coverage (this session / global)", expanded=False):
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric("Session Matched", ss.get("reconciled_pairs", 0))
                with s2:
                    sess_conf = ss.get("confidence_breakdown", {})
                    st.metric("H / M / L",
                              f"{sess_conf.get('high', 0)}/{sess_conf.get('medium', 0)}/{sess_conf.get('low', 0)}")
                with s3:
                    st.metric("Global Pairs", gs.get("reconciled_pairs", 0))
                with s4:
                    st.metric("Global Coverage", f"{gs.get('coverage_pct', 0):.1f}%")
    except Exception:
        pass

    # --- Figure quality sanity check ---
    figure_warnings: dict = {}  # race_number -> pct_missing
    _race_quality: dict = {}    # race_number -> pct_missing (all races)
    _race_detail: dict = {}     # race_number -> list of per-horse detail dicts
    _temp_groups: dict = {}
    for h in all_horses:
        rn = h.get('race_number', 0) or 0
        _temp_groups.setdefault(rn, []).append(h)
    for rn, horses_in_race in _temp_groups.items():
        missing = 0
        first_timers = 0
        details = []
        for h in horses_in_race:
            name = h.get('horse_name', '?')
            lines = h.get('lines', [])
            figs = [_extract_figure(ln) for ln in lines]
            max_fig = max(figs) if figs else 0
            is_stub = (len(lines) == 1 and not lines[0].get('raw_text', ''))
            if is_stub:
                first_timers += 1
                details.append({
                    'name': name, 'lines': len(lines), 'max_fig': 0,
                    'status': 'First-timer', 'raw': []})
            elif not figs or max_fig == 0:
                missing += 1
                raw_texts = [ln.get('raw_text', '') for ln in lines]
                details.append({
                    'name': name, 'lines': len(lines), 'max_fig': 0,
                    'status': 'Missing', 'raw': raw_texts})
            else:
                details.append({
                    'name': name, 'lines': len(lines), 'max_fig': max_fig,
                    'status': 'OK', 'raw': []})
        scoreable = len(horses_in_race) - first_timers
        pct = missing / scoreable if scoreable > 0 else 0
        _race_quality[rn] = pct
        if pct > 0.3:
            figure_warnings[rn] = pct
        _race_detail[rn] = details

    figures_ok = len(figure_warnings) == 0
    if not figures_ok:
        warn_parts = [f"R{rn}: {pct:.0%} missing" for rn, pct in sorted(figure_warnings.items())]
        st.warning(f"Figure quality issue — {', '.join(warn_parts)}. Projections may be unreliable.")

        # Race completeness details expander
        for rn in sorted(figure_warnings.keys()):
            with st.expander(f"Race {rn} completeness details"):
                details = _race_detail.get(rn, [])
                rows = []
                for d in details:
                    rows.append({
                        'Horse': d['name'],
                        'Lines': d['lines'],
                        'Max Figure': d['max_fig'] if d['max_fig'] else '-',
                        'Status': d['status'],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Show raw text for missing horses
                miss_list = [d for d in details if d['status'] == 'Missing']
                if miss_list:
                    st.caption("Missing horses — raw line text:")
                    for d in miss_list:
                        st.code('\n'.join(d['raw']) if d['raw'] else '(empty)', language=None)

                # Parser diagnostic dump
                try:
                    diag_resp = requests.get(
                        f"{API_BASE_URL}/parser/diagnose/{sel_id}/{rn}", timeout=15)
                    if diag_resp.ok:
                        diag = diag_resp.json()
                        for pg in diag.get('pages', []):
                            st.markdown(f"**{pg['horse_name']}** (page {pg['page_num']})")
                            for cl in pg.get('classified_lines', []):
                                kind = cl['kind']
                                marker = {'figure': 'FIG', 'data': 'DAT',
                                          'concat_figure_data': 'FIG+DAT',
                                          'horse_name': 'HDR', 'race_header': 'HDR',
                                          'conditions': 'HDR', 'year_summary': 'YR',
                                          'noise': '...'}.get(kind, kind)
                                st.text(f"  [{marker:7s}] {cl['text']}")
                except Exception:
                    pass

    # Group horses by race_number
    race_groups: dict = {}
    for h in all_horses:
        rn = h.get('race_number', 0) or 0
        race_groups.setdefault(rn, []).append(h)

    race_summary_parts = []
    for rn in sorted(race_groups.keys()):
        count = len(race_groups[rn])
        label = f"R{rn}" if rn else "Ungrouped"
        q_pct = _race_quality.get(rn)
        badge = f" {_quality_badge(q_pct)}" if q_pct is not None else ""
        race_summary_parts.append(f"{label}: {count}{badge}")
    st.markdown(" | ".join(race_summary_parts))

    # =====================================================================
    # (A) Bias Settings
    # =====================================================================
    st.subheader("Bias Settings")
    bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
    with bcol1:
        e_pct = st.slider("E %", 0, 100, 25)
    with bcol2:
        ep_pct = st.slider("EP %", 0, 100, 25)
    with bcol3:
        p_pct = st.slider("P %", 0, 100, 25)
    with bcol4:
        s_pct = st.slider("S %", 0, 100, 25)
    with bcol5:
        speed_fav = st.checkbox("Speed favoring")

    bias = BiasInput(
        e_pct=float(e_pct), ep_pct=float(ep_pct),
        p_pct=float(p_pct), s_pct=float(s_pct),
        speed_favoring=speed_fav,
    )

    # =====================================================================
    # (B) Race Analysis
    # =====================================================================
    st.divider()
    st.subheader("Race Analysis")

    race_options = sorted(race_groups.keys())
    race_labels = {
        rn: f"Race {rn} ({len(race_groups[rn])} horses)" if rn
        else f"All ({len(race_groups[rn])} horses)"
        for rn in race_options
    }
    selected_race = st.selectbox(
        "Race:",
        options=race_options,
        format_func=lambda rn: race_labels[rn],
    )

    race_all_horses = race_groups[selected_race]
    race_names = [h.get('horse_name', 'Unknown') for h in race_all_horses]
    scratch_key = f"scratch_{sel_id}_{selected_race}"
    scratches = st.multiselect("Scratches:", options=race_names, key=scratch_key)

    race_horses = [h for h in race_all_horses if h.get('horse_name', '') not in scratches]
    if not race_horses:
        st.warning("All horses scratched.")
        return

    if st.button("Run Projections", type="primary"):
        projections, audit = _run_race_projections(engine, race_horses, bias, scratches, selected_race)
        if not projections:
            st.toast("No projections produced. All horses may lack usable figures.")
        else:
            st.session_state['last_projections'] = projections
            st.session_state['last_proj_race'] = selected_race
            st.session_state['last_audit'] = audit
            s_track = selected.get('track') or selected.get('track_name', '')
            s_date = selected.get('date') or selected.get('race_date', '')
            _persist_predictions(sel_id, s_track, s_date, selected_race, projections)
            _mark_engine_run(sel_id, selected_race, projections=projections, audit=audit)

    # =====================================================================
    # (C) Outputs
    # =====================================================================
    projections = st.session_state.get('last_projections')
    proj_race = st.session_state.get('last_proj_race')
    audit = st.session_state.get('last_audit')
    if projections and not hasattr(projections[0], 'tossed'):
        st.session_state.pop('last_projections', None)
        st.session_state.pop('last_proj_race', None)
        st.session_state.pop('last_audit', None)
        projections = None
        audit = None

    if not projections or proj_race != selected_race:
        st.info("Click **Run Projections** to analyze this race.")
    else:
        sorted_by_bias = sorted(projections, key=lambda p: p.bias_score, reverse=True)

        # Filter: New Top Setups only (Ragozin-only)
        is_ragozin_primary = not any(
            h.get('figure_source') == 'brisnet' for h in race_horses
        )
        show_only_new_top = False
        if is_ragozin_primary:
            show_only_new_top = st.checkbox(
                "Show only \u2b50 New Top Setups",
                key=f"filter_newtop_{sel_id}_{selected_race}",
            )

        display_projections = sorted_by_bias
        if show_only_new_top:
            display_projections = [p for p in sorted_by_bias if p.new_top_setup]
            if not display_projections:
                st.info("No New Top Setup horses found in this race.")

        # --- Top 3 Picks ---
        _render_top_picks(display_projections, selected_race, key_prefix=f"eng_{sel_id}")

        # --- Full Ranked Table ---
        show_odds = st.checkbox("Show Odds", value=True, key=f"show_odds_{sel_id}_{selected_race}")
        _render_ranked_table(display_projections, race_horses, show_odds,
                             _odds_data, _odds_by_post, selected_race,
                             key_prefix=f"eng_{sel_id}")

        # --- Analysis expanders ---
        _render_analysis_expanders(sorted_by_bias, audit, race_horses,
                                   key_prefix=f"eng_{sel_id}")

    # =====================================================================
    # (B) Best Bets - across all races
    # =====================================================================
    st.divider()
    st.subheader("Best Bets (All Races)")

    best_bets_blocked = not figures_ok
    if best_bets_blocked:
        override = st.checkbox("Override figure quality warning and allow Best Bets")
        if override:
            best_bets_blocked = False

    if best_bets_blocked:
        st.info("Best Bets disabled due to figure quality issues. Check the override box above to proceed anyway.")
    elif st.button("Generate Best Bets"):
        all_bets = []
        progress = st.progress(0)
        race_keys = sorted(race_groups.keys())

        for idx, rn in enumerate(race_keys):
            # Apply per-race scratches from session state
            rn_scratch_key = f"scratch_{sel_id}_{rn}"
            rn_scratches = st.session_state.get(rn_scratch_key, [])
            rh = [h for h in race_groups[rn] if h.get('horse_name', '') not in rn_scratches]
            if not rh:
                progress.progress((idx + 1) / len(race_keys))
                continue
            projs, _audit = _run_race_projections(engine, rh, bias, rn_scratches, rn)
            for p in projs:
                all_bets.append((rn, p))
            # Persist predictions per race
            s_track = selected.get('track') or selected.get('track_name', '')
            s_date = selected.get('date') or selected.get('race_date', '')
            _persist_predictions(sel_id, s_track, s_date, rn, projs)
            _mark_engine_run(sel_id, rn, projections=projs, audit=_audit, bias=bias)
            progress.progress((idx + 1) / len(race_keys))

        progress.empty()

        if not all_bets:
            st.warning("No projections across any race.")
        else:
            # Sort by composite best-bet score
            all_bets.sort(key=lambda x: _best_bet_score(x[1]), reverse=True)
            st.session_state['best_bets'] = all_bets

    # Display best bets if available
    best_bets = st.session_state.get('best_bets')
    if best_bets and not hasattr(best_bets[0][1], 'new_top_setup'):
        st.session_state.pop('best_bets', None)
        best_bets = None
    if best_bets:
        _render_best_bets_table(best_bets, _odds_data, _odds_by_post,
                                key_prefix=f"eng_bb_{sel_id}")

    # =====================================================================
    # (C) Bet Builder — inline on Engine page
    # =====================================================================
    st.divider()
    st.subheader("Bet Builder")
    st.caption("Odds source: BRISNET morning line")

    if not best_bets:
        st.info("Generate Best Bets above first to use the Bet Builder.")
    else:
        s_track = selected.get('track') or selected.get('track_name', '')
        s_date = selected.get('date') or selected.get('race_date', '')

        with st.expander("Bet Settings", expanded=False):
            bb_c1, bb_c2, bb_c3 = st.columns(3)
            with bb_c1:
                bb_bankroll = st.number_input(
                    "Bankroll ($)", min_value=100, max_value=100000,
                    value=1000, step=100, key="bb_bankroll")
                bb_risk_profile = st.selectbox(
                    "Risk Profile", ["conservative", "standard", "aggressive"],
                    index=1, key="bb_risk_profile")
            with bb_c2:
                bb_max_race = st.number_input(
                    "Max risk/race (%)", min_value=0.5, max_value=10.0,
                    value=1.5, step=0.5, key="bb_max_race")
                bb_max_day = st.number_input(
                    "Max risk/day (%)", min_value=1.0, max_value=20.0,
                    value=6.0, step=1.0, key="bb_max_day")
            with bb_c3:
                bb_min_odds_a = st.number_input(
                    "Min odds A-grade", min_value=0.5, max_value=20.0,
                    value=2.0, step=0.5, key="bb_min_odds_a")
                bb_min_odds_b = st.number_input(
                    "Min odds B-grade", min_value=0.5, max_value=20.0,
                    value=4.0, step=0.5, key="bb_min_odds_b")
            bb_c4, bb_c5 = st.columns(2)
            with bb_c4:
                bb_min_conf = st.number_input(
                    "Min confidence (0 = defaults per grade)", min_value=0.0,
                    max_value=1.0, value=0.75, step=0.05, key="bb_min_conf",
                    help="Default 75%. Set to 0 to use built-in A/B thresholds.")
            with bb_c5:
                bb_allow_missing = st.checkbox(
                    "Allow missing odds (flat-stake WIN)", value=False,
                    key="bb_allow_missing",
                    help="When ON, places a flat-stake WIN bet even if track odds are unavailable.")

        if st.button("Generate Bet Plan (Paper Mode)", key="bb_generate"):
            payload = {
                "session_id": sel_id,
                "track": s_track,
                "race_date": s_date,
                "bankroll": bb_bankroll,
                "risk_profile": bb_risk_profile,
                "max_risk_per_race_pct": bb_max_race,
                "max_risk_per_day_pct": bb_max_day,
                "min_confidence": bb_min_conf,
                "min_odds_a": bb_min_odds_a,
                "min_odds_b": bb_min_odds_b,
                "paper_mode": True,
                "allow_missing_odds": bb_allow_missing,
                "save": True,
            }
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/bets/build", json=payload, timeout=30)
                if resp.ok:
                    st.session_state['bb_engine_plan'] = resp.json()
                else:
                    st.error(f"Bet Builder error: {resp.text}")
            except Exception as e:
                st.error(f"API connection error: {e}")

        # ----- Display plan / diagnostics -----
        plan_data = st.session_state.get('bb_engine_plan')
        if plan_data:
            plan = plan_data.get("plan", {})
            diag = plan.get("diagnostics", {})
            race_plans = plan.get("race_plans", [])
            total_risk = plan.get("total_risk", 0)
            total_tickets = diag.get("total_tickets", 0)
            grade_counts = diag.get("grade_counts", {})
            blocker_list = diag.get("blockers", [])

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Risk", f"${total_risk:.0f}")
            m2.metric("Tickets", total_tickets)
            m3.metric("Races", len(race_plans))
            m4.metric("Passed", diag.get("total_passed", 0))

            if plan_data.get("plan_id"):
                st.caption(f"Plan saved (ID {plan_data['plan_id']}) — PAPER MODE")

            # Grade breakdown
            grades_str = "  |  ".join(
                f"**{g}**: {grade_counts.get(g, 0)}" for g in ["A", "B", "C"])
            st.markdown(f"Grade breakdown: {grades_str}")

            # Odds toggle + coverage for tickets
            bb_ticket_show_odds = st.checkbox(
                "Show Odds", value=True, key=f"bb_ticket_show_odds_{sel_id}")
            if bb_ticket_show_odds:
                _ticket_horses_total = 0
                _ticket_horses_with_odds = 0
                for _rp in race_plans:
                    for _t in _rp.get("tickets", []):
                        _d = _t.get("details", {})
                        if _t["bet_type"] == "WIN":
                            _ticket_horses_total += 1
                            if _d.get("odds_raw") or _d.get("odds"):
                                _ticket_horses_with_odds += 1
                        elif _t["bet_type"] == "EXACTA":
                            for _hdata in _d.get("horses", {}).values():
                                _ticket_horses_total += 1
                                if _hdata.get("odds_raw") or _hdata.get("odds_decimal"):
                                    _ticket_horses_with_odds += 1
                st.caption(
                    f"Odds coverage: {_ticket_horses_with_odds}/{_ticket_horses_total} horses in tickets")

            if total_tickets > 0:
                # Show tickets
                for rp in race_plans:
                    if rp.get("passed"):
                        continue
                    with st.expander(
                        f"R{rp['race_number']} — Grade {rp['grade']} "
                        f"(${rp['total_cost']:.0f})", expanded=True
                    ):
                        for t in rp.get("tickets", []):
                            details = t.get("details", {})
                            bet_type = t["bet_type"]

                            if bet_type == "WIN":
                                name = t["selections"][0] if t["selections"] else ""
                                if bb_ticket_show_odds:
                                    raw = details.get("odds_raw") or ""
                                    dec = details.get("odds")
                                    if raw:
                                        odds_tag = f" @ {raw}"
                                    elif dec:
                                        odds_tag = f" @ {dec:.1f}-1"
                                    else:
                                        odds_tag = ""
                                else:
                                    odds_tag = ""
                                header = f"**WIN** {name}{odds_tag}"

                            elif bet_type == "EXACTA":
                                horses = details.get("horses", {})
                                structure = details.get("structure", "")

                                def _horse_odds_label(hname):
                                    if not bb_ticket_show_odds:
                                        return hname
                                    h = horses.get(hname, {})
                                    raw = h.get("odds_raw") or ""
                                    dec = h.get("odds_decimal")
                                    if raw:
                                        return f"{hname} @ {raw}"
                                    if dec is not None:
                                        return f"{hname} @ {dec:.1f}-1"
                                    return hname

                                if structure == "key":
                                    top = details.get("top", "")
                                    unders = details.get("unders", [])
                                    top_lbl = _horse_odds_label(top)
                                    under_lbls = [_horse_odds_label(u) for u in unders]
                                    header = f"**EXACTA** KEY {top_lbl} over {', '.join(under_lbls)}"
                                elif structure == "saver":
                                    overs = details.get("overs", [])
                                    under = details.get("under", "")
                                    over_lbls = [_horse_odds_label(o) for o in overs]
                                    header = f"**EXACTA** SAVER {', '.join(over_lbls)} / {under}"
                                else:
                                    sels = " / ".join(t["selections"])
                                    header = f"**EXACTA** {sels}"

                            else:
                                sels = " / ".join(t["selections"])
                                header = f"**{bet_type}** {sels}"

                            st.markdown(
                                f"{header} — ${t['cost']:.0f}  \n"
                                f"{t.get('rationale', '')}")
            else:
                # No bets — show diagnostics
                st.warning("No bets generated. See blockers below.")

            # Blockers section (always show if any)
            if blocker_list:
                with st.expander(
                    f"Blockers ({len(blocker_list)})", expanded=total_tickets == 0
                ):
                    for b in blocker_list:
                        st.markdown(
                            f"- **R{b['race']}** (Grade {b['grade']}): "
                            f"{b['reason']}")

                # "Relax rules" quick-buttons
                if total_tickets == 0:
                    st.markdown("**Relax rules:**")
                    rx1, rx2, rx3 = st.columns(3)
                    with rx1:
                        if st.button("Lower min confidence by 5%",
                                     key="bb_relax_conf"):
                            new_val = max(
                                0, st.session_state.get("bb_min_conf", 0.75) - 0.05)
                            st.session_state["bb_min_conf"] = round(new_val, 2)
                            st.rerun()
                    with rx2:
                        if st.button("Lower min odds by 0.5",
                                     key="bb_relax_odds"):
                            new_a = max(
                                0.5,
                                st.session_state.get("bb_min_odds_a", 2.0) - 0.5)
                            new_b = max(
                                0.5,
                                st.session_state.get("bb_min_odds_b", 4.0) - 0.5)
                            st.session_state["bb_min_odds_a"] = new_a
                            st.session_state["bb_min_odds_b"] = new_b
                            st.rerun()
                    with rx3:
                        if st.button("Allow missing odds",
                                     key="bb_relax_missing"):
                            st.session_state["bb_allow_missing"] = True
                            st.rerun()

    # =====================================================================
    # (D) Bet Commander link
    # =====================================================================
    st.divider()
    with st.expander("Bet Commander", expanded=False):
        st.markdown("Full-card betting: recommender, per-leg overrides, export.")
        if st.button("Open Bet Commander", key="eng_open_commander"):
            st.session_state['_nav_target'] = "Bet Commander"
            st.rerun()


def upload_page():
    st.header("Upload Racing Sheet")

    tab_new, tab_secondary = st.tabs(["New Session", "Add Secondary to Existing"])

    # ===== Tab 1: New Session (Primary + optional Secondary) =====
    with tab_new:
        if 'use_gpt_parser' not in st.session_state:
            st.session_state['use_gpt_parser'] = False
        st.checkbox(
            "Use GPT parser",
            key='use_gpt_parser',
            help="When enabled, uses GPT-4 Vision for parsing. Falls back to traditional parser on failure."
        )

        primary_files = st.file_uploader(
            "Step 1: Upload PRIMARY — Ragozin Sheets (required for cycle-based picks)",
            type=['pdf'],
            key='primary_uploader',
            accept_multiple_files=True,
            help="Primary drives picks using Sheets cycles. Upload one or more PDFs."
        )

        if primary_files:
            st.info(f"{len(primary_files)} file(s) selected")

            if st.button("Parse Primary", type="primary", key="btn_parse_primary"):
                use_gpt = st.session_state['use_gpt_parser']
                all_results = []
                progress = st.progress(0)
                for idx, pf in enumerate(primary_files):
                    with st.spinner(f"Parsing {pf.name} ({idx+1}/{len(primary_files)})..."):
                        try:
                            files = {"file": (pf.name, pf.getvalue(), "application/pdf")}
                            params = {"use_gpt": use_gpt}
                            response = requests.post(
                                f"{API_BASE_URL}/upload_primary",
                                files=files, params=params, timeout=300
                            )
                            if response.status_code == 200:
                                result = response.json()
                                all_results.append(result)
                                st.success(
                                    f"{pf.name}: {result['horses_count']} horses, "
                                    f"{result['races_count']} races "
                                    f"({result.get('track', '?')} — {result.get('parser_used', '?')})"
                                )
                            else:
                                st.error(f"{pf.name}: {response.text}")
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot connect to API. Is the backend running on http://localhost:8000?")
                        except Exception as e:
                            st.error(f"{pf.name}: {str(e)}")
                    progress.progress((idx + 1) / len(primary_files))

                if all_results:
                    last = all_results[-1]
                    st.session_state['new_session_id'] = last['session_id']
                    st.session_state['last_race_id'] = last['session_id']
                    total_horses = sum(r['horses_count'] for r in all_results)
                    st.info(f"Parsed {len(all_results)} file(s): {total_horses} total horses across {len(all_results)} session(s)")
                    # Show last session details
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Session ID", last['session_id'][:8] + "...")
                    with col2:
                        st.metric("Track", last.get('track', 'N/A'))
                    with col3:
                        st.metric("Parser", last.get('parser_used', 'unknown'))
                    if last.get('reconciliation'):
                        recon = last['reconciliation']
                        rc1, rc2 = st.columns(2)
                        with rc1:
                            st.metric("New Matches", recon.get('new_matches', 0))
                        with rc2:
                            st.metric("Global Coverage", f"{last.get('db_coverage_pct', 0):.1f}%")

        # Step 2: secondary upload (conditional on step 1)
        new_sid = st.session_state.get('new_session_id')
        if new_sid:
            st.divider()
            secondary_file = st.file_uploader(
                "Step 2: Add SECONDARY — BRISNET/DRF Racing Form (optional enrichment)",
                type=['pdf'],
                key='secondary_uploader_new',
                help="Secondary adds program #, runstyle, workouts, trainer/jockey stats, and track-bias tables. It must not override Sheets cycles."
            )
            if secondary_file is not None and st.button("Attach Secondary", key="btn_attach_secondary_new"):
                with st.spinner("Parsing secondary PDF..."):
                    try:
                        files = {"file": (secondary_file.name, secondary_file.getvalue(), "application/pdf")}
                        response = requests.post(
                            f"{API_BASE_URL}/upload_secondary",
                            files=files, params={"session_id": new_sid}, timeout=300
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Secondary attached: {result['horses_count']} horses")
                            if result.get('merge_coverage'):
                                st.info(f"Merge coverage: {result['merge_coverage']}")
                            if result.get('reconciliation'):
                                recon = result['reconciliation']
                                rc1, rc2 = st.columns(2)
                                with rc1:
                                    st.metric("New Matches", recon.get('new_matches', 0))
                                with rc2:
                                    st.metric("Global Coverage", f"{result.get('db_coverage_pct', 0):.1f}%")
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            st.divider()
            col_nav1, col_nav2, _ = st.columns([1, 1, 2])
            with col_nav1:
                if st.button("Open in Engine", type="primary", key="btn_engine_new"):
                    st.session_state['active_session_id'] = new_sid
                    st.session_state['_nav_target'] = "Engine"
                    st.rerun()
            with col_nav2:
                if st.button("Upload another", key="btn_another_new"):
                    st.session_state.pop('new_session_id', None)
                    st.session_state['_nav_target'] = "Upload PDF"
                    st.rerun()

    # ===== Tab 2: Add Secondary to Existing Session =====
    with tab_secondary:
        try:
            resp = api_get(f"/sessions", timeout=15)
            if resp.status_code != 200:
                st.error("Could not fetch sessions.")
                return
            sess_list = resp.json().get("sessions", [])
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API.")
            return
        except Exception as e:
            st.error(f"Error: {e}")
            return

        eligible = [s for s in sess_list if s.get('has_primary') and not s.get('has_secondary')]
        if not eligible:
            st.info("No sessions without a secondary upload. Create a new session first.")
        else:
            selected_sess = st.selectbox(
                "Select session:",
                options=eligible,
                format_func=lambda s: (
                    f"{s.get('primary_pdf_filename', '?')} | "
                    f"{s.get('track', 'N/A')} | {s.get('date', 'N/A')} | "
                    f"{s.get('primary_horses_count', 0)} horses"
                ),
                key="secondary_session_selector",
            )

            if selected_sess:
                sec_file = st.file_uploader(
                    "Upload SECONDARY — BRISNET/DRF Racing Form (enrichment only)",
                    type=['pdf'],
                    key='secondary_uploader_existing',
                    help="Adds program #, runstyle, workouts, trainer/jockey stats. Does not override Sheets cycles."
                )
                if sec_file is not None and st.button("Attach Secondary", key="btn_attach_secondary_exist"):
                    sid = selected_sess['session_id']
                    with st.spinner("Parsing secondary PDF..."):
                        try:
                            files = {"file": (sec_file.name, sec_file.getvalue(), "application/pdf")}
                            response = requests.post(
                                f"{API_BASE_URL}/upload_secondary",
                                files=files, params={"session_id": sid}, timeout=300
                            )
                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"Secondary attached: {result['horses_count']} horses")
                                if result.get('merge_coverage'):
                                    st.info(f"Merge coverage: {result['merge_coverage']}")
                            else:
                                st.error(f"Error: {response.text}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

def horses_overview_page():
    st.header("🐎 Horses Overview with AI Analysis")

    try:
        response = api_get(f"/races", timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            races = data["races"]
            
            if not races:
                st.info("No races have been parsed yet. Upload a PDF to get started!")
                return
            
            st.metric("Total Parsed Sessions", len(races))
            
            # Create a DataFrame for display
            df = pd.DataFrame(races)
            df['parsed_at'] = pd.to_datetime(df['parsed_at'])
            
            # Display sessions in a table
            st.subheader("📋 All Parsing Sessions")
            display_columns = ['track_name', 'race_date', 'horses_count', 'analysis_date', 'analysis_time']
            if 'parser_used' in df.columns:
                display_columns.append('parser_used')
            if 'processing_duration' in df.columns:
                display_columns.append('processing_duration')
            st.dataframe(
                df[display_columns],
                use_container_width=True
            )
            
            # Session selection
            st.subheader("🔍 Select Session for Detailed Analysis")
            
            if 'last_race_id' in st.session_state:
                target = st.session_state.last_race_id
                default_index = next(
                    (i for i, race in enumerate(races)
                     if (race.get('session_id') or race.get('id')) == target),
                    0
                )
            else:
                default_index = 0

            selected_race = st.selectbox(
                "Choose a session:",
                options=races,
                format_func=lambda x: (
                    f"{x.get('primary_pdf_filename') or x.get('original_filename', 'unknown')} | "
                    f"{x.get('track') or x.get('track_name', '')} | "
                    f"{x.get('date') or x.get('analysis_date', x.get('race_date', ''))} | "
                    f"{x.get('parser_used', 'unknown')}"
                ),
                index=default_index
            )
            
            if selected_race:
                display_enhanced_horses_details(selected_race)
                
        else:
            st.error(f"Error fetching races: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is the backend running on http://localhost:8000?")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def display_enhanced_horses_details(race):
    source = race.get('parser_used', 'unknown')
    st.caption(
        f"Loaded races: {race.get('total_races', 'N/A')}, "
        f"horses: {race['horses_count']} "
        f"(source: {source})"
    )
    st.subheader(f"🏆 {race['track_name']} - {race['horses_count']} Horses with AI Analysis")
    
    # Session info cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Horses", race['horses_count'])
    with col2:
        st.metric("Analysis Date", race.get('analysis_date', race['race_date']))
    with col3:
        st.metric("Analysis Time", race.get('analysis_time', 'N/A'))
    with col4:
        st.metric("Processing Time", race.get('processing_duration', 'N/A'))
    
    # Fetch detailed horse data
    try:
        response = api_get(f"/races/{race['id']}/horses")
        
        if response.status_code == 200:
            horse_data = response.json()
            horses = horse_data['horses']
            
            if horses:
                # Process horses data with enhanced analysis
                all_races = []
                horse_summary = []
                
                for horse in horses:
                    try:
                        horse_name = horse.get('horse_name', 'Unknown')
                        sex = horse.get('sex', 'Unknown')
                        age = horse.get('age', 0)
                        breeder_owner = horse.get('breeder_owner', 'Unknown')
                        foal_date = horse.get('foal_date', 'Unknown')
                        reg_code = horse.get('reg_code', 'Unknown')
                        total_races = horse.get('races', 0)
                        top_fig = horse.get('top_fig', 'Unknown')
                        horse_analysis = horse.get('horse_analysis', 'No analysis available')
                        performance_trend = horse.get('performance_trend', 'No trend analysis available')
                        
                        # Handle both enhanced and legacy formats
                        lines = horse.get('lines', [])
                        races = horse.get('races', [])  # Enhanced format
                        
                        # Use enhanced races if available, otherwise fall back to lines
                        race_entries = races if races else lines
                        
                        # Calculate statistics with robust None handling
                        try:
                            # Extract numeric figures from enhanced or legacy format
                            ragozin_figures = []
                            for race_entry in race_entries:
                                # Try enhanced format first
                                parsed_fig = race_entry.get('parsed_figure', 0.0)
                                if parsed_fig and parsed_fig > 0:
                                    ragozin_figures.append(parsed_fig)
                                else:
                                    # Fallback to legacy format
                                    fig = race_entry.get('fig', '')
                                    if fig:
                                        # Extract numeric part (remove flags like +, -, ~, etc.)
                                        import re
                                        numeric_match = re.search(r'(\d+(?:\.\d+)?)', fig)
                                        if numeric_match:
                                            ragozin_figures.append(float(numeric_match.group(1)))
                            
                            avg_ragozin = sum(ragozin_figures) / len(ragozin_figures) if ragozin_figures else 0
                            best_ragozin = min(ragozin_figures) if ragozin_figures else 0
                            
                            horse_summary.append({
                                'name': horse_name,
                                'sex': sex,
                                'age': age,
                                'breeder_owner': breeder_owner,
                                'total_races': total_races,
                                'top_fig': top_fig,
                                'avg_ragozin': avg_ragozin,
                                'best_ragozin': best_ragozin,
                                'horse_analysis': horse_analysis,
                                'performance_trend': performance_trend
                            })
                        except Exception as e:
                            st.error(f"Error calculating summary for {horse_name}: {str(e)}")
                            horse_summary.append({
                                'name': horse_name,
                                'sex': sex,
                                'age': age,
                                'breeder_owner': breeder_owner,
                                'total_races': total_races,
                                'top_fig': top_fig,
                                'avg_ragozin': 0,
                                'best_ragozin': 0,
                                'horse_analysis': horse_analysis,
                                'performance_trend': performance_trend
                            })
                        
                        # Add individual race lines with enhanced data
                        for race_entry in race_entries:
                            try:
                                # Enhanced race data
                                race_data = {
                                    'horse_name': horse_name,
                                    'sex': sex,
                                    'age': age,
                                    'breeder_owner': breeder_owner,
                                    'total_races': total_races,
                                    'top_fig': top_fig,
                                    'horse_analysis': horse_analysis,
                                    'performance_trend': performance_trend,
                                    # Enhanced fields
                                    'race_year': race_entry.get('race_year', 0),
                                    'race_index': race_entry.get('race_index', 0),
                                    'figure_raw': race_entry.get('figure_raw', ''),
                                    'parsed_figure': race_entry.get('parsed_figure', 0.0),
                                    'pre_symbols': race_entry.get('pre_symbols', []),
                                    'post_symbols': race_entry.get('post_symbols', []),
                                    'distance_bracket': race_entry.get('distance_bracket', ''),
                                    'surface_type': race_entry.get('surface_type', ''),
                                    'track_code': race_entry.get('track_code', ''),
                                    'date_code': race_entry.get('date_code', ''),
                                    'month_label': race_entry.get('month_label', ''),
                                    'race_class_code': race_entry.get('race_class_code', ''),
                                    'trouble_indicators': race_entry.get('trouble_indicators', []),
                                    'ai_analysis': race_entry.get('ai_analysis', {}),
                                    # Legacy fields for compatibility
                                    'fig': race_entry.get('fig', ''),
                                    'flags': race_entry.get('flags', []),
                                    'track': race_entry.get('track', ''),
                                    'month': race_entry.get('month', ''),
                                    'surface': race_entry.get('surface', ''),
                                    'race_type': race_entry.get('race_type', ''),
                                    'race_date': race_entry.get('race_date', ''),
                                    'notes': race_entry.get('notes', ''),
                                    'race_analysis': race_entry.get('race_analysis', '')
                                }
                                all_races.append(race_data)
                            except Exception as e:
                                st.error(f"Error processing race entry for {horse_name}: {str(e)}")
                                continue
                                
                    except Exception as e:
                        st.error(f"Error processing horse {horse.get('name', 'Unknown')}: {str(e)}")
                        continue
                
                # Create DataFrames with error handling
                try:
                    horses_df = pd.DataFrame(horse_summary)
                    races_df = pd.DataFrame(all_races)
                    
                    # Ensure all required columns exist
                    if len(horses_df) > 0:
                        required_columns = ['name', 'sex', 'age', 'breeder_owner', 'total_races', 'top_fig', 'avg_ragozin', 'best_ragozin', 'horse_analysis', 'performance_trend']
                        for col in required_columns:
                            if col not in horses_df.columns:
                                horses_df[col] = 0 if col in ['age', 'total_races', 'avg_ragozin', 'best_ragozin'] else ''
                    
                    if len(races_df) > 0:
                        required_race_columns = [
                            'horse_name', 'sex', 'age', 'breeder_owner', 'total_races', 'top_fig', 
                            'horse_analysis', 'performance_trend', 
                            # Enhanced fields
                            'race_year', 'race_index', 'figure_raw', 'parsed_figure', 'pre_symbols', 
                            'post_symbols', 'distance_bracket', 'surface_type', 'track_code', 
                            'date_code', 'month_label', 'race_class_code', 'trouble_indicators', 'ai_analysis',
                            # Legacy fields for compatibility
                            'fig', 'flags', 'track', 'month', 'surface', 'race_type', 'race_date', 
                            'notes', 'race_analysis'
                        ]
                        for col in required_race_columns:
                            if col not in races_df.columns:
                                if col in ['age', 'total_races', 'race_year', 'race_index', 'parsed_figure']:
                                    races_df[col] = 0
                                elif col in ['pre_symbols', 'post_symbols', 'trouble_indicators', 'flags']:
                                    races_df[col] = []
                                elif col == 'ai_analysis':
                                    races_df[col] = {}
                                else:
                                    races_df[col] = ''
                        
                except Exception as e:
                    st.error(f"Error creating DataFrames: {str(e)}")
                    horses_df = pd.DataFrame(columns=['name', 'sex', 'age', 'breeder_owner', 'total_races', 'top_fig', 'avg_ragozin', 'best_ragozin', 'horse_analysis', 'performance_trend'])
                    races_df = pd.DataFrame(columns=[
                        'horse_name', 'sex', 'age', 'breeder_owner', 'total_races', 'top_fig', 
                        'horse_analysis', 'performance_trend', 
                        # Enhanced fields
                        'race_year', 'race_index', 'figure_raw', 'parsed_figure', 'pre_symbols', 
                        'post_symbols', 'distance_bracket', 'surface_type', 'track_code', 
                        'date_code', 'month_label', 'race_class_code', 'trouble_indicators', 'ai_analysis',
                        # Legacy fields for compatibility
                        'fig', 'flags', 'track', 'month', 'surface', 'race_type', 'race_date', 
                        'notes', 'race_analysis'
                    ])
                
                # Display enhanced horses summary
                st.subheader("🐎 Horses Summary with AI Analysis")
                
                # Display horses with their AI analysis
                for _, horse_row in horses_df.iterrows():
                    with st.expander(f"🏇 {horse_row['name']} - {horse_row['sex']}/{horse_row['age']} - {horse_row['breeder_owner']} ({horse_row['total_races']} races)"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Races", horse_row['total_races'])
                            st.metric("Age", horse_row['age'])
                        
                        with col2:
                            st.metric("Top Figure", horse_row['top_fig'])
                            st.metric("Avg Ragozin", f"{horse_row['avg_ragozin']:.1f}")
                        
                        with col3:
                            st.metric("Best Ragozin", f"{horse_row['best_ragozin']:.1f}")
                            st.metric("Sex", horse_row['sex'])
                        
                        with col4:
                            st.metric("Breeder/Owner", horse_row['breeder_owner'])
                            st.metric("Reg Code", horse_row.get('reg_code', 'N/A'))
                        
                        # AI Analysis
                        st.subheader("🤖 AI Analysis")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Overall Performance Analysis:**")
                            st.info(horse_row['horse_analysis'])
                        
                        with col2:
                            st.write("**Performance Trend Analysis:**")
                            st.info(horse_row['performance_trend'])
                        
                        # Race history with symbols
                        horse_races = races_df[races_df['horse_name'] == horse_row['name']]
                        if len(horse_races) > 0:
                            st.subheader("📋 Race History with Symbol Analysis")
                            
                            # Create enhanced race display
                            race_display_data = []
                            for _, race_row in horse_races.iterrows():
                                race_display_data.append({
                                    'Date': race_row['race_date'],
                                    'Track': race_row['track'],
                                    'Month': race_row['month'],
                                    'Type': race_row['race_type'],
                                    'Figure': race_row['fig'],
                                    'Flags': race_row['flags'],
                                    'Surface': race_row['surface'],
                                    'Analysis': race_row['race_analysis'][:100] + "..." if len(race_row['race_analysis']) > 100 else race_row['race_analysis']
                                })
                            
                            race_display_df = pd.DataFrame(race_display_data)
                            st.dataframe(race_display_df, use_container_width=True)
                            
                            # Show detailed race analysis
                            st.subheader("🔍 Detailed Race Analysis")
                            for _, race_row in horse_races.iterrows():
                                if race_row['race_analysis'] and race_row['race_analysis'] != '':
                                    with st.expander(f"Race: {race_row['race_date']} - {race_row['track']} - {race_row['race_type']}"):
                                        st.write(f"**Figure:** {race_row['fig']}")
                                        if race_row['flags']:
                                            st.write(f"**Flags:** {race_row['flags']}")
                                        st.write(f"**Surface:** {race_row['surface']}")
                                        st.write(f"**Month:** {race_row['month']}")
                                        if race_row['notes']:
                                            st.write(f"**Notes:** {race_row['notes']}")
                                        st.write("**AI Analysis:**")
                                        st.info(race_row['race_analysis'])
                
                # Store data in session state for other pages
                st.session_state.horses_df = horses_df
                st.session_state.races_df = races_df
                
            else:
                st.warning("No horse data found for this session.")
                
        else:
            st.error(f"Error fetching horse data: {response.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

def individual_horse_analysis_page():
    st.header("🔍 Individual Horse Analysis")
    
    # Check if we have data from the overview page
    if 'horses_df' not in st.session_state or 'races_df' not in st.session_state:
        st.info("Please go to 'Horses Overview' first to load the data.")
        return
    
    horses_df = st.session_state.horses_df
    races_df = st.session_state.races_df
    
    if len(horses_df) == 0:
        st.warning("No horse data available. Please load data from the overview page.")
        return
    
    # Horse selection
    st.subheader("🏇 Select Horse for Deep Analysis")
    selected_horse = st.selectbox(
        "Choose a horse:",
        options=horses_df['name'].tolist(),
        format_func=lambda x: f"{x} ({horses_df[horses_df['name']==x]['total_races'].iloc[0]} races)"
    )
    
    if selected_horse:
        horse_data = horses_df[horses_df['name'] == selected_horse].iloc[0]
        horse_races = races_df[races_df['horse_name'] == selected_horse]
        
        st.subheader(f"🏇 {selected_horse} - Deep Analysis")
        
        # Horse overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Races", horse_data['total_races'])
            st.metric("Wins", horse_data['wins'])
        with col2:
            st.metric("Average Ragozin", f"{horse_data['avg_ragozin']:.1f}")
            st.metric("Best Ragozin", f"{horse_data['best_ragozin']:.1f}")
        with col3:
            st.metric("Top 3 Finishes", horse_data['top3'])
            st.metric("Win Rate", f"{(horse_data['wins']/horse_data['total_races']*100):.1f}%" if horse_data['total_races'] > 0 else "0%")
        with col4:
            st.metric("Top 3 Rate", f"{(horse_data['top3']/horse_data['total_races']*100):.1f}%" if horse_data['total_races'] > 0 else "0%")
        
        # AI Analysis
        st.subheader("🤖 AI Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Overall Performance Analysis:**")
            st.info(horse_data['horse_analysis'])
        
        with col2:
            st.write("**Performance Trend Analysis:**")
            st.info(horse_data['performance_trend'])
        
        # Performance trends
        if len(horse_races) > 1:
            st.subheader("📈 Performance Trends")
            
            # Sort by date
            horse_races_sorted = horse_races.sort_values('race_date')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Ragozin figure trend
                fig = px.line(
                    horse_races_sorted,
                    x='race_date',
                    y='ragozin_figure',
                    title=f"{selected_horse} - Ragozin Figure Trend",
                    labels={'race_date': 'Date', 'ragozin_figure': 'Ragozin Figure'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Finish position trend
                fig = px.line(
                    horse_races_sorted,
                    x='race_date',
                    y='finish_position',
                    title=f"{selected_horse} - Finish Position Trend",
                    labels={'race_date': 'Date', 'finish_position': 'Finish Position'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Symbol analysis
        st.subheader("🔍 Symbol Analysis")
        
        # Count symbols
        symbol_before_counts = horse_races['symbol_before'].value_counts()
        symbol_after_counts = horse_races['symbol_after'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(symbol_before_counts) > 0:
                st.write("**Symbols Before Ragozin Figures:**")
                fig = px.pie(
                    values=symbol_before_counts.values,
                    names=symbol_before_counts.index,
                    title="Symbols Before Ragozin Figures"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No symbols found before Ragozin figures")
        
        with col2:
            if len(symbol_after_counts) > 0:
                st.write("**Symbols After Ragozin Figures:**")
                fig = px.pie(
                    values=symbol_after_counts.values,
                    names=symbol_after_counts.index,
                    title="Symbols After Ragozin Figures"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No symbols found after Ragozin figures")
        
        # Detailed race analysis
        st.subheader("📋 Detailed Race Analysis")
        
        for _, race in horse_races.iterrows():
            with st.expander(f"Race: {race['race_date']} - {race['track']} - {race['race_type']} - Ragozin: {race['ragozin_figure']:.1f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Race Details:**")
                    st.write(f"**Date:** {race['race_date']}")
                    st.write(f"**Track:** {race['track']}")
                    st.write(f"**Race Type:** {race['race_type']}")
                    st.write(f"**Surface:** {race['surface']}")
                    st.write(f"**Distance:** {race['distance']}")
                    st.write(f"**Jockey:** {race['jockey']}")
                    st.write(f"**Weight:** {race['weight']}")
                    st.write(f"**Odds:** {race['odds']}")
                
                with col2:
                    st.write("**Performance:**")
                    st.write(f"**Ragozin Figure:** {race['ragozin_figure']:.1f}")
                    st.write(f"**Finish Position:** {race['finish_position']}")
                    if race['symbol_before']:
                        st.write(f"**Symbol Before:** {race['symbol_before']}")
                    if race['symbol_after']:
                        st.write(f"**Symbol After:** {race['symbol_after']}")
                    if race['comments']:
                        st.write(f"**Comments:** {race['comments']}")
                
                if race['race_analysis']:
                    st.write("**AI Race Analysis:**")
                    st.info(race['race_analysis'])

def race_analysis_page():
    st.header("🏁 Race Analysis")
    
    # Check if we have data
    if 'races_df' not in st.session_state:
        st.info("Please go to 'Horses Overview' first to load the data.")
        return
    
    races_df = st.session_state.races_df
    
    if len(races_df) == 0:
        st.warning("No race data available.")
        return
    
    st.subheader("📊 Overall Race Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Races", len(races_df))
    with col2:
        st.metric("Unique Horses", races_df['horse_name'].nunique())
    with col3:
        st.metric("Unique Tracks", races_df['track'].nunique())
    with col4:
        st.metric("Date Range", f"{races_df['race_date'].min()} to {races_df['race_date'].max()}")
    
    # Analysis filters
    st.subheader("🔍 Analysis Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        track_filter = st.multiselect(
            "Select tracks:",
            options=sorted(races_df['track'].unique()),
            default=sorted(races_df['track'].unique())
        )
    
    with col2:
        surface_filter = st.multiselect(
            "Select surfaces:",
            options=sorted(races_df['surface'].unique()),
            default=sorted(races_df['surface'].unique())
        )
    
    with col3:
        race_type_filter = st.multiselect(
            "Select race types:",
            options=sorted(races_df['race_type'].unique()),
            default=sorted(races_df['race_type'].unique())
        )
    
    # Apply filters
    filtered_races = races_df[
        (races_df['track'].isin(track_filter)) &
        (races_df['surface'].isin(surface_filter)) &
        (races_df['race_type'].isin(race_type_filter))
    ]
    
    # Charts
    st.subheader("📈 Race Analysis Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ragozin figure distribution
        fig = px.histogram(
            filtered_races,
            x='ragozin_figure',
            title="Ragozin Figure Distribution",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Finish position distribution
        fig = px.histogram(
            filtered_races,
            x='finish_position',
            title="Finish Position Distribution",
            nbins=10
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Symbol analysis
    st.subheader("🔍 Symbol Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol_before_counts = filtered_races['symbol_before'].value_counts()
        if len(symbol_before_counts) > 0:
            fig = px.bar(
                x=symbol_before_counts.index,
                y=symbol_before_counts.values,
                title="Symbols Before Ragozin Figures",
                labels={'x': 'Symbol', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No symbols found before Ragozin figures")
    
    with col2:
        symbol_after_counts = filtered_races['symbol_after'].value_counts()
        if len(symbol_after_counts) > 0:
            fig = px.bar(
                x=symbol_after_counts.index,
                y=symbol_after_counts.values,
                title="Symbols After Ragozin Figures",
                labels={'x': 'Symbol', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No symbols found after Ragozin figures")

def database_page():
    st.header("Cumulative Database")
    st.caption("Every uploaded PDF adds to this persistent store. Studies draw from the full history.")

    try:
        resp = api_get(f"/db/stats")
        if resp.status_code != 200:
            st.error("Could not fetch database stats.")
            return
        stats = resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is it running?")
        return

    # --- Summary metrics ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Sheets Horses", stats.get("sheets_horses", 0))
    with col2:
        st.metric("Sheets Lines", stats.get("sheets_lines", 0))
    with col3:
        st.metric("BRISNET Horses", stats.get("brisnet_horses", 0))
    with col4:
        st.metric("Reconciled Pairs", stats.get("reconciled_pairs", 0))
    with col5:
        st.metric("Coverage", f"{stats.get('coverage_pct', 0):.1f}%")

    st.divider()

    sh = stats.get("sheets_horses", 0)
    bh = stats.get("brisnet_horses", 0)
    rp = stats.get("reconciled_pairs", 0)
    conf = stats.get("confidence_breakdown", {})

    if sh > 0 or bh > 0:
        # --- Confidence breakdown ---
        st.subheader("Match Confidence Breakdown")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("HIGH", conf.get("high", 0))
        with c2:
            st.metric("MED", conf.get("medium", 0))
        with c3:
            st.metric("LOW", conf.get("low", 0))

        # --- Enrichment coverage pie ---
        st.subheader("Enrichment Coverage")
        coverage_data = {
            "Source": ["Sheets Only", "BRISNET Only", "Both (Reconciled)"],
            "Count": [max(sh - rp, 0), max(bh - rp, 0), rp],
        }
        import plotly.express as px
        fig = px.pie(
            pd.DataFrame(coverage_data),
            values="Count", names="Source",
            color_discrete_sequence=["#4A90D9", "#50C878", "#DAA520"],
        )
        fig.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # --- Collision warnings ---
        collisions = stats.get("collision_warnings", [])
        if collisions:
            st.warning(f"Collision warnings ({len(collisions)} names map to multiple horses): "
                        + ", ".join(collisions))

        # --- Unmatched names with alias workflow ---
        um_sh = stats.get("unmatched_sheets", [])
        um_bh = stats.get("unmatched_brisnet", [])
        if um_sh or um_bh:
            st.subheader("Unmatched Horses (top 20 each)")
            left, right = st.columns(2)
            with left:
                st.caption("Sheets (no BRISNET match)")
                if um_sh:
                    for h in um_sh:
                        st.text(f"{h['name']}  {h['track']} {h['date']}")
                else:
                    st.success("All sheets horses matched!")
            with right:
                st.caption("BRISNET (no Sheets match)")
                if um_bh:
                    for h in um_bh:
                        st.text(f"{h['name']}  {h['track']} {h['date']}")
                else:
                    st.success("All BRISNET horses matched!")

        # --- Alias resolution workflow ---
        st.subheader("Alias Resolution")
        st.caption("Create aliases to link unmatched BRISNET horses to their Sheets counterpart.")

        # Fetch unmatched with suggestions
        try:
            um_resp = api_get(f"/db/unmatched", timeout=15)
            unmatched_data = um_resp.json() if um_resp.status_code == 200 else []
        except Exception:
            unmatched_data = []

        if not unmatched_data:
            st.success("No unmatched BRISNET horses!")
        else:
            for i, um in enumerate(unmatched_data):
                bname = um["horse_name"]
                suggs = um.get("suggestions", [])
                with st.expander(
                    f"R{um['race_number']} P{um['post']}  {bname}"
                    + (f"  -- top suggestion: {suggs[0]['horse_name']} ({suggs[0]['score']}%)"
                       if suggs else "  -- no suggestion"),
                    expanded=False,
                ):
                    if suggs:
                        st.markdown("**Suggested matches:**")
                        for j, s in enumerate(suggs):
                            c1, c2, c3 = st.columns([3, 1, 1])
                            with c1:
                                st.text(f"{s['horse_name']} (score {s['score']}%)")
                            with c2:
                                if st.button(
                                    "Create alias",
                                    key=f"alias_{i}_{j}",
                                ):
                                    try:
                                        r = requests.post(
                                            f"{API_BASE_URL}/db/alias",
                                            json={
                                                "canonical": s["horse_name"],
                                                "alias": bname,
                                            },
                                            timeout=15,
                                        )
                                        if r.status_code == 200:
                                            recon = r.json().get("reconciliation", {})
                                            st.success(
                                                f"Alias created: {bname} -> {s['horse_name']}. "
                                                f"New matches: {recon.get('new_matches', 0)}"
                                            )
                                            st.rerun()
                                        else:
                                            st.error(r.text)
                                    except Exception as e:
                                        st.error(str(e))
                    else:
                        st.info("No fuzzy matches found in sheets data.")

                    # Manual entry fallback
                    manual = st.text_input(
                        "Or type sheets horse name:",
                        key=f"manual_{i}",
                    )
                    if manual and st.button("Link manually", key=f"manual_btn_{i}"):
                        try:
                            r = requests.post(
                                f"{API_BASE_URL}/db/alias",
                                json={"canonical": manual, "alias": bname},
                                timeout=15,
                            )
                            if r.status_code == 200:
                                recon = r.json().get("reconciliation", {})
                                st.success(
                                    f"Alias created. New matches: {recon.get('new_matches', 0)}"
                                )
                                st.rerun()
                            else:
                                st.error(r.text)
                        except Exception as e:
                            st.error(str(e))

    # --- Tracks ---
    tracks = stats.get("tracks", [])
    if tracks:
        st.subheader("Tracks in Database")
        st.write(", ".join(tracks))

    # --- Upload count ---
    uploads_count = stats.get("uploads_count", 0)
    if uploads_count > 0:
        st.subheader(f"Uploads ({uploads_count})")
        st.info(f"{uploads_count} PDF uploads in the database across {len(tracks)} tracks.")

    if sh == 0 and bh == 0:
        st.info("Database is empty. Upload PDFs via the Upload page or CLI:\n\n"
                "`python ingest.py --sheets <pdf>`\n\n"
                "`python ingest.py --brisnet <pdf>`\n\n"
                "`python ingest.py --backfill-json` (import existing output/*.json)")


_LOW_SAMPLE_THRESHOLD = 30


def _render_roi_table(title: str, data: list, label_key: str, label_col: str):
    """Render a single ROI table with N column and sample-size warnings.

    *data* is a list of dicts. Each dict has *label_key*, plus either
    "N"/"bets" for count, "wins", "win_pct", "roi_pct".
    """
    if not data:
        return
    st.subheader(title)

    rows = []
    has_low = False
    for d in data:
        n = d.get("N", d.get("bets", 0))
        low = n < _LOW_SAMPLE_THRESHOLD
        if low:
            has_low = True
        rows.append({
            label_col: d.get(label_key, ""),
            "N": n,
            "Wins": d.get("wins", 0),
            "Win%": d.get("win_pct", 0),
            "ROI%": d.get("roi_pct", 0),
            "_low": low,
        })

    df = pd.DataFrame(rows)

    # Style low-sample rows
    def _grey_low(row):
        if row["_low"]:
            return ["color: #999; font-style: italic"] * len(row)
        return [""] * len(row)

    display_cols = [label_col, "N", "Wins", "Win%", "ROI%"]
    styled = df[display_cols + ["_low"]].style.apply(_grey_low, axis=1)
    styled = styled.hide(axis="columns", subset=["_low"])
    styled = styled.format({"Win%": "{:.1f}", "ROI%": "{:+.1f}"})
    st.dataframe(styled, hide_index=True, use_container_width=True)

    if has_low:
        st.caption("*Greyed rows: N < 30 — LOW SAMPLE, DO NOT TRUST*")


def _render_recommendations(roi_cycle: list, roi_conf: list, roi_odds: list):
    """Show a 'Recommended settings' panel based on ROI data."""
    st.subheader("Recommended Settings")
    st.caption("Suggestions based on historical ROI. These do NOT auto-change any settings.")

    suggestions = []

    # Best confidence bucket (only consider N >= 30)
    valid_conf = [c for c in roi_conf
                  if c.get("N", c.get("bets", 0)) >= _LOW_SAMPLE_THRESHOLD]
    if valid_conf:
        best_conf = max(valid_conf, key=lambda x: x.get("roi_pct", 0))
        bucket = best_conf.get("confidence", "")
        roi = best_conf.get("roi_pct", 0)
        n = best_conf.get("N", best_conf.get("bets", 0))
        if roi > 0:
            suggestions.append(
                f"**Confidence {bucket}%** has best ROI at **{roi:+.1f}%** (N={n})"
            )

    # Best odds bucket
    valid_odds = [o for o in roi_odds
                  if o.get("N", o.get("bets", 0)) >= _LOW_SAMPLE_THRESHOLD
                  and o.get("odds", "") != "No Odds"]
    if valid_odds:
        best_odds = max(valid_odds, key=lambda x: x.get("roi_pct", 0))
        bucket = best_odds.get("odds", "")
        roi = best_odds.get("roi_pct", 0)
        n = best_odds.get("N", best_odds.get("bets", 0))
        if roi > 0:
            suggestions.append(
                f"**Odds {bucket}** has best ROI at **{roi:+.1f}%** (N={n})"
            )

    # Best cycle pattern
    valid_cycle = [c for c in roi_cycle
                   if c.get("N", c.get("bets", 0)) >= _LOW_SAMPLE_THRESHOLD]
    if valid_cycle:
        best_cycle = max(valid_cycle, key=lambda x: x.get("roi_pct", 0))
        pattern = best_cycle.get("cycle", "")
        roi = best_cycle.get("roi_pct", 0)
        n = best_cycle.get("N", best_cycle.get("bets", 0))
        if roi > 0:
            suggestions.append(
                f"**{pattern}** cycle has best ROI at **{roi:+.1f}%** (N={n})"
            )

    # Combined suggestion
    if valid_conf and valid_odds:
        bc = max(valid_conf, key=lambda x: x.get("roi_pct", 0))
        bo = max(valid_odds, key=lambda x: x.get("roi_pct", 0))
        if bc.get("roi_pct", 0) > 0 and bo.get("roi_pct", 0) > 0:
            conf_label = bc.get("confidence", "")
            odds_label = bo.get("odds", "")
            suggestions.append(
                f"Combined suggestion: **confidence >= {conf_label}%**, **odds {odds_label}**"
            )

    if suggestions:
        for s in suggestions:
            st.markdown(f"- {s}")
    else:
        st.info("Not enough data with N >= 30 to make recommendations yet.")


def _render_exports(roi_params: dict, det: dict,
                    roi_rank: list, roi_cycle: list, roi_conf: list,
                    roi_odds: list, roi_surface: list, roi_distance: list):
    """Render CSV export buttons for ROI tables and full bets list."""
    st.subheader("Exports")

    col1, col2 = st.columns(2)

    # --- ROI tables CSV ---
    with col1:
        roi_sections = []
        for label, data, key in [
            ("Pick Rank", roi_rank, "rank"),
            ("Cycle Pattern", roi_cycle, "cycle"),
            ("Confidence", roi_conf, "confidence"),
            ("Odds Bucket", roi_odds, "odds"),
            ("Surface", roi_surface, "surface"),
            ("Distance", roi_distance, "distance"),
        ]:
            if data:
                roi_sections.append(f"\n# {label}")
                header_label = key.title()
                roi_sections.append(f"{header_label},N,Wins,Win%,ROI%")
                for d in data:
                    n = d.get("N", d.get("bets", 0))
                    roi_sections.append(
                        f"{d.get(key, '')},{n},{d.get('wins', 0)},"
                        f"{d.get('win_pct', 0):.1f},{d.get('roi_pct', 0):+.1f}"
                    )

        if roi_sections:
            csv_text = "\n".join(roi_sections)
            st.download_button(
                "Download ROI Tables (CSV)",
                data=csv_text,
                file_name="roi_tables.csv",
                mime="text/csv",
                key="dl_roi_tables",
            )
        else:
            st.caption("No ROI data to export.")

    # --- Full bets/picks audit CSV ---
    with col2:
        try:
            bets_resp = requests.get(
                f"{API_BASE_URL}/predictions/export-bets",
                params=roi_params, timeout=30,
            )
            bets = bets_resp.json() if bets_resp.status_code == 200 else []
        except Exception:
            bets = []

        if bets:
            bets_df = pd.DataFrame(bets)
            csv_bytes = bets_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Download All Bets ({len(bets)} picks, CSV)",
                data=csv_bytes,
                file_name="all_bets_audit.csv",
                mime="text/csv",
                key="dl_all_bets",
            )
        else:
            st.caption("No bets data to export.")


def _show_ingest_result(res: dict, session_id: str = ""):
    """Display ingestion result with linking diagnostics."""
    link_rate = res.get("link_rate", 0)
    st.success(
        f"Imported {res.get('races', 0)} races, {res.get('entries', 0)} entries, "
        f"{res.get('linked', 0)} linked ({link_rate:.1f}%)"
    )

    # Confidence caption for PDF
    if res.get("parse_confidence") is not None:
        st.caption(f"Parse confidence: {res['parse_confidence'] * 100:.0f}%")

    # Linking warning when session is attached and rate is low
    if session_id and link_rate < 70:
        st.warning(
            f"Linking rate is {link_rate:.1f}% (< 70%). Some results could not be matched "
            "to session entries. Check that race numbers and post/program positions align. "
            "You can also try switching the **Identifier mapping** between post and program."
        )

    # Unmatched entries with reasons
    unmatched = res.get("unmatched", [])
    if unmatched:
        with st.expander(f"Unmatched entries ({len(unmatched)})", expanded=link_rate < 70):
            for u in unmatched:
                st.text(f"R{u['race']} P{u['post']} {u['name']}: {u['reason']}")


def results_page():
    st.header("Race Results & ROI")
    st.caption("Upload race results CSVs or Equibase chart PDFs to track ROI by pick rank and cycle pattern.")

    # --- Fetch sessions for dropdown ---
    session_options = {"(none)": ""}
    try:
        sr = api_get(f"/sessions", timeout=10)
        if sr.status_code == 200:
            for s in sr.json().get("sessions", []):
                label = f"{s['track']} {s['date']} ({s['session_id'][:8]}...)"
                session_options[label] = s["session_id"]
    except Exception:
        pass

    # --- Upload CSV or PDF ---
    st.subheader("Import Results")
    results_files = st.file_uploader(
        "Upload results CSV or Equibase chart PDF (multiple files supported)",
        type=["csv", "pdf"], key="results_file_uploader",
        accept_multiple_files=True,
    )

    sel_session = st.selectbox(
        "Attach to session (links results to picks)",
        options=list(session_options.keys()),
        key="results_session_select",
    )
    chosen_sid = session_options.get(sel_session, "")

    col_t, col_d = st.columns(2)
    with col_t:
        r_track = st.text_input("Track (e.g. GP)", key="results_track")
    with col_d:
        r_date = st.text_input("Date (MM/DD/YYYY)", key="results_date")

    # Program vs post selector (relevant when chart program != session post)
    pgm_map = st.radio(
        "Identifier mapping",
        options=["post", "program"],
        horizontal=True,
        index=0,
        help="Use 'program' if the chart uses program numbers (1, 1A) that differ from session post positions.",
        key="results_pgm_map",
    )

    if results_files and st.button("Import Results", type="primary", key="btn_import_results"):
        _pending_review = None  # track first PDF needing review
        imported_count = 0
        for rf in results_files:
            with st.spinner(f"Importing {rf.name}..."):
                try:
                    fname = rf.name.lower()
                    mime = "text/csv" if fname.endswith(".csv") else "application/pdf"
                    files_payload = {"file": (rf.name, rf.getvalue(), mime)}
                    params = {}
                    if r_track:
                        params["track"] = r_track
                    if r_date:
                        params["date"] = r_date
                    if chosen_sid:
                        params["session_id"] = chosen_sid
                    resp = requests.post(
                        f"{API_BASE_URL}/results/upload",
                        files=files_payload, params=params, timeout=60,
                    )
                    if resp.status_code == 200:
                        res = resp.json()
                        is_pdf = fname.endswith(".pdf")

                        if is_pdf and res.get("needs_review") and _pending_review is None:
                            _pending_review = (rf.name, res)
                            st.info(f"{rf.name}: needs manual review (see below)")
                        elif is_pdf and res.get("ingested"):
                            _show_ingest_result(res, chosen_sid)
                            imported_count += 1
                        elif is_pdf and (res.get("sample_rows") or res.get("needs_review")):
                            # High-confidence PDF preview
                            conf = res.get("parse_confidence", 0)
                            st.info(f"{rf.name}: PDF parsed (confidence {conf*100:.0f}%)")
                            if not res.get("needs_review"):
                                _show_ingest_result(res, chosen_sid)
                                imported_count += 1
                        elif not is_pdf:
                            _show_ingest_result(res, chosen_sid)
                            imported_count += 1
                    else:
                        st.error(f"{rf.name}: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API.")
                except Exception as e:
                    st.error(f"{rf.name}: {e}")

        if imported_count > 0:
            st.success(f"Imported {imported_count} file(s) successfully")

        # Show review UI for the first PDF that needs manual review
        if _pending_review:
            review_name, res = _pending_review
            conf = res.get("parse_confidence", 0)
            conf_pct = conf * 100

            st.divider()
            st.warning(
                f"**{review_name}** — Confidence {conf_pct:.0f}% is below the auto-import threshold. "
                "Review and correct the data below, then click **Confirm & Import**."
            )

            # Preview
            sample = res.get("sample_rows", [])
            if sample:
                display_cols = ["race_number", "program", "post", "horse_name",
                                "finish_pos", "odds", "win_payoff", "surface"]
                sdf = pd.DataFrame(sample)
                show_cols = [c for c in display_cols if c in sdf.columns]
                st.dataframe(sdf[show_cols], use_container_width=True, hide_index=True)

            missing = res.get("missing_fields", [])
            if missing:
                with st.expander(f"Missing fields ({len(missing)})", expanded=True):
                    for mf in missing:
                        st.text(f"  - {mf}")

            extracted = res.get("extracted_rows", [])
            if extracted:
                df = pd.DataFrame(extracted)
                edited_df = st.data_editor(
                    df, num_rows="dynamic", key="pdf_review_editor",
                )

                with st.expander("Raw extracted text", expanded=False):
                    st.text(res.get("raw_text", "")[:5000])

                if st.button("Confirm & Import", type="primary", key="btn_confirm_pdf"):
                    with st.spinner("Importing confirmed results..."):
                        confirm_rows = edited_df.to_dict(orient="records")
                        confirm_resp = requests.post(
                            f"{API_BASE_URL}/results/confirm-pdf",
                            json={
                                "rows": confirm_rows,
                                "track": res.get("track", r_track),
                                "date": res.get("race_date", r_date),
                                "session_id": chosen_sid,
                                "program_map": pgm_map,
                            },
                            timeout=60,
                        )
                        if confirm_resp.status_code == 200:
                            cr = confirm_resp.json()
                            _show_ingest_result(cr, chosen_sid)
                        else:
                            st.error(f"Error: {confirm_resp.text}")
            else:
                st.error("No data could be extracted from the PDF.")

    st.divider()

    # --- Stats ---
    try:
        params = {}
        if r_track:
            params["track"] = r_track
        resp = api_get(f"/results/stats", params=params, timeout=15)
        if resp.status_code != 200:
            st.info("No results data yet. Upload a results CSV above.")
            return
        stats = resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")
        return
    except Exception as e:
        st.error(f"Error: {e}")
        return

    if stats.get("total_entries", 0) == 0:
        st.info("No results in database. Upload a results CSV to see ROI analysis.\n\n"
                "CLI: `python ingest_results.py --csv results.csv --track GP --date 02/26/2026`")
        return

    # Summary metrics
    render_metrics_row({
        "Races": stats["total_races"],
        "Entries": stats["total_entries"],
        "With Odds": stats["total_with_odds"],
        "Linking Rate": f"{stats['linking_rate']:.1f}%",
    })

    st.divider()

    # --- Fetch detailed ROI from predictions ---
    roi_params = {}
    if r_track:
        roi_params["track"] = r_track
    if r_date:
        roi_params["date"] = r_date
    if chosen_sid:
        roi_params["session_id"] = chosen_sid

    try:
        det_resp = requests.get(
            f"{API_BASE_URL}/predictions/roi-detailed",
            params=roi_params, timeout=15,
        )
        det = det_resp.json() if det_resp.status_code == 200 else {}
    except Exception:
        det = {}

    # Also fetch legacy simple ROI for fallback
    try:
        roi_resp = requests.get(
            f"{API_BASE_URL}/predictions/roi",
            params=roi_params, timeout=15,
        )
        pred_roi = roi_resp.json() if roi_resp.status_code == 200 else {}
    except Exception:
        pred_roi = {}

    has_detailed = det.get("matched_with_results", 0) > 0
    has_pred_roi = pred_roi.get("total_predictions", 0) > 0

    if has_detailed:
        render_section_header("Predictions vs Results")
        render_metrics_row({
            "Predictions": det["total_predictions"],
            "Matched w/ Results": det["matched_with_results"],
            "Match Rate": f"{det['match_rate']:.1f}%",
        })

        # ---------- ROI tables with sample-size warnings ----------

        # ROI by Pick Rank (from simple endpoint — has rank column)
        roi_rank = pred_roi.get("roi_by_rank", [])
        if roi_rank:
            _render_roi_table(
                "ROI by Pick Rank ($2 Win)", roi_rank,
                label_key="rank", label_col="Rank",
            )

        # ROI by Cycle Label
        roi_cycle = det.get("roi_by_cycle", [])
        if roi_cycle:
            _render_roi_table(
                "ROI by Cycle Pattern ($2 Win)", roi_cycle,
                label_key="cycle", label_col="Pattern",
            )

        # ROI by Confidence Bucket
        roi_conf = det.get("roi_by_confidence", [])
        if roi_conf:
            _render_roi_table(
                "ROI by Confidence ($2 Win)", roi_conf,
                label_key="confidence", label_col="Confidence",
            )

        # ROI by Odds Bucket
        roi_odds = det.get("roi_by_odds", [])
        if roi_odds:
            _render_roi_table(
                "ROI by Odds Bucket ($2 Win)", roi_odds,
                label_key="odds", label_col="Odds",
            )

        # ROI by Surface
        roi_surface = det.get("roi_by_surface", [])
        if roi_surface:
            _render_roi_table(
                "ROI by Surface ($2 Win)", roi_surface,
                label_key="surface", label_col="Surface",
            )

        # ROI by Distance
        roi_distance = det.get("roi_by_distance", [])
        if roi_distance:
            _render_roi_table(
                "ROI by Distance ($2 Win)", roi_distance,
                label_key="distance", label_col="Distance",
            )

        st.divider()

        # ---------- Recommended settings panel ----------
        _render_recommendations(roi_cycle, roi_conf, roi_odds)

        st.divider()

        # ---------- Detailed predictions table ----------
        try:
            pvr_resp = requests.get(
                f"{API_BASE_URL}/predictions/vs-results",
                params=roi_params, timeout=15,
            )
            pvr = pvr_resp.json() if pvr_resp.status_code == 200 else []
        except Exception:
            pvr = []

        if pvr:
            with st.expander(f"Full predictions table ({len(pvr)} entries)", expanded=False):
                pvr_rows = []
                for r in pvr:
                    fp = r.get("finish_pos")
                    win_p = r.get("win_payoff")
                    pvr_rows.append({
                        "Race": r["race_number"],
                        "Horse": r["horse_name"],
                        "Rank": r["pick_rank"],
                        "Cycle": r["projection_type"],
                        "Score": f"{r['bias_score']:.1f}" if r.get("bias_score") else "",
                        "Conf": f"{r['confidence']:.0%}" if r.get("confidence") else "",
                        "Odds": f"{r['odds']:.1f}" if r.get("odds") else "-",
                        "Finish": fp if fp else "-",
                        "Win$": f"${win_p:.2f}" if win_p else "",
                    })
                pvr_df = pd.DataFrame(pvr_rows)
                st.dataframe(pvr_df, hide_index=True, use_container_width=True)

        st.divider()

        # ---------- Exports ----------
        _render_exports(roi_params, det, roi_rank, roi_cycle, roi_conf, roi_odds,
                        roi_surface, roi_distance)

    elif has_pred_roi:
        # Simpler display when detailed endpoint returned empty but simple has data
        st.subheader("Predictions vs Results")
        pm1, pm2, pm3 = st.columns(3)
        with pm1:
            st.metric("Predictions", pred_roi["total_predictions"])
        with pm2:
            st.metric("Matched w/ Results", pred_roi["matched_with_results"])
        with pm3:
            st.metric("Match Rate", f"{pred_roi['match_rate']:.1f}%")

        roi_rank = pred_roi.get("roi_by_rank", [])
        if roi_rank:
            _render_roi_table("ROI by Pick Rank ($2 Win)", roi_rank,
                              label_key="rank", label_col="Rank")
        roi_cycle = pred_roi.get("roi_by_cycle", [])
        if roi_cycle:
            _render_roi_table("ROI by Cycle Pattern ($2 Win)", roi_cycle,
                              label_key="cycle", label_col="Pattern")

    else:
        # Fallback: result_entries-based ROI (legacy)
        roi_rank = stats.get("roi_by_rank", [])
        if roi_rank:
            _render_roi_table("ROI by Pick Rank ($2 Win)", roi_rank,
                              label_key="rank", label_col="Rank")
        roi_cycle = stats.get("roi_by_cycle", [])
        if roi_cycle:
            _render_roi_table("ROI by Cycle Pattern ($2 Win)", roi_cycle,
                              label_key="cycle", label_col="Pattern")
        if not roi_rank and not roi_cycle:
            st.info("No predictions saved yet. Run the engine on a session, then upload results.")


def calibration_page():
    st.header("Track / Surface Calibration")
    st.caption("ROI analysis by track, surface, and distance. "
               "Find optimal confidence and odds thresholds for each spot type.")

    col1, col2 = st.columns(2)
    with col1:
        cal_track = st.text_input("Filter by track (blank = all)", key="cal_track")
    with col2:
        cal_min_n = st.number_input("Min sample size", value=30, min_value=10, step=5, key="cal_min_n")

    if st.button("Load Calibration Data", type="primary", key="btn_calibration"):
        with st.spinner("Loading calibration data..."):
            try:
                params = {"min_n": cal_min_n}
                if cal_track:
                    params["track"] = cal_track
                resp = api_get(f"/calibration/roi", params=params, timeout=30)
                if resp.status_code == 200:
                    st.session_state["cal_data"] = resp.json()
                else:
                    st.error(f"Error: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is the backend running?")

    data = st.session_state.get("cal_data")
    if not data:
        return

    buckets = data.get("buckets", [])
    recommendations = data.get("recommendations", [])
    min_n_used = data.get("min_n", 30)

    if not buckets:
        st.info("No data available. Upload results and run the engine first.")
        return

    # --- Buckets table ---
    st.subheader("ROI by Track / Surface / Distance")
    bucket_rows = []
    for b in buckets:
        bucket_rows.append({
            "Spot": b["label"],
            "N": b["N"],
            "Wins": b["wins"],
            "Win%": f"{b['win_pct']:.1f}",
            "ROI%": f"{b['roi_pct']:+.1f}",
            "_low": b["N"] < min_n_used,
        })

    df_buckets = pd.DataFrame(bucket_rows)

    def _grey_low(row):
        if row["_low"]:
            return ["color: #999; font-style: italic"] * len(row)
        return [""] * len(row)

    display_cols = ["Spot", "N", "Wins", "Win%", "ROI%"]
    styled = df_buckets[display_cols + ["_low"]].style.apply(_grey_low, axis=1)
    styled = styled.hide(axis="columns", subset=["_low"])
    st.dataframe(styled, hide_index=True, use_container_width=True)

    if any(r["_low"] for r in bucket_rows):
        st.caption(f"*Greyed rows: N < {min_n_used} -- LOW SAMPLE, DO NOT TRUST*")

    # --- Recommendations ---
    if recommendations:
        st.divider()
        st.subheader("Optimal Thresholds per Spot")
        st.caption("Best-performing confidence and odds thresholds found via grid search (N >= min sample).")

        rec_rows = []
        for r in recommendations:
            rec_rows.append({
                "Spot": r["label"],
                "N": r["total_N"],
                "Base ROI%": f"{r['total_roi']:+.1f}",
                "Best Conf >=": f"{r['best_conf_threshold']:.0%}",
                "Conf ROI%": f"{r['best_conf_roi']:+.1f}",
                "Conf N": r["best_conf_n"],
                "Best Odds >=": f"{r['best_odds_threshold']:.1f}-1",
                "Odds ROI%": f"{r['best_odds_roi']:+.1f}",
                "Odds N": r["best_odds_n"],
            })

        st.dataframe(pd.DataFrame(rec_rows), hide_index=True, use_container_width=True)

        # Highlight profitable spots
        profitable = [r for r in recommendations if r["best_conf_roi"] > 0 or r["best_odds_roi"] > 0]
        if profitable:
            st.divider()
            st.subheader("Actionable Suggestions")
            for r in profitable:
                parts = []
                if r["best_conf_roi"] > 0:
                    parts.append(
                        f"confidence >= {r['best_conf_threshold']:.0%} "
                        f"(ROI {r['best_conf_roi']:+.1f}%, N={r['best_conf_n']})"
                    )
                if r["best_odds_roi"] > 0:
                    parts.append(
                        f"odds >= {r['best_odds_threshold']:.1f}-1 "
                        f"(ROI {r['best_odds_roi']:+.1f}%, N={r['best_odds_n']})"
                    )
                st.markdown(f"- **{r['label']}**: {' | '.join(parts)}")

    # --- Detailed ROI Dashboard ---
    st.divider()
    st.header("Detailed ROI Breakdown")
    st.caption("ROI by cycle pattern, confidence tier, odds bucket, and pick rank (top pick = rank 1).")

    if st.button("Load Detailed ROI", key="btn_detailed_roi"):
        with st.spinner("Loading detailed ROI data..."):
            try:
                d_params = {}
                if cal_track:
                    d_params["track"] = cal_track
                d_resp = api_get("/calibration/detailed-roi", params=d_params, timeout=30)
                if d_resp.status_code == 200:
                    st.session_state["detailed_roi"] = d_resp.json()
                else:
                    st.error(f"Error: {d_resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API.")

    d_roi = st.session_state.get("detailed_roi")
    if d_roi:
        # By Cycle
        by_cycle = d_roi.get("by_cycle", [])
        if by_cycle:
            st.subheader("ROI by Cycle Pattern (Top Pick)")
            cycle_rows = []
            for c in by_cycle:
                cycle_rows.append({
                    "Cycle": c["cycle"],
                    "N": c["N"],
                    "Wins": c["wins"],
                    "Win%": f"{c['win_pct']:.1f}",
                    "ROI%": f"{c['roi_pct']:+.1f}",
                })
            st.dataframe(pd.DataFrame(cycle_rows), hide_index=True, use_container_width=True)

        # By Confidence
        by_conf = d_roi.get("by_confidence", [])
        if by_conf:
            st.subheader("ROI by Confidence Tier (Top Pick)")
            conf_rows = []
            for c in by_conf:
                conf_rows.append({
                    "Tier": c["tier"],
                    "N": c["N"],
                    "Wins": c["wins"],
                    "Win%": f"{c['win_pct']:.1f}",
                    "ROI%": f"{c['roi_pct']:+.1f}",
                })
            st.dataframe(pd.DataFrame(conf_rows), hide_index=True, use_container_width=True)

        # By Odds Bucket
        by_odds = d_roi.get("by_odds", [])
        if by_odds:
            st.subheader("ROI by Odds Bucket (Top Pick)")
            odds_rows = []
            for o in by_odds:
                odds_rows.append({
                    "Odds": o["bucket"],
                    "N": o["N"],
                    "Wins": o["wins"],
                    "Win%": f"{o['win_pct']:.1f}",
                    "ROI%": f"{o['roi_pct']:+.1f}",
                })
            st.dataframe(pd.DataFrame(odds_rows), hide_index=True, use_container_width=True)

        # By Pick Rank
        by_rank = d_roi.get("by_rank", [])
        if by_rank:
            st.subheader("ROI by Pick Rank")
            rank_rows = []
            for r in by_rank:
                rank_rows.append({
                    "Rank": f"#{r['rank']}",
                    "N": r["N"],
                    "Wins": r["wins"],
                    "Win%": f"{r['win_pct']:.1f}",
                    "ROI%": f"{r['roi_pct']:+.1f}",
                })
            st.dataframe(pd.DataFrame(rank_rows), hide_index=True, use_container_width=True)

        # Top Leaks
        leaks = d_roi.get("leaks", [])
        if leaks:
            st.divider()
            st.subheader("Top Leaks (Worst ROI Spots)")
            for lk in leaks:
                st.markdown(
                    f"- **{lk['spot']}**: ROI {lk['roi_pct']:+.1f}%, "
                    f"Win% {lk['win_pct']:.1f}%, N={lk['N']}"
                )

        # Tuning Suggestions
        suggestions = d_roi.get("suggestions", [])
        if suggestions:
            st.divider()
            st.subheader("Tuning Suggestions")
            for s in suggestions:
                st.markdown(f"- {s}")

    # --- Export ---
    st.divider()
    st.subheader("Export")
    if buckets:
        csv_lines = ["spot,track,surface,distance,N,wins,win_pct,roi_pct"]
        for b in buckets:
            csv_lines.append(
                f"{b['label']},{b['track']},{b['surface']},{b['distance']},"
                f"{b['N']},{b['wins']},{b['win_pct']:.1f},{b['roi_pct']:+.1f}"
            )
        st.download_button("Download Calibration CSV", "\n".join(csv_lines),
                           "calibration.csv", "text/csv", key="dl_calibration")


def results_inbox_page():
    st.header("Results Inbox")
    st.caption("Unattached race results awaiting session linkage. One-click attach to matching sessions.")

    try:
        resp = api_get(f"/results/inbox", timeout=15)
        if resp.status_code != 200:
            st.error(f"Error: {resp.text}")
            return
        data = resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is the backend running?")
        return

    unattached = data.get("unattached_races", [])
    matching = data.get("matching_sessions", {})

    if not unattached:
        st.success("All results are attached to sessions. Nothing to do.")
        return

    # Group by (track, date)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in unattached:
        groups[(r["track"], r["race_date"])].append(r)

    st.metric("Unattached Race Groups", len(groups))

    for (track, race_date), races in sorted(groups.items()):
        key = f"{track}_{race_date}"
        sessions_list = matching.get(key, [])
        total_unattached = sum(r["unattached"] for r in races)
        total_attached = sum(r["attached"] for r in races)

        with st.expander(
            f"{track} {race_date} -- {len(races)} races "
            f"({total_unattached} unattached, {total_attached} attached)",
            expanded=True,
        ):
            # Race details table
            rows = []
            for r in races:
                rows.append({
                    "Race": r["race_number"],
                    "Surface": r.get("surface", ""),
                    "Distance": r.get("distance", ""),
                    "Entries": r["total_entries"],
                    "Attached": r["attached"],
                    "Unattached": r["unattached"],
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            if sessions_list:
                if len(sessions_list) == 1:
                    s = sessions_list[0]
                    st.info(
                        f"Matching session: {s['session_id'][:8]}... "
                        f"({s.get('parser_used', '')} | {s.get('horses_count', 0)} horses)"
                    )
                    if st.button(
                        f"Attach to {s['session_id'][:8]}...",
                        key=f"attach_{track}_{race_date}",
                    ):
                        with st.spinner("Attaching..."):
                            attach_resp = requests.post(
                                f"{API_BASE_URL}/results/attach",
                                json={"track": track, "race_date": race_date,
                                      "session_id": s["session_id"]},
                                timeout=30,
                            )
                            if attach_resp.status_code == 200:
                                ar = attach_resp.json()
                                st.success(
                                    f"Attached {ar.get('updated_entries', 0)} entries."
                                )
                                st.rerun()
                            else:
                                st.error(f"Error: {attach_resp.text}")
                else:
                    st.info(f"{len(sessions_list)} matching sessions found. Choose one:")
                    options = {
                        f"{s['session_id'][:8]}... ({s.get('parser_used', '')} | "
                        f"{s.get('horses_count', 0)} horses | {s.get('created_at', '')[:10]})":
                        s["session_id"]
                        for s in sessions_list
                    }
                    choice = st.selectbox(
                        "Session", list(options.keys()),
                        key=f"inbox_select_{track}_{race_date}",
                    )
                    if st.button("Attach", key=f"attach_{track}_{race_date}"):
                        sid = options[choice]
                        with st.spinner("Attaching..."):
                            attach_resp = requests.post(
                                f"{API_BASE_URL}/results/attach",
                                json={"track": track, "race_date": race_date,
                                      "session_id": sid},
                                timeout=30,
                            )
                            if attach_resp.status_code == 200:
                                ar = attach_resp.json()
                                st.success(
                                    f"Attached {ar.get('updated_entries', 0)} entries."
                                )
                                st.rerun()
                            else:
                                st.error(f"Error: {attach_resp.text}")
            else:
                st.warning("No matching session found. Upload sheets for this track and date first.")


def _render_exotic_table(title: str, plays: list, multi_race: bool = False):
    """Render a single exotic recommendations table (DD, EX, or TRI)."""
    st.divider()
    st.subheader(title)

    active = [p for p in plays if not p.get("passed")]
    passed = [p for p in plays if p.get("passed")]

    if not active:
        st.info(f"No qualifying plays found.")
    else:
        rows = []
        for p in active:
            race_str = "-".join(str(r) for r in p.get("race_numbers", []))
            badges = ", ".join(p.get("reason_badges", [])) or "—"
            flags = ", ".join(p.get("risk_flags", [])) or "—"
            rows.append({
                "Track": p.get("track", ""),
                "Race(s)" if multi_race else "Race": race_str,
                "Ticket": p.get("ticket_desc", ""),
                "Conf": p.get("confidence", 0),
                "Edge Badges": badges,
                "Risk Flags": flags,
                "Cost": f"${p.get('cost', 0):.0f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if passed:
        with st.expander(f"PASS entries ({len(passed)})"):
            pass_rows = []
            for p in passed:
                race_str = "-".join(str(r) for r in p.get("race_numbers", []))
                pass_rows.append({
                    "Track": p.get("track", ""),
                    "Race(s)" if multi_race else "Race": race_str,
                    "Grade": p.get("grade", ""),
                    "Reason": p.get("pass_reason", ""),
                })
            st.dataframe(pd.DataFrame(pass_rows), hide_index=True, use_container_width=True)


def daily_wins_page():
    st.header("Daily Best WIN Bets")
    st.caption("Cross-track A-grade WIN bets ranked by value overlay, with Kelly sizing.")

    # --- Date input ---
    from datetime import date as _date
    dw_date = st.date_input("Race Date", value=_date.today(), key="dw_date")
    dw_date_str = dw_date.strftime("%m/%d/%Y") if dw_date else ""

    # --- Settings ---
    st.subheader("Settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        bankroll = st.number_input("Bankroll ($)", value=1000, min_value=100, step=100, key="dw_bankroll")
        risk_profile = st.selectbox("Risk Profile", ["conservative", "standard", "aggressive"],
                                    index=1, key="dw_risk_profile")
    with c2:
        max_day_pct = st.number_input("Max risk/day (%)", value=6.0, min_value=1.0, max_value=20.0,
                                      step=1.0, key="dw_max_day")
        min_confidence = st.number_input("Min confidence", value=0.65, min_value=0.0, max_value=1.0,
                                         step=0.05, key="dw_min_conf")
    with c3:
        min_odds = st.number_input("Min odds (A-grade)", value=2.0, min_value=1.0, step=0.5, key="dw_min_odds")
        max_bets = st.number_input("Max bets", value=10, min_value=5, max_value=15, step=1, key="dw_max_bets")
        min_overlay = st.number_input("Min overlay", value=1.10, min_value=1.0, max_value=2.0,
                                       step=0.05, key="dw_min_overlay",
                                       help="Minimum overlay ratio (ML odds / fair odds). 1.10 = require 10% value edge.")

    paper_mode = st.checkbox("Paper mode", value=True, key="dw_paper")

    # --- Generate ---
    if st.button("Generate Daily WIN Bets", type="primary", key="btn_daily_wins"):
        if not dw_date_str:
            st.warning("Select a race date.")
        else:
            with st.spinner("Building daily WIN bet plan across all tracks..."):
                payload = {
                    "race_date": dw_date_str,
                    "bankroll": bankroll,
                    "risk_profile": risk_profile,
                    "max_risk_per_day_pct": max_day_pct,
                    "min_confidence": min_confidence,
                    "min_odds_a": min_odds,
                    "paper_mode": paper_mode,
                    "max_bets": max_bets,
                    "min_overlay": min_overlay,
                    "save": True,
                }
                try:
                    resp = api_post(f"/bets/daily-wins", json=payload, timeout=30)
                    if resp.status_code == 200:
                        st.session_state["dw_last_result"] = resp.json()
                        st.session_state["dw_last_date"] = dw_date_str
                        plan_id = resp.json().get("plan_id", "?")
                        st.success(f"Plan generated (plan_id={plan_id})")
                        # Also fetch exotics
                        try:
                            ex_resp = api_post("/bets/daily-exotics", json=payload, timeout=30)
                            if ex_resp.status_code == 200:
                                st.session_state["dw_exotics"] = ex_resp.json()
                            else:
                                st.session_state["dw_exotics"] = None
                        except Exception:
                            st.session_state["dw_exotics"] = None
                    elif resp.status_code == 404:
                        st.warning("No predictions found for this date. Run the engine on sessions first.")
                    else:
                        st.error(f"Error: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Is the backend running?")
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- Display results ---
    result = st.session_state.get("dw_last_result")
    if result:
        candidates = result.get("candidates", [])
        plan_id = result.get("plan_id")

        st.divider()
        st.subheader(f"Top {len(candidates)} WIN Bets")

        # Summary metrics
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.metric("Total Risk", f"${result.get('total_risk', 0):.0f}")
        with w2:
            st.metric("Bets", len(candidates))
        with w3:
            tracks = result.get("tracks", [])
            st.metric("Tracks", len(tracks))
        with w4:
            st.metric("Plan ID", plan_id or "N/A")

        if not candidates:
            st.info("No qualifying WIN bets found. Try relaxing min confidence or min odds.")
        else:
            # Candidates table
            rows = []
            for c in candidates:
                rows.append({
                    "Track": c.get("track", ""),
                    "Race": c.get("race_number", ""),
                    "Horse": c.get("horse_name", ""),
                    "Cycle": c.get("projection_type", ""),
                    "Conf": f"{c.get('confidence', 0):.0%}",
                    "WinProb": f"{c.get('win_prob', 0):.1%}",
                    "Odds": f"{c['odds_decimal']:.1f}-1" if c.get("odds_decimal") else "-",
                    "FairOdds": f"{c['fair_odds']:.1f}-1" if c.get("fair_odds") else "-",
                    "Overlay": f"{c['overlay']:.2f}x" if c.get("overlay") else "-",
                    "Edge": f"{c.get('edge', 0):.1%}",
                    "Stake": f"${c.get('stake', 0):.0f}",
                    "Score": f"{c.get('best_bet_score', 0):.1f}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # --- Export ---
            st.divider()
            st.subheader("Export")
            ec1, ec2 = st.columns(2)

            csv_header = "track,race,horse,cycle,confidence,win_prob,odds,fair_odds,overlay,edge,stake,score"
            csv_lines = [csv_header]
            for c in candidates:
                fair_v = f"{c['fair_odds']:.2f}" if c.get("fair_odds") else ""
                over_v = f"{c['overlay']:.3f}" if c.get("overlay") else ""
                csv_lines.append(
                    f"{c.get('track','')},{c.get('race_number','')},\"{c.get('horse_name','')}\","
                    f"{c.get('projection_type','')},{c.get('confidence', 0):.2f},"
                    f"{c.get('win_prob', 0):.3f},"
                    f"{c.get('odds_decimal', '')},{fair_v},{over_v},"
                    f"{c.get('edge', 0):.4f},"
                    f"{c.get('stake', 0):.2f},{c.get('best_bet_score', 0):.1f}"
                )
            with ec1:
                st.download_button("Download CSV", "\n".join(csv_lines),
                                   "daily_wins.csv", "text/csv", key="dw_dl_csv")

            text_lines = [f"=== DAILY WIN BETS ({st.session_state.get('dw_last_date', '')}) ==="]
            for c in candidates:
                odds_str = f"{c['odds_decimal']:.1f}-1" if c.get("odds_decimal") else "?"
                fair_str = f"(fair: {c['fair_odds']:.1f}-1)" if c.get("fair_odds") else ""
                overlay_str = f"[{c['overlay']:.2f}x]" if c.get("overlay") else ""
                text_lines.append(
                    f"{c.get('track','')} R{c.get('race_number','')}: "
                    f"{c.get('horse_name','')} @ {odds_str} {fair_str} {overlay_str} — ${c.get('stake', 0):.0f}"
                )
            text_lines.append(f"\nTotal risk: ${result.get('total_risk', 0):.0f}")
            with ec2:
                st.download_button("Download Text", "\n".join(text_lines),
                                   "daily_wins.txt", "text/plain", key="dw_dl_txt")

        # --- Exotic Recommendations ---
        exotics = st.session_state.get("dw_exotics")
        if exotics:
            _render_exotic_table(
                "Best Daily Doubles (Top 5)",
                exotics.get("daily_doubles", []),
                multi_race=True,
            )
            _render_exotic_table(
                "Best Exactas (Top 5)",
                exotics.get("exactas", []),
            )
            _render_exotic_table(
                "Best Trifectas (Top 5)",
                exotics.get("trifectas", []),
            )

        # --- Past plans ---
        st.divider()
        st.subheader("Past Daily Plans")
        try:
            plans_resp = requests.get(
                f"{API_BASE_URL}/bets/plans", params={"date": dw_date_str}, timeout=10
            )
            if plans_resp.status_code == 200:
                all_plans = plans_resp.json()
                daily_plans = [p for p in all_plans if str(p.get("session_id", "")).startswith("daily_")]
                if daily_plans:
                    for dp in daily_plans:
                        dp_id = dp.get("plan_id")
                        dp_risk = dp.get("total_risk", 0)
                        dp_mode = "Paper" if dp.get("paper_mode") else "Live"
                        dp_created = dp.get("created_at", "")[:19]
                        st.markdown(
                            f"**Plan #{dp_id}** — Risk: ${dp_risk:.0f} | "
                            f"Mode: {dp_mode} | Created: {dp_created}"
                        )
                        if st.button(f"Evaluate ROI (Plan #{dp_id})", key=f"dw_eval_{dp_id}"):
                            try:
                                eval_resp = requests.get(
                                    f"{API_BASE_URL}/bets/evaluate/{dp_id}", timeout=15
                                )
                                if eval_resp.status_code == 200:
                                    ev = eval_resp.json()
                                    rc1, rc2, rc3 = st.columns(3)
                                    with rc1:
                                        st.metric("Wagered", f"${ev.get('total_wagered', 0):.0f}")
                                    with rc2:
                                        st.metric("Returned", f"${ev.get('total_returned', 0):.0f}")
                                    with rc3:
                                        roi = ev.get("roi_pct", 0)
                                        st.metric("ROI", f"{roi:+.1f}%")
                                else:
                                    st.warning("No results linked yet. Upload results first.")
                            except Exception:
                                st.error("Could not evaluate plan.")
                else:
                    st.info("No daily plans found for this date.")
        except Exception:
            pass


def dual_mode_page():
    st.header("Dual Mode Betting")
    st.caption("Profit Mode (WIN + DD) and Score Mode (Pick3 + Pick6) under separate budgets.")

    # --- Session selector ---
    try:
        sr = api_get("/sessions", timeout=10)
        races = sr.json().get("sessions", []) if sr.ok else []
    except Exception:
        races = []

    if not races:
        st.info("No sessions found. Upload and run the engine first.")
        return

    def _dm_label(x):
        name = x.get('primary_pdf_filename') or x.get('original_filename', 'unknown')
        trk = x.get('track') or x.get('track_name', '')
        dt = x.get('date') or x.get('race_date', '')
        return f"{name} | {trk} | {dt}"

    dm_sel = st.selectbox("Session:", options=races, format_func=_dm_label, key="dm_session")
    if not dm_sel:
        return

    dm_sid = dm_sel.get('session_id') or dm_sel.get('id')
    dm_track = dm_sel.get('track') or dm_sel.get('track_name', '')
    dm_date = dm_sel.get('date') or dm_sel.get('race_date', '')

    # --- Mode toggle ---
    mode = st.radio("Mode", ["Profit", "Score", "Both"], horizontal=True, index=2, key="dm_mode")
    mode_val = mode.lower()

    # --- Settings ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Shared**")
        bankroll = st.number_input("Bankroll ($)", min_value=50, max_value=50000,
                                   value=1000, step=50, key="dm_bankroll")
        score_pct = st.slider("Score budget %", 5, 50, 20, key="dm_score_pct") / 100.0
        risk_profile = st.selectbox("Risk profile",
                                    ["conservative", "standard", "aggressive"],
                                    index=1, key="dm_risk_profile")
    with c2:
        st.markdown("**Profit Mode**")
        profit_overlay = st.number_input("Min overlay", min_value=1.0, max_value=3.0,
                                         value=1.25, step=0.05, key="dm_profit_overlay")
        profit_odds_a = st.number_input("Min odds (A)", min_value=1.0, max_value=20.0,
                                         value=2.0, step=0.5, key="dm_profit_odds_a")
        profit_odds_b = st.number_input("Min odds (B)", min_value=1.0, max_value=20.0,
                                         value=4.0, step=0.5, key="dm_profit_odds_b")
    with c3:
        st.markdown("**Score Mode**")
        score_odds = st.number_input("Min odds", min_value=3.0, max_value=30.0,
                                      value=8.0, step=1.0, key="dm_score_odds")
        score_overlay = st.number_input("Min overlay", min_value=1.0, max_value=4.0,
                                         value=1.60, step=0.1, key="dm_score_overlay")
        mandatory = st.checkbox("Mandatory payout / big carryover day", key="dm_mandatory")

    # Budget preview
    profit_bud = bankroll * (1.0 - score_pct)
    score_bud = bankroll * score_pct
    bc1, bc2 = st.columns(2)
    bc1.caption(f"Profit budget: ${profit_bud:.0f}")
    bc2.caption(f"Score budget: ${score_bud:.0f}")

    # --- Engine output check ---
    engine_store = st.session_state.get("engine_outputs_by_session", {})
    dm_engine_entry = engine_store.get(dm_sid)
    if not dm_engine_entry or not dm_engine_entry.get("races_run"):
        st.warning("No engine outputs for this session. Run the engine first or click below.")
        if st.button("Run Engine Now", key="dm_run_engine"):
            with st.spinner("Running engine on full card..."):
                ok = _run_engine_full_card(dm_sid, dm_sel)
                if ok:
                    st.success("Engine complete — ready to generate plan.")
                    st.rerun()
                else:
                    st.error("No horse data found for this session.")
            return

    # --- Generate ---
    if st.button("Generate Plan", type="primary", key="dm_generate"):
        payload = {
            "session_id": dm_sid,
            "track": dm_track,
            "race_date": dm_date,
            "mode": mode_val,
            "bankroll": bankroll,
            "risk_profile": risk_profile,
            "score_budget_pct": score_pct,
            "profit_min_overlay": profit_overlay,
            "profit_min_odds_a": profit_odds_a,
            "profit_min_odds_b": profit_odds_b,
            "score_min_odds": score_odds,
            "score_min_overlay": score_overlay,
            "mandatory_payout": mandatory,
            "save": True,
        }
        try:
            resp = api_post("/bets/dual-mode", json=payload, timeout=30)
            if resp.ok:
                st.session_state["dm_result"] = resp.json()
            elif resp.status_code == 404:
                st.warning("No predictions found. Run the engine on this session first.")
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    # --- Display ---
    result = st.session_state.get("dm_result")
    if not result:
        return

    plan = result.get("plan", {})
    if result.get("plan_id"):
        st.caption(f"Plan saved (ID {result['plan_id']})")

    # Budget summary
    m1, m2, m3 = st.columns(3)
    profit_data = plan.get("profit")
    score_data = plan.get("score")
    p_risk = profit_data["total_risk"] if profit_data else 0
    s_risk = score_data["total_risk"] if score_data else 0
    p_bud = plan.get("profit_budget", 0)
    s_bud = plan.get("score_budget", 0)

    m1.metric("Profit", f"${p_risk:.0f} / ${p_bud:.0f}")
    m2.metric("Score", f"${s_risk:.0f} / ${s_bud:.0f}")
    m3.metric("Total", f"${plan.get('total_risk', 0):.0f} / ${plan.get('settings', {}).get('bankroll', 0):.0f}")

    # Progress bars
    pb1, pb2 = st.columns(2)
    with pb1:
        st.progress(min(p_risk / p_bud, 1.0) if p_bud > 0 else 0.0, text="Profit budget used")
    with pb2:
        st.progress(min(s_risk / s_bud, 1.0) if s_bud > 0 else 0.0, text="Score budget used")

    # Top warnings
    for w in plan.get("warnings", []):
        st.warning(w)

    # --- Profit Mode ---
    if profit_data:
        with st.expander("Profit Mode — WIN + Daily Double", expanded=(mode_val in ("profit", "both"))):
            wins = profit_data.get("win_bets", [])
            if wins:
                st.subheader(f"WIN Bets ({len(wins)})")
                rows = []
                for w in wins:
                    rows.append({
                        "Race": w.get("race", ""),
                        "Horse": w.get("horse", ""),
                        "Grade": w.get("grade", ""),
                        "Odds": w.get("odds", ""),
                        "Model%": f"{w.get('model_prob', 0):.1%}",
                        "Overlay": f"{w.get('overlay', 0):.2f}x",
                        "Stake": f"${w.get('stake', 0):.0f}",
                        "Cycle": w.get("projection_type", ""),
                        "Single": "YES" if w.get("true_single") else "",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No qualifying WIN bets.")

            dds = profit_data.get("dd_plans", [])
            if dds:
                st.subheader(f"Daily Doubles ({len(dds)})")
                for dd in dds:
                    r1 = dd.get("start_race", 0)
                    r2 = r1 + 1
                    st.markdown(f"**R{r1}–R{r2}** | {dd.get('leg1_grade', '?')}/{dd.get('leg2_grade', '?')} | ${dd.get('total_cost', 0):.0f}")
                    tickets = dd.get("tickets", [])
                    if tickets:
                        t_rows = []
                        for i, t in enumerate(tickets, 1):
                            t_rows.append({
                                "#": i,
                                f"Leg 1 (R{r1})": ", ".join(t.get("leg1", [])),
                                f"Leg 2 (R{r2})": ", ".join(t.get("leg2", [])),
                                "Cost": f"${t.get('cost', 0):.0f}",
                                "Type": t.get("reason", ""),
                            })
                        st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

            # Passed races
            passed = profit_data.get("passed_races", [])
            if passed:
                st.subheader("Passed Races")
                for pr in passed:
                    st.caption(f"R{pr['race']}: PASS — {pr['reason']}")

            for w in profit_data.get("warnings", []):
                st.warning(w)

    # --- Score Mode ---
    if score_data:
        with st.expander("Score Mode — Pick3 + Pick6", expanded=(mode_val in ("score", "both"))):
            if score_data.get("budget_exhausted"):
                st.error("Score budget exhausted — hard stop")

            p3s = score_data.get("pick3_plans", [])
            if p3s:
                st.subheader(f"Pick3 Plans ({len(p3s)})")
                for p3 in p3s:
                    legs = p3.get("legs", [])
                    st.markdown(f"**R{p3.get('start_race', '?')}–R{p3.get('start_race', 0) + 2}** | ${p3.get('cost', 0):.0f}")
                    if legs:
                        lr = []
                        for j, lg in enumerate(legs, 1):
                            lr.append({
                                "Leg": j,
                                "Race": lg.get("race_number", ""),
                                "Grade": lg.get("grade", ""),
                                "Horses": ", ".join(lg.get("horses", [])),
                                "Count": lg.get("horse_count", 0),
                            })
                        st.dataframe(pd.DataFrame(lr), use_container_width=True, hide_index=True)
            else:
                st.info("No qualifying Pick3 windows.")

            p6 = score_data.get("pick6_plan")
            if p6:
                st.subheader("Pick6 Plan")
                st.markdown(f"**R{p6.get('start_race', '?')}–R{p6.get('start_race', 0) + 5}** | ${p6.get('cost', 0):.0f}")
                legs = p6.get("legs", [])
                if legs:
                    lr = []
                    for j, lg in enumerate(legs, 1):
                        lr.append({
                            "Leg": j,
                            "Race": lg.get("race_number", ""),
                            "Grade": lg.get("grade", ""),
                            "Horses": ", ".join(lg.get("horses", [])),
                            "Count": lg.get("horse_count", 0),
                        })
                    st.dataframe(pd.DataFrame(lr), use_container_width=True, hide_index=True)
            elif settings_val := plan.get("settings", {}):
                if settings_val.get("mandatory_payout"):
                    st.info("No qualifying Pick6 sequence found.")
                else:
                    st.info("Pick6 requires mandatory payout flag.")

            # Passed sequences
            passed_seq = score_data.get("passed_sequences", [])
            if passed_seq:
                st.subheader("Passed Sequences")
                for ps in passed_seq:
                    races_str = ", ".join(f"R{r}" for r in ps.get("races", []))
                    st.caption(f"{races_str}: PASS — {ps['reason']}")

            for w in score_data.get("warnings", []):
                st.warning(w)

    # --- Export ---
    from bet_builder import dual_mode_plan_to_text, DualModeDayPlan, ProfitModePlan, ScoreModePlan
    # Reconstruct lightweight plan for text export
    plan_obj = DualModeDayPlan(
        mode=plan.get("mode", "both"),
        profit=ProfitModePlan(**{k: v for k, v in (profit_data or {}).items()
                                 if k in ProfitModePlan.__dataclass_fields__}) if profit_data else None,
        score=ScoreModePlan(**{k: v for k, v in (score_data or {}).items()
                               if k in ScoreModePlan.__dataclass_fields__}) if score_data else None,
        total_risk=plan.get("total_risk", 0),
        profit_budget=plan.get("profit_budget", 0),
        score_budget=plan.get("score_budget", 0),
        settings=plan.get("settings", {}),
        warnings=plan.get("warnings", []),
    )
    st.download_button("Export Text", dual_mode_plan_to_text(plan_obj),
                       "dual_mode_plan.txt", "text/plain", key="dm_export_txt")


def bet_builder_page():
    st.header("Bet Builder")
    st.caption("Generate WIN + EXACTA tickets from engine predictions with Kelly sizing and bankroll guardrails.")

    # --- Session selector ---
    session_options = {"(none)": ""}
    try:
        sr = api_get(f"/sessions", timeout=10)
        if sr.status_code == 200:
            for s in sr.json().get("sessions", []):
                label = f"{s['track']} {s['date']} ({s['session_id'][:8]}...)"
                session_options[label] = s["session_id"]
    except Exception:
        pass

    sel = st.selectbox("Session", options=list(session_options.keys()), key="bb_session")
    chosen_sid = session_options.get(sel, "")

    col_t, col_d = st.columns(2)
    with col_t:
        bb_track = st.text_input("Track", key="bb_track")
    with col_d:
        bb_date = st.text_input("Date (MM/DD/YYYY)", key="bb_date")

    st.subheader("Settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        bankroll = st.number_input("Bankroll ($)", value=1000, min_value=100, step=100, key="bb_bankroll")
        risk_profile = st.selectbox("Risk Profile", ["conservative", "standard", "aggressive"],
                                    index=1, key="bb_risk_profile")
    with c2:
        max_race_pct = st.number_input("Max risk/race (%)", value=1.5, min_value=0.5, max_value=5.0,
                                       step=0.5, key="bb_max_race")
        max_day_pct = st.number_input("Max risk/day (%)", value=6.0, min_value=1.0, max_value=20.0,
                                      step=1.0, key="bb_max_day")
    with c3:
        min_odds_a = st.number_input("Min odds (A races)", value=2.0, min_value=1.0, step=0.5, key="bb_min_odds_a")
        min_odds_b = st.number_input("Min odds (B races)", value=4.0, min_value=1.0, step=0.5, key="bb_min_odds_b")

    paper_mode = st.checkbox("Paper mode", value=True, key="bb_paper",
                             help="Mark plan as paper (no real money). Always on by default.")
    if not paper_mode:
        st.warning("LIVE mode: plans generated will be marked as live bets. "
                   "Ensure bankroll and caps are set correctly before proceeding.")

    if st.button("Generate Bet Plan", type="primary", key="btn_build_bets"):
        if not chosen_sid or not bb_track or not bb_date:
            st.warning("Select a session and enter track + date.")
        else:
            with st.spinner("Building bet plan..."):
                try:
                    payload = {
                        "session_id": chosen_sid,
                        "track": bb_track,
                        "race_date": bb_date,
                        "bankroll": bankroll,
                        "risk_profile": risk_profile,
                        "max_risk_per_race_pct": max_race_pct,
                        "max_risk_per_day_pct": max_day_pct,
                        "min_odds_a": min_odds_a,
                        "min_odds_b": min_odds_b,
                        "paper_mode": paper_mode,
                        "save": True,
                    }
                    resp = api_post(f"/bets/build", json=payload, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["bb_last_plan"] = data
                        st.success(f"Plan generated (plan_id={data.get('plan_id', '?')})")
                    elif resp.status_code == 404:
                        st.warning("No predictions found for this card. Run the engine first.")
                    else:
                        st.error(f"Error: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- Display plan ---
    plan_data = st.session_state.get("bb_last_plan", {}).get("plan")
    plan_id = st.session_state.get("bb_last_plan", {}).get("plan_id")
    if plan_data:
        st.divider()
        st.subheader("Day Plan")

        settings = plan_data.get("settings", {})
        mode_label = "PAPER" if settings.get("paper_mode", True) else "LIVE"

        # Live-mode guardrail: block if any race has poor figure quality
        if not settings.get("paper_mode", True):
            poor_races = [
                rp for rp in plan_data.get("race_plans", [])
                if rp.get("figure_quality_pct") is not None
                and rp["figure_quality_pct"] < 0.80
                and not rp.get("passed", False)
            ]
            if poor_races:
                st.error(
                    f"LIVE MODE BLOCKED: {len(poor_races)} race(s) have poor data quality "
                    f"(< 80% figures). Switch to Paper mode or improve data before placing live bets."
                )

        w1, w2, w3 = st.columns(3)
        with w1:
            st.metric("Total Risk", f"${plan_data['total_risk']:.0f}")
        with w2:
            bet_count = sum(len(rp.get("tickets", [])) for rp in plan_data.get("race_plans", []))
            st.metric("Bets", bet_count)
        with w3:
            st.metric("Mode", mode_label)

        for w in plan_data.get("warnings", []):
            st.warning(w)

        # Per-race details
        for rp in plan_data.get("race_plans", []):
            grade = rp["grade"]
            race_num = rp["race_number"]
            if rp["passed"]:
                st.markdown(f"**Race {race_num}** — Grade {grade} PASS")
                with st.expander(f"Race {race_num} details"):
                    st.caption(rp.get("rationale", ""))
            else:
                st.markdown(f"**Race {race_num}** — Grade **{grade}** | ${rp['total_cost']:.0f}")
                tickets_rows = []
                for t in rp.get("tickets", []):
                    tickets_rows.append({
                        "Type": t["bet_type"],
                        "Selections": " / ".join(t["selections"]),
                        "Cost": f"${t['cost']:.0f}",
                        "Rationale": t["rationale"],
                    })
                if tickets_rows:
                    st.dataframe(pd.DataFrame(tickets_rows), hide_index=True, use_container_width=True)
                with st.expander(f"Race {race_num} grading"):
                    for r in rp.get("grade_reasons", []):
                        st.text(f"  - {r}")

        # --- Exports ---
        st.divider()
        st.subheader("Export")
        ec1, ec2 = st.columns(2)

        # CSV export
        csv_rows = ["race,grade,bet_type,selections,cost,rationale"]
        for rp in plan_data.get("race_plans", []):
            if rp["passed"]:
                csv_rows.append(f"{rp['race_number']},{rp['grade']},PASS,,0,\"{rp.get('rationale','')}\"")
            else:
                for t in rp.get("tickets", []):
                    sels = " / ".join(t["selections"])
                    csv_rows.append(
                        f"{rp['race_number']},{rp['grade']},{t['bet_type']},"
                        f"\"{sels}\",{t['cost']:.2f},\"{t['rationale']}\""
                    )
        with ec1:
            st.download_button(
                "Download CSV", "\n".join(csv_rows), "bet_plan.csv", "text/csv",
                key="bb_dl_csv",
            )

        # Text export
        lines = [f"=== BET PLAN ({mode_label}) ===",
                 f"Bankroll: ${settings.get('bankroll',0):.0f}",
                 f"Risk profile: {settings.get('risk_profile','standard')}",
                 f"Total risk: ${plan_data['total_risk']:.0f}", ""]
        for rp in plan_data.get("race_plans", []):
            if rp["passed"]:
                lines.append(f"Race {rp['race_number']}: PASS (Grade {rp['grade']})")
            else:
                lines.append(f"Race {rp['race_number']}: Grade {rp['grade']} — ${rp['total_cost']:.0f}")
                for t in rp.get("tickets", []):
                    lines.append(f"  {t['rationale']}")
            lines.append("")
        with ec2:
            st.download_button(
                "Download Text", "\n".join(lines), "bet_plan.txt", "text/plain",
                key="bb_dl_txt",
            )

    # --- Saved plans & evaluation ---
    st.divider()
    st.subheader("Saved Plans")
    try:
        params = {}
        if chosen_sid:
            params["session_id"] = chosen_sid
        if bb_track:
            params["track"] = bb_track
        if bb_date:
            params["date"] = bb_date
        plans_resp = api_get(f"/bets/plans", params=params, timeout=10)
        plans = plans_resp.json() if plans_resp.status_code == 200 else []
    except Exception:
        plans = []

    if plans:
        plan_rows = []
        for p in plans:
            plan_rows.append({
                "ID": p["plan_id"],
                "Track": p["track"],
                "Date": p["race_date"],
                "Risk": f"${p['total_risk']:.0f}",
                "Mode": "Paper" if p["paper_mode"] else "Live",
                "Created": p["created_at"][:16],
            })
        st.dataframe(pd.DataFrame(plan_rows), hide_index=True, use_container_width=True)

        eval_id = st.number_input("Plan ID to evaluate", min_value=1, step=1, key="bb_eval_id")
        if st.button("Evaluate vs Results", key="btn_eval_plan"):
            try:
                er = api_get(f"/bets/evaluate/{int(eval_id)}", timeout=15)
                if er.status_code == 200:
                    ev = er.json()

                    # Resolution stats
                    resolved = ev.get("resolved", 0)
                    total_tk = ev.get("total_tickets", 0)
                    unresolved = ev.get("unresolved", 0)
                    if total_tk > 0:
                        if unresolved == 0:
                            st.success(f"Results linked: {resolved}/{total_tk} tickets resolved")
                        else:
                            st.warning(f"Results linked: {resolved}/{total_tk} tickets resolved "
                                       f"({unresolved} unresolved)")

                    e1, e2, e3 = st.columns(3)
                    with e1:
                        st.metric("Wagered", f"${ev['total_wagered']:.2f}")
                    with e2:
                        st.metric("Returned", f"${ev['total_returned']:.2f}")
                    with e3:
                        st.metric("ROI", f"{ev['roi_pct']:.1f}%")

                    tr_rows = []
                    unresolved_rows = []
                    for tr in ev.get("ticket_results", []):
                        tr_rows.append({
                            "Race": tr["race_number"],
                            "Type": tr["bet_type"],
                            "Selections": " / ".join(tr["selections"]),
                            "Cost": f"${tr['cost']:.0f}",
                            "Outcome": tr["outcome"],
                            "Returned": f"${tr['returned']:.2f}" if tr["returned"] else "-",
                            "Match": tr.get("match_tier", ""),
                        })
                        if tr.get("outcome") == "no_result":
                            unresolved_rows.append({
                                "Race": tr["race_number"],
                                "Type": tr["bet_type"],
                                "Selections": " / ".join(tr["selections"]),
                                "Reason": tr.get("match_reason", "unknown"),
                            })
                    if tr_rows:
                        st.dataframe(pd.DataFrame(tr_rows), hide_index=True, use_container_width=True)
                    if unresolved_rows:
                        with st.expander(f"Unresolved tickets ({len(unresolved_rows)})"):
                            st.dataframe(pd.DataFrame(unresolved_rows), hide_index=True,
                                         use_container_width=True)
                else:
                    st.error(f"Error: {er.text}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("No saved bet plans. Generate one above.")


def statistics_page():
    st.header("📊 Statistics")
    
    try:
        response = api_get(f"/stats")
        
        if response.status_code == 200:
            stats = response.json()
            
            # Overall stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sessions", stats['total_sessions'])
            with col2:
                st.metric("Total Horses", stats['total_horses'])
            with col3:
                st.metric("Total Individual Races", stats['total_individual_races'])
            with col4:
                st.metric("Avg Races/Horse", f"{stats['average_races_per_horse']:.1f}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                if stats['surface_breakdown']:
                    surface_df = pd.DataFrame(list(stats['surface_breakdown'].items()), columns=['Surface', 'Count'])
                    fig = px.bar(
                        surface_df,
                        x='Surface',
                        y='Count',
                        title="Individual Races by Surface",
                        color='Surface'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if stats['track_breakdown']:
                    track_df = pd.DataFrame(list(stats['track_breakdown'].items()), columns=['Track', 'Count'])
                    fig = px.pie(
                        track_df,
                        values='Count',
                        names='Track',
                        title="Individual Races by Track"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Raw stats
            st.subheader("📋 Raw Statistics")
            st.json(stats)
            
        else:
            st.error(f"Error fetching statistics: {response.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

def api_status_page():
    st.header("🔧 API Status")
    
    try:
        # Health check
        response = api_get(f"/health")
        
        if response.status_code == 200:
            health = response.json()
            st.success("✅ API is healthy")
            st.json(health)
        else:
            st.error("❌ API is not responding")
        
        # Parser status
        st.subheader("🤖 Parser Status")
        try:
            parser_response = api_get(f"/parser-status")
            if parser_response.status_code == 200:
                parser_status = parser_response.json()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Traditional Parser", "✅" if parser_status["traditional_parser"] else "❌")
                with col2:
                    st.metric("GPT Parser", "✅" if parser_status["gpt_parser_available"] else "❌")
                with col3:
                    st.metric("OpenAI API Key", "✅" if parser_status["openai_api_key_set"] else "❌")
                with col4:
                    st.metric("Symbol Sheet", "✅" if parser_status["symbol_sheet_loaded"] else "❌")
                
                st.json(parser_status)
            else:
                st.error("❌ Could not fetch parser status")
        except Exception as e:
            st.error(f"❌ Error checking parser status: {str(e)}")
            
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {str(e)}")
        st.info("Make sure the API server is running on http://localhost:8000")

def _clear_stale_session_state(deleted_ids: list):
    """Remove session_state keys that reference any of the deleted race IDs."""
    engine_id = st.session_state.get('active_session_id')
    if engine_id and engine_id in deleted_ids:
        for key in ['active_session_id', 'last_projections',
                     'last_proj_race', 'best_bets']:
            st.session_state.pop(key, None)
    last_id = st.session_state.get('last_race_id')
    if last_id and last_id in deleted_ids:
        for key in ['last_race_id', 'parsed_horse_data', 'horses_df', 'races_df']:
            st.session_state.pop(key, None)
    new_sid = st.session_state.get('new_session_id')
    if new_sid and new_sid in deleted_ids:
        st.session_state.pop('new_session_id', None)


def manage_sheets_page():
    st.header("Manage Sheets")

    try:
        resp = api_get(f"/races", timeout=15)
        if resp.status_code != 200:
            st.error("Could not fetch sessions from API.")
            return
        races = resp.json().get("races", [])
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is the backend running on http://localhost:8000?")
        return
    except Exception as e:
        st.error(f"Error: {e}")
        return

    if not races:
        st.info("No parsed sessions found. Upload a PDF to get started.")
        return

    st.markdown(f"**{len(races)}** parsed session(s) available.")

    selected_ids = []
    for race in races:
        rid = race.get('session_id') or race.get('id')
        name = race.get('primary_pdf_filename') or race.get('original_filename', 'unknown')
        track = race.get('track') or race.get('track_name', 'N/A')
        date = race.get('date') or race.get('analysis_date', race.get('race_date', 'N/A'))
        # Show primary/secondary badges
        if race.get('has_secondary'):
            badge = "Primary+Secondary"
        elif race.get('has_primary'):
            badge = "Primary Only"
        else:
            badge = race.get('parser_used', 'unknown')
        label = f"{name} | {track} | {date} | {badge}"
        if st.checkbox(label, key=f"manage_chk_{rid}"):
            selected_ids.append(rid)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Delete Selected", disabled=len(selected_ids) == 0):
            st.session_state['pending_delete_ids'] = selected_ids
    with col2:
        if st.button("Delete All", disabled=len(races) == 0):
            st.session_state['pending_delete_ids'] = [r.get('session_id') or r.get('id') for r in races]

    ids_to_delete = st.session_state.get('pending_delete_ids', [])
    if ids_to_delete:
        st.warning(f"You are about to delete {len(ids_to_delete)} session(s). This cannot be undone.")
        confirm = st.text_input("Type DELETE to confirm:", key="manage_confirm_input")
        if confirm == "DELETE":
            progress = st.progress(0)
            errors = []
            for i, rid in enumerate(ids_to_delete):
                try:
                    del_resp = api_delete(f"/races/{rid}", timeout=15)
                    if del_resp.status_code != 200:
                        errors.append(f"{rid}: {del_resp.text}")
                except Exception as e:
                    errors.append(f"{rid}: {str(e)}")
                progress.progress((i + 1) / len(ids_to_delete))
            progress.empty()
            if errors:
                st.error(f"Errors during deletion: {'; '.join(errors)}")
            else:
                st.success(f"Deleted {len(ids_to_delete)} session(s).")
            _clear_stale_session_state(ids_to_delete)
            st.session_state.pop('pending_delete_ids', None)
            st.rerun()
        elif confirm:
            st.error("Confirmation text does not match. Type exactly: DELETE")
        if st.button("Cancel"):
            st.session_state.pop('pending_delete_ids', None)
            st.rerun()


def horse_past_performance_page():
    st.header("🐎 Horse Past Performance Viewer")
    
    # Check if we have parsed data
    if 'parsed_horse_data' not in st.session_state:
        st.info("Please upload and parse a Ragozin sheet first to view horse past performance data.")
        return
    
    horse_data = st.session_state.parsed_horse_data
    
    # Display horse metadata
    st.subheader("🏇 Horse Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Horse Name", horse_data.get('horse_name', 'Unknown'))
        st.metric("Sex", horse_data.get('sex', 'Unknown'))
        st.metric("Age", horse_data.get('age', 0))
    
    with col2:
        st.metric("Sire", horse_data.get('sire', 'Unknown'))
        st.metric("Dam", horse_data.get('dam', 'Unknown'))
        st.metric("State Bred", horse_data.get('state_bred', 'Unknown'))
    
    with col3:
        st.metric("Total Races", horse_data.get('races', 0))
        st.metric("Top Figure", horse_data.get('top_fig', 'Unknown'))
        st.metric("Foaling Year", horse_data.get('foaling_year', 0))
    
    with col4:
        st.metric("Track Code", horse_data.get('track_code', 'Unknown'))
        st.metric("Sheet Page", horse_data.get('sheet_page_number', 'Unknown'))
        st.metric("Race Number", horse_data.get('race_number', 0))
    
    # Display AI Analysis
    st.subheader("🤖 AI Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Overall Performance Analysis:**")
        st.info(horse_data.get('horse_analysis', 'No analysis available'))
    
    with col2:
        st.write("**Performance Trend Analysis:**")
        st.info(horse_data.get('performance_trend', 'No trend analysis available'))
    
    # Display race history
    st.subheader("📋 Race History")
    lines = horse_data.get('lines', [])
    
    if lines:
        # Create race history table
        race_data = []
        for i, line in enumerate(lines):
            race_data.append({
                'Race #': i + 1,
                'Date': line.get('race_date', ''),
                'Track': line.get('track', ''),
                'Surface': line.get('surface', ''),
                'Figure': line.get('fig', ''),
                'Flags': ', '.join(line.get('flags', [])) if line.get('flags') else '',
                'Race Type': line.get('race_type', ''),
                'Month': line.get('month', ''),
                'Notes': line.get('notes', ''),
                'Analysis': line.get('race_analysis', '')[:100] + "..." if len(line.get('race_analysis', '')) > 100 else line.get('race_analysis', '')
            })
        
        df = pd.DataFrame(race_data)
        st.dataframe(df, use_container_width=True)
        
        # Detailed race analysis
        st.subheader("🔍 Detailed Race Analysis")
        for i, line in enumerate(lines):
            with st.expander(f"Race {i + 1}: {line.get('race_date', '')} - {line.get('track', '')} - {line.get('race_type', '')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Figure:** {line.get('fig', '')}")
                    if line.get('flags'):
                        st.write(f"**Flags:** {', '.join(line.get('flags', []))}")
                    st.write(f"**Surface:** {line.get('surface', '')}")
                    st.write(f"**Month:** {line.get('month', '')}")
                    if line.get('notes'):
                        st.write(f"**Notes:** {line.get('notes', '')}")
                
                with col2:
                    st.write("**AI Analysis:**")
                    st.info(line.get('race_analysis', 'No analysis available'))
                    
                    # Enhanced fields if available
                    if line.get('ai_analysis'):
                        ai_analysis = line.get('ai_analysis', {})
                        if isinstance(ai_analysis, dict):
                            st.write("**Detailed AI Analysis:**")
                            if ai_analysis.get('left_side'):
                                st.write(f"**Left Side:** {ai_analysis.get('left_side')}")
                            if ai_analysis.get('middle'):
                                st.write(f"**Middle:** {ai_analysis.get('middle')}")
                            if ai_analysis.get('right_side'):
                                st.write(f"**Right Side:** {ai_analysis.get('right_side')}")
                            if ai_analysis.get('full_interpretation'):
                                st.write(f"**Full Interpretation:** {ai_analysis.get('full_interpretation')}")
    else:
        st.warning("No race history data available.")

if __name__ == "__main__":
    main() 