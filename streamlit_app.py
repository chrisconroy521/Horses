import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import re
from pathlib import Path
import io
import base64
from datetime import datetime

from handicap_engine import HandicappingEngine, HorseInput, FigureEntry, BiasInput

# Configure page
st.set_page_config(
    page_title="Ragozin Sheets Parser - Enhanced",
    page_icon="🐎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

def main():
    st.title("🐎 Ragozin Sheets Parser - Enhanced Analysis")
    st.markdown("Upload and analyze horse racing performance sheets with AI-powered insights and symbol analysis")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Upload PDF", "Engine", "Horse Past Performance", "Horses Overview", "Individual Horse Analysis", "Race Analysis", "Statistics", "API Status"]
    )

    if page == "Upload PDF":
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
    elif page == "Statistics":
        statistics_page()
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
    """Convert a parsed horse dict to a HorseInput for the engine."""
    figures = []
    for line in horse.get('lines', []):
        val = _extract_figure(line)
        surface = line.get('surface', line.get('surface_type', ''))
        flags = line.get('flags', []) + line.get('post_symbols', [])
        figures.append(FigureEntry(value=val, surface=surface, flags=flags))
    return HorseInput(
        name=horse.get('horse_name', 'Unknown'),
        post=post,
        style="P",
        figures=figures,
    )


def engine_page():
    st.header("🏁 Handicapping Engine")

    engine = HandicappingEngine()

    # --- Load last parsed session button ---
    if st.button("Load last parsed session"):
        try:
            resp = requests.get(f"{API_BASE_URL}/races", timeout=15)
            if resp.status_code == 200:
                races = resp.json().get("races", [])
                if races:
                    latest = races[-1]
                    st.session_state['engine_race_id'] = latest['id']
                    st.session_state['engine_race_meta'] = latest
                else:
                    st.warning("No parsed sessions found. Upload a PDF first.")
            else:
                st.error(f"API error: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Is the backend running on http://localhost:8000?")
        except Exception as e:
            st.error(f"Error: {e}")

    # --- Session selector (same pattern as Horses Overview) ---
    try:
        resp = requests.get(f"{API_BASE_URL}/races", timeout=15)
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

    # Default to last loaded or most recent
    default_idx = 0
    if 'engine_race_id' in st.session_state:
        for i, r in enumerate(races):
            if r['id'] == st.session_state['engine_race_id']:
                default_idx = i
                break

    selected = st.selectbox(
        "Session:",
        options=races,
        format_func=lambda x: f"{x['track_name']} - {x['horses_count']} horses ({x.get('analysis_date', x['race_date'])})",
        index=default_idx,
    )

    if not selected:
        return

    # Fetch horse data for the session
    try:
        h_resp = requests.get(f"{API_BASE_URL}/races/{selected['id']}/horses", timeout=30)
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

    # --- Status panel ---
    source = selected.get('parser_used', 'unknown')
    total_lines = sum(len(h.get('lines', [])) for h in all_horses)
    st.caption(
        f"Loaded races: {total_lines}, horses: {len(all_horses)} (source: {source})"
    )

    # Group horses by race_number (fall back to a single group)
    race_groups: dict = {}
    for h in all_horses:
        rn = h.get('race_number', 0) or 0
        race_groups.setdefault(rn, []).append(h)

    # List races with counts
    race_summary_parts = []
    for rn in sorted(race_groups.keys()):
        count = len(race_groups[rn])
        label = f"R{rn}" if rn else "Ungrouped"
        race_summary_parts.append(f"{label}: {count} horses")
    st.caption(" | ".join(race_summary_parts))

    # --- Scratch selector ---
    all_names = [h.get('horse_name', 'Unknown') for h in all_horses]
    scratches = st.multiselect("Scratches:", options=all_names, default=[])

    # Filter out scratched horses
    active_horses = [h for h in all_horses if h.get('horse_name', '') not in scratches]

    # Rebuild race groups after scratches
    active_groups: dict = {}
    for h in active_horses:
        rn = h.get('race_number', 0) or 0
        active_groups.setdefault(rn, []).append(h)

    # Only show races with horses remaining
    populated_races = {rn: horses for rn, horses in active_groups.items() if len(horses) > 0}

    if not populated_races:
        st.warning("All races are empty after scratches. Adjust your scratch list.")
        return

    # Race selector — only populated races
    race_options = sorted(populated_races.keys())
    race_labels = {rn: f"Race {rn} ({len(populated_races[rn])} horses)" if rn else f"All ({len(populated_races[rn])} horses)" for rn in race_options}

    selected_race = st.selectbox(
        "Race:",
        options=race_options,
        format_func=lambda rn: race_labels[rn],
    )

    race_horses = populated_races[selected_race]

    # --- Bias controls ---
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
        e_pct=float(e_pct),
        ep_pct=float(ep_pct),
        p_pct=float(p_pct),
        s_pct=float(s_pct),
        speed_favoring=speed_fav,
    )

    # --- Build HorseInput list ---
    horse_inputs = []
    for i, h in enumerate(race_horses):
        post = str(h.get('race_number', i + 1))
        hi = _horse_dict_to_input(h, post)
        horse_inputs.append(hi)

    # --- Run engine ---
    if st.button("Run Projections", type="primary"):
        projections = engine.analyze_race(horse_inputs, bias, scratches=scratches)

        if not projections:
            st.warning("No projections produced. All horses may lack usable figures.")
            return

        st.subheader("Projections")

        rows = []
        for p in projections:
            rows.append({
                'Horse': p.name,
                'Post': p.post,
                'Style': p.style,
                'Proj Low': f"{p.projected_low:.1f}",
                'Proj High': f"{p.projected_high:.1f}",
                'Confidence': f"{p.confidence:.0%}",
                'Tags': ', '.join(p.tags) if p.tags else '-',
                'Raw Score': f"{p.raw_score:.1f}",
                'Bias Score': f"{p.bias_score:.1f}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Confidence bar chart
        fig = px.bar(
            x=[p.name for p in projections],
            y=[p.confidence for p in projections],
            labels={'x': 'Horse', 'y': 'Confidence'},
            title="Projection Confidence",
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Projection range chart
        fig2 = go.Figure()
        for p in projections:
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
            yaxis_title="Ragozin Figure",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)


def upload_page():
    st.header("📄 Upload Ragozin Sheet")

    # Parser toggle — persisted across reruns
    if 'use_gpt_parser' not in st.session_state:
        st.session_state['use_gpt_parser'] = False
    st.checkbox(
        "Use GPT parser",
        key='use_gpt_parser',
        help="When enabled, uses GPT-4 Vision for parsing. Falls back to traditional parser on failure."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a Ragozin performance sheet PDF"
    )

    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")

        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col2:
            st.metric("File Type", uploaded_file.type)
        with col3:
            st.metric("File Name", uploaded_file.name)

        # Upload button
        if st.button("🚀 Parse PDF", type="primary"):
            use_gpt = st.session_state['use_gpt_parser']
            spinner_msg = "Parsing PDF with GPT..." if use_gpt else "Parsing PDF with traditional parser..."
            with st.spinner(spinner_msg):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    params = {"use_gpt": use_gpt}

                    response = requests.post(
                        f"{API_BASE_URL}/upload-pdf",
                        files=files,
                        params=params,
                        timeout=300
                    )

                    if response.status_code == 200:
                        result = response.json()

                        # --- Parser status display ---
                        parser_info = result.get("parser_info", {})
                        parse_source = parser_info.get("parser_used", "unknown")
                        gpt_attempted = parser_info.get("gpt_attempted", False)
                        fallback = parser_info.get("fallback_to_traditional", False)

                        if fallback:
                            st.warning("GPT returned 0/error -> using traditional")
                            parse_source_label = "fallback"
                        elif gpt_attempted and parse_source == "gpt":
                            st.success("Parsed with GPT parser")
                            parse_source_label = "gpt"
                        else:
                            st.success("Parsed with traditional parser")
                            parse_source_label = "traditional"

                        # Show gpt_error_text collapsed if present
                        gpt_error = parser_info.get("gpt_error_text")
                        if gpt_error:
                            with st.expander("GPT error details"):
                                st.code(gpt_error)

                        # --- Parsing results ---
                        st.subheader("📊 Parsing Results")

                        race_info = result["race_info"]
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Horses", race_info["horses_count"])
                        with col2:
                            st.metric("Total Races", race_info.get("total_races", "N/A"))
                        with col3:
                            st.metric("Tracks", race_info.get("tracks_count", "N/A"))
                        with col4:
                            st.metric("Parse Source", parse_source_label)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Analysis Date", result.get("analysis_date", "N/A"))
                        with col2:
                            st.metric("Analysis Time", result.get("analysis_time", "N/A"))
                        with col3:
                            st.metric("Processing Duration", result.get("processing_duration", "N/A"))

                        # Store race ID in session state
                        st.session_state.last_race_id = result["race_id"]

                        # Fetch and store the parsed horse data
                        try:
                            horse_response = requests.get(
                                f"{API_BASE_URL}/races/{result['race_id']}/horses",
                                timeout=30
                            )
                            if horse_response.status_code == 200:
                                horse_data = horse_response.json()
                                horses_list = horse_data.get("horses", [])
                                if horses_list:
                                    st.session_state.parsed_horse_data = horses_list[0]
                                    st.caption(
                                        f"Loaded races: {race_info.get('total_races', 'N/A')}, "
                                        f"horses: {race_info['horses_count']} "
                                        f"(source: {parse_source_label})"
                                    )
                                else:
                                    st.warning("No horse data found in the parsed results")
                            else:
                                st.warning("Could not fetch horse data")
                        except Exception as e:
                            st.warning(f"Could not load horse data: {str(e)}")

                        st.info("Navigate to 'Horse Past Performance' to see detailed analysis")

                    else:
                        st.error(f"Error parsing PDF: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Is the backend running on http://localhost:8000?")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The PDF may be too large or the API is overloaded.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def horses_overview_page():
    st.header("🐎 Horses Overview with AI Analysis")

    try:
        response = requests.get(f"{API_BASE_URL}/races", timeout=15)
        
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
                default_index = next((i for i, race in enumerate(races) if race['id'] == st.session_state.last_race_id), 0)
            else:
                default_index = 0
            
            selected_race = st.selectbox(
                "Choose a session:",
                options=races,
                format_func=lambda x: f"{x['track_name']} - {x['horses_count']} horses ({x.get('analysis_date', x['race_date'])})",
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
        response = requests.get(f"{API_BASE_URL}/races/{race['id']}/horses")
        
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

def statistics_page():
    st.header("📊 Statistics")
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        
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
        response = requests.get(f"{API_BASE_URL}/health")
        
        if response.status_code == 200:
            health = response.json()
            st.success("✅ API is healthy")
            st.json(health)
        else:
            st.error("❌ API is not responding")
        
        # Parser status
        st.subheader("🤖 Parser Status")
        try:
            parser_response = requests.get(f"{API_BASE_URL}/parser-status")
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