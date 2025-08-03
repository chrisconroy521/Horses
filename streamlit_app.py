import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from pathlib import Path
import io
import base64
from datetime import datetime

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
        ["Upload PDF", "Horses Overview", "Individual Horse Analysis", "Race Analysis", "Statistics", "API Status"]
    )
    
    if page == "Upload PDF":
        upload_page()
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

def upload_page():
    st.header("📄 Upload Ragozin Sheet")
    
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
        if st.button("🚀 Parse PDF with Enhanced Analysis", type="primary"):
            with st.spinner("Parsing PDF with AI analysis... This may take a few minutes for large files."):
                try:
                    # Prepare file for upload
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    
                    # Send to API
                    response = requests.post(f"{API_BASE_URL}/upload-pdf", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ PDF parsed successfully with enhanced analysis!")
                        
                        # Display results
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
                            st.metric("Date Range", race_info.get("date_range", "N/A"))
                        
                        # Analysis date and time
                        st.subheader("📅 Analysis Information")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Analysis Date", result.get("analysis_date", "N/A"))
                        with col2:
                            st.metric("Analysis Time", result.get("analysis_time", "N/A"))
                        with col3:
                            st.metric("Processing Duration", result.get("processing_duration", "N/A"))
                        
                        # Show parser information
                        if "parser_info" in result:
                            st.subheader("🤖 Parser Information")
                            parser_info = result["parser_info"]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Parser Used", parser_info["parser_used"].title())
                            with col2:
                                st.metric("GPT Available", "✅" if parser_info["gpt_available"] else "❌")
                        
                        # Show summary
                        st.subheader("📋 Summary")
                        st.info(f"Successfully parsed {race_info['horses_count']} horses with enhanced AI analysis including symbol interpretation and performance insights.")
                        
                        # Store race ID in session state
                        st.session_state.last_race_id = result["race_id"]
                        
                        # Auto-navigate to view page
                        st.info("Navigate to 'Horses Overview' to see detailed analysis with AI insights")
                        
                    else:
                        st.error(f"❌ Error parsing PDF: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

def horses_overview_page():
    st.header("🐎 Horses Overview with AI Analysis")
    
    try:
        # Fetch races from API
        response = requests.get(f"{API_BASE_URL}/races")
        
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
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

def display_enhanced_horses_details(race):
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
                        lines = horse.get('lines', [])
                        
                        # Calculate statistics with robust None handling
                        try:
                            # Extract numeric figures from fig strings (e.g., "20", "17-", "26+")
                            ragozin_figures = []
                            for line in lines:
                                fig = line.get('fig', '')
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
                        for line_entry in lines:
                            try:
                                all_races.append({
                                    'horse_name': horse_name,
                                    'sex': sex,
                                    'age': age,
                                    'breeder_owner': breeder_owner,
                                    'total_races': total_races,
                                    'top_fig': top_fig,
                                    'horse_analysis': horse_analysis,
                                    'performance_trend': performance_trend,
                                    'fig': line_entry.get('fig', ''),
                                    'flags': line_entry.get('flags', []),
                                    'track': line_entry.get('track', ''),
                                    'month': line_entry.get('month', ''),
                                    'surface': line_entry.get('surface', ''),
                                    'race_type': line_entry.get('race_type', ''),
                                    'race_date': line_entry.get('race_date', ''),
                                    'notes': line_entry.get('notes', ''),
                                    'race_analysis': line_entry.get('race_analysis', '')
                                })
                            except Exception as e:
                                st.error(f"Error processing line entry for {horse_name}: {str(e)}")
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
                        required_race_columns = ['horse_name', 'sex', 'age', 'breeder_owner', 'total_races', 'top_fig', 'horse_analysis', 'performance_trend', 'fig', 'flags', 'track', 'month', 'surface', 'race_type', 'race_date', 'notes', 'race_analysis']
                        for col in required_race_columns:
                            if col not in races_df.columns:
                                races_df[col] = 0 if col in ['age', 'total_races'] else ''
                        
                except Exception as e:
                    st.error(f"Error creating DataFrames: {str(e)}")
                    horses_df = pd.DataFrame(columns=['name', 'sex', 'age', 'breeder_owner', 'total_races', 'top_fig', 'avg_ragozin', 'best_ragozin', 'horse_analysis', 'performance_trend'])
                    races_df = pd.DataFrame(columns=['horse_name', 'sex', 'age', 'breeder_owner', 'total_races', 'top_fig', 'horse_analysis', 'performance_trend', 'fig', 'flags', 'track', 'month', 'surface', 'race_type', 'race_date', 'notes', 'race_analysis'])
                
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

if __name__ == "__main__":
    main() 