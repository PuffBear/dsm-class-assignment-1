import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Disaster Pulse | Modern Analytics", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")

# --- Modern Minimalist CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #374151;
        background-color: #F9FAFB;
    }
    
    h1, h2, h3, h4, .stHeader {
        font-family: 'Inter', -apple-system, sans-serif !important;
        font-weight: 600 !important;
        color: #111827 !important;
        letter-spacing: -0.025em;
    }
    
    /* Clean background */
    .stApp {
        background-color: #F9FAFB;
    }
    
    /* Panels */
    .mission-panel {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        margin-bottom: 24px;
        color: #4B5563;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .mission-panel b {
        color: #111827;
        font-weight: 600;
    }
    
    /* Stats Boxes */
    .stat-box {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 24px 20px;
        border-radius: 8px;
        text-align: center;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    .stat-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .stat-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }
    
    .stat-label {
        color: #6B7280;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-top: 8px;
    }

    .alert-red { border-top: 4px solid #EF4444; } 
    .alert-red .stat-value { color: #DC2626; }
    
    .alert-orange { border-top: 4px solid #F59E0B; }
    .alert-orange .stat-value { color: #D97706; }
    
    .alert-green { border-top: 4px solid #10B981; }
    .alert-green .stat-value { color: #059669; }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F9FAFB;
        border-radius: 6px 6px 0 0;
        border: 1px solid #E5E7EB;
        border-bottom: none;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #6B7280;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        color: #111827;
        font-weight: 600;
        border-bottom: 2px solid #3B82F6; /* Modern Blue Accent */
        margin-bottom: -1px;
    }
    
    /* Dataframe padding */
    .stDataFrame {
        margin-top: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    file_path = "/Users/Agriya/Desktop/spring26/dsm/Assignment 1/task3_4/cleaned_dataset.csv"
    if not os.path.exists(file_path):
        st.error("Data not found. Please ensure the cleaning script generated standard data.")
        st.stop()
    df = pd.read_csv(file_path)
    
    def extract_lat(loc_str):
        try: return float(loc_str.split(',')[0].strip())
        except: return 0.0
    def extract_lon(loc_str):
        try: return float(loc_str.split(',')[1].strip())
        except: return 0.0
            
    df['lat'] = df['lat_lon'].apply(extract_lat)
    df['lon'] = df['lat_lon'].apply(extract_lon)
    
    import urllib.request
    import json
    
    # --- BONUS REAL-TIME DATA INTEGRATION ---
    # Fetch Real GDP from World Bank JSON API
    def fetch_gdp_billions(country_name):
        iso_map = {'Philippines': 'PHL', 'Afghanistan': 'AFG'}
        iso_code = iso_map.get(country_name, None)
        if not iso_code: return None
        try:
            url = f"http://api.worldbank.org/v2/country/{iso_code}/indicator/NY.GDP.MKTP.CD?format=json&mrnev=1"
            # Setting a browser User-Agent can sometimes help stabilize API pings
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                payload = json.loads(response.read().decode())
                if len(payload) > 1 and len(payload[1]) > 0:
                    raw_gdp = payload[1][0]['value']
                    return round(raw_gdp / 1_000_000_000, 2) # convert to billions
        except Exception as e:
            return None # Fallback trigger

    # Execute fetch for dataset countries
    gdp_cache = {}
    for c in df['country'].unique():
        gdp_cache[c] = fetch_gdp_billions(c)
        
    # Map the live fetched GDP. If API fails for any reason, provide the 2023 baseline absolute fallback.
    df['gdp_b'] = df['country'].map(lambda c: gdp_cache.get(c) if gdp_cache.get(c) is not None else {'Philippines': 437.15, 'Afghanistan': 14.50}.get(c, 100))

    # (Note: Simulated relief funds kept, as centralized global relief API requires private UN keys)
    df['relief_funds_m'] = df['country'].map({'Philippines': 120, 'Afghanistan': 35})
    
    # Predictive Framework Execution
    df['pred_recovery_months'] = (df['coping_capacity_clean'] * 10) / (df['relief_funds_m'] / 10 + 1)
    df['pred_recovery_months'] = df['pred_recovery_months'].round(1)
    
    # Extract NLP Casualty Percentage
    df['casualty_pct'] = df['casualty_articles'].str.extract(r'\(([\d\.]+)%\)').astype(float)
    
    return df

df = load_data()
df_recent = df[df['period'] == 'Recent'].copy()

# --- Modern Color Palette ---
COLOR_RED = '#EF4444' # Tailwind Red 500
COLOR_ORANGE = '#F59E0B' # Tailwind Amber 500
COLOR_GREEN = '#10B981' # Tailwind Emerald 500
COLOR_DARK = '#111827' # Tailwind Gray 900
COLOR_GRAY = '#6B7280' # Tailwind Gray 500

color_map = {'Red': COLOR_RED, 'Orange': COLOR_ORANGE, 'Green': COLOR_GREEN}

with st.sidebar:
    st.markdown("<h2 style='text-align: left; color: #111827; font-size: 1.5rem; font-weight: 700;'><span style='color: #3B82F6;'>⚡</span> Disaster Pulse</h2>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Navigation", 
                    ["🌍 Global Overview", 
                     "📊 Impact Intel", 
                     "📈 Trends & Radar", 
                     "⚡ Predictive Modeling (Bonus)",
                     "📑 Intelligence Brief"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.caption("Release v2.0.0 • Modern Minimalist Interface")

if menu == "🌍 Global Overview":
    st.markdown("<h1>Global Overview</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='mission-panel'>A comprehensive overview of recent and historical seismic events. Analyzing key metrics and evaluating disaster impact on a global scale.</div>", unsafe_allow_html=True)
    
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"<div class='stat-box'><div class='stat-value'>{len(df)}</div><div class='stat-label'>Total Events Logged</div></div>", unsafe_allow_html=True)
    with cols[1]:
        max_mag = df['magnitude_clean'].max()
        st.markdown(f"<div class='stat-box alert-red'><div class='stat-value'>{max_mag}M</div><div class='stat-label'>Peak Magnitude</div></div>", unsafe_allow_html=True)
    with cols[2]:
        total_exposed = df['exposed_population_mmi_clean'].sum() / 1_000_000
        st.markdown(f"<div class='stat-box alert-orange'><div class='stat-value'>{total_exposed:.1f}M</div><div class='stat-label'>Population Exposed</div></div>", unsafe_allow_html=True)
    with cols[3]:
        avg_vuln = df['coping_capacity_clean'].mean()
        st.markdown(f"<div class='stat-box alert-green'><div class='stat-value'>{avg_vuln:.1f}</div><div class='stat-label'>Avg Vulnerability Score</div></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # BONUS: Interactive Mapping via Mapbox
    st.markdown("### 🌍 Global Exposure Heatmap")
    st.markdown("<span style='color: #6B7280; font-size: 0.9em; margin-bottom: 10px; display: inline-block;'>(Interactive 3D Globe: Visualizing population exposure concentration.)</span>", unsafe_allow_html=True)
    
    # BONUS: Live Comparison Toggle
    toggle_map = st.radio("Dataset Filter:", ["All Events", "Recent Only"], horizontal=True)
    map_df = df if toggle_map == "All Events" else df_recent

    fig_map = px.scatter_geo(map_df, lat='lat', lon='lon', 
                             size='exposed_population_mmi_clean',
                             color='exposed_population_mmi_clean',
                             hover_name='label',
                             projection="orthographic",
                             color_continuous_scale=px.colors.sequential.YlOrRd,
                             opacity=0.8,
                             size_max=40)
    
    fig_map.update_geos(
        showcountries=True, countrycolor="#E5E7EB",
        showcoastlines=True, coastlinecolor="#D1D5DB",
        showland=True, landcolor="#F3F4F6",
        showocean=True, oceancolor="rgba(0,0,0,0)",
        projection_rotation=dict(lon=85, lat=20, roll=0)
    )
    
    fig_map.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(title="Population<br>Exposed")
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Display comparison context briefly
    st.markdown("<br>", unsafe_allow_html=True)
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("**Historical Average Magnitude:**")
        st.markdown(f"<h3 style='margin-top: 0;'>{df[df['period']=='Historical']['magnitude_clean'].mean():.1f}M</h3>", unsafe_allow_html=True)
    with comp_col2:
        st.markdown("**Recent Average Magnitude:**")
        st.markdown(f"<h3 style='margin-top: 0;'>{df[df['period']=='Recent']['magnitude_clean'].mean():.1f}M</h3>", unsafe_allow_html=True)

elif menu == "📊 Impact Intel":
    st.markdown("<h1>Impact Intel</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["[3.1] 'Forgotten Crisis' Index", "[3.2] Vulnerability Benchmark", "[3.3] Sensationalism & Severity"])
    
    with tab1:
        st.markdown("<div class='mission-panel'><b>Objective:</b> Determine which disaster received relative 'over-coverage' and which was 'under-reported' by extracting a ratio of News Volume down to Population Impact.</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            most_forgotten = df.loc[df['forgotten_crisis_index'].idxmin()]
            st.markdown(f"<div class='stat-box alert-red'><div class='stat-value'>{most_forgotten['country']}</div><div class='stat-label'>Most Under-Reported Region</div></div>", unsafe_allow_html=True)
            st.markdown("<br><span style='color: #4B5563; font-size: 0.9em;'><b>Metric Definition:</b> A higher index signifies more media coverage generated per affected individual. A lower index directly points to a systematically overlooked crisis despite population exposure.</span>", unsafe_allow_html=True)
            
        with col2:
            fig_fci = px.bar(df, x='forgotten_crisis_index', y='label', color='alert_level',
                             orientation='h',
                             labels={'forgotten_crisis_index': 'Articles per Affected Person', 'label': ''},
                             color_discrete_map=color_map)
            
            fig_fci.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color=COLOR_DARK),
                xaxis=dict(showgrid=True, gridcolor='#E5E7EB'),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_fci, use_container_width=True)

    with tab2:
        st.markdown("<div class='mission-panel'><b>Objective:</b> Analyze if a pervasive lack of Coping Capacity during an event leads definitively to a higher Alert Level, regardless of raw seismic Magnitude.</div>", unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_vuln = px.scatter(df_recent, x='magnitude_clean', y='coping_capacity_clean',
                                  size='exposed_population_mmi_clean', color='alert_level',
                                  hover_name='country', text='label',
                                  labels={'magnitude_clean': 'Seismic Magnitude (M)', 'coping_capacity_clean': 'Lack of Coping Capacity (INFORM Score)', 'alert_level': 'Alert Level'},
                                  color_discrete_map=color_map)
            fig_vuln.update_traces(textposition='top center', textfont_color=COLOR_DARK)
            fig_vuln.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color=COLOR_DARK),
                xaxis=dict(showgrid=True, gridcolor='#E5E7EB', zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='#E5E7EB', zeroline=False)
            )
            st.plotly_chart(fig_vuln, use_container_width=True)
            
        with c2:
            st.markdown("<div class='mission-panel' style='background: #FEF2F2; border-color: #FCA5A5;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #991B1B; margin-top:0;'>🔴 Afghanistan (Red Alert)</h4>", unsafe_allow_html=True)
            st.markdown("• **Magnitude:** 6.3M (Lower)<br>• **Vulnerability:** 7.5 (High Lack of Capacity)<br>→ **Outcome:** Severe infrastructure scaling issues triggered maximum alert warnings.", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='mission-panel' style='background: #FFFBEB; border-color: #FDE68A;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #B45309; margin-top:0;'>🟠 Philippines (Orange Alert)</h4>", unsafe_allow_html=True)
            st.markdown("• **Magnitude:** 6.9M (Higher)<br>• **Vulnerability:** 4.2 (Lower Lack of Capacity)<br>→ **Outcome:** Higher baseline resilience effectively managed the extreme tremor, resulting in a lower alert classification.", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='mission-panel'><b>Objective:</b> Measure Media 'Sensationalism / Severity Focus' by analyzing the NLP-extracted percentage of total press articles explicitly highlighting human casualties compared to general event coverage.</div>", unsafe_allow_html=True)
        
        c_bar, c_text = st.columns([2, 1])
        with c_bar:
            fig_cas = px.bar(df, x='country', y='casualty_pct', color='alert_level',
                             text='casualty_pct',
                             labels={'country': 'Event Region', 'casualty_pct': '% of Articles Reporting Casualties'},
                             color_discrete_map=color_map, barmode='group')
            fig_cas.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_cas.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color=COLOR_DARK),
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#E5E7EB', range=[0, 100]),
                title="Proportion of Press Dedicated to Casualties"
            )
            st.plotly_chart(fig_cas, use_container_width=True)
            
        with c_text:
            st.markdown("### Severity Focus", unsafe_allow_html=True)
            st.markdown("This metric dynamically extracts the NLP string scraped from GDACS reflecting exactly how many subset articles focus on immediate death/injury.")
            st.markdown("Events mapped with devastating relative vulnerability metrics (like Afghanistan) tend to generate a proportionally **vastly higher ratio** of casualty-focused reporting (>55%), whereas disasters successfully mitigated by infrastructure (Philippines) attract coverage distributed more toward general economic/structural recovery (~33%).")

elif menu == "📈 Trends & Radar":
    st.markdown("<h1>Trends & Radar</h1>", unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        # [4.1] Core Metrics & Media Lag Theory
        st.markdown("<div class='mission-panel'><b>Media Lifecycle Analysis:</b> Evaluating the response Delta_T. Plotting normalized news drops over time to determine if 'Compassion Fatigue' takes hold faster for recent events compared to historical events.</div>", unsafe_allow_html=True)
        
        timeline_data = []
        for idx, row in df.iterrows():
            news_str = row['news_per_day']
            if pd.isna(news_str): continue
            days = news_str.split(';')
            for d in days:
                try:
                    date_str, count = d.strip().split(':')
                    timeline_data.append({'Event': row['label'], 'Period': row['period'], 'DateStr': date_str, 'Count': int(count)})
                except: pass
                
        df_timeline_raw = pd.DataFrame(timeline_data)
        df_timeline_aligned = pd.DataFrame()
        for event in df_timeline_raw['Event'].unique():
            subset = df_timeline_raw[df_timeline_raw['Event'] == event].iloc[::-1].reset_index(drop=True)
            subset['Days_Since'] = subset.index
            # Normalize to peak=100 for compassion fatigue analysis
            max_val = subset['Count'].max()
            if max_val > 0:
                subset['Fatigue_Ratio'] = subset['Count'] / max_val * 100
            else:
                subset['Fatigue_Ratio'] = 0
            df_timeline_aligned = pd.concat([df_timeline_aligned, subset])
            
        fig_time = px.line(df_timeline_aligned, x='Days_Since', y='Fatigue_Ratio', color='Event', line_dash="Period",
                           markers=True, color_discrete_sequence=[COLOR_ORANGE, COLOR_RED, COLOR_GREEN, COLOR_GRAY],
                           labels={'Fatigue_Ratio': 'Relative Volume (% of Peak)', 'Days_Since': 'Days Elapsed Since Event'})
                           
        fig_time.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color=COLOR_DARK),
            xaxis=dict(showgrid=True, gridcolor='#E5E7EB'),
            yaxis=dict(showgrid=True, gridcolor='#E5E7EB'),
            title="Attention Entropy: The 'Media Lag' Theory",
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
    with col_t2:
        st.markdown("<div class='mission-panel'><b>Resilience Radar:</b> A normalized comparative breakdown of core event metrics: Seismic Magnitude, Population Exposure, Media Coverage (Scale), and System Vulnerability.</div>", unsafe_allow_html=True)
        
        def normalize(series):
            return (series - series.min()) / (series.max() - series.min() + 1e-9)
        
        df_radar = df_recent.copy()
        df_radar['Norm_Magnitude'] = normalize(df_radar['magnitude_clean'])
        df_radar['Norm_Exposure'] = normalize(df_radar['exposed_population_mmi_clean'])
        df_radar['Norm_Media'] = normalize(df_radar['total_articles'])
        df_radar['Norm_Vulnerability'] = normalize(df_radar['coping_capacity_clean'])
        
        categories = ['Magnitude', 'Exposure', 'Total Media', 'Fragility']
        fig_radar = go.Figure()
        
        fill_colors = ['rgba(245, 158, 11, 0.2)', 'rgba(239, 68, 68, 0.2)']
        line_colors = [COLOR_ORANGE, COLOR_RED]
        
        for idx, (i, row) in enumerate(df_radar.iterrows()):
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Norm_Magnitude'], row['Norm_Exposure'], row['Norm_Media'], row['Norm_Vulnerability']],
                theta=categories,
                fill='toself',
                name=row['country'],
                fillcolor=fill_colors[idx%2],
                line=dict(color=line_colors[idx%2], width=2)
            ))
            
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1]), 
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color=COLOR_DARK),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=40, b=40, l=40, r=40),
            title="Symmetrical Resilience Output"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

elif menu == "⚡ Predictive Modeling (Bonus)":
    st.markdown("<h1>Predictive Modeling & Integrated Data</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='mission-panel'><b>Cross-Domain Correlation:</b> Combining macro-economic constraints (Live GDP metrics), external funding flows, and dynamic sentiment scraping alongside GDACS fragility scores to project recovery times.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px 16px; border-radius: 6px; margin-bottom: 24px; font-size: 0.9em; color: #475569;'>
        <b style='color: #0F172A;'>⚠️ Data Sourcing Disclaimer:</b> This demonstration utilizes a hybrid data approach:
        <ul style='margin-top: 8px; margin-bottom: 0; padding-left: 20px;'>
            <li><b>Real-Time Data:</b> GDP metrics are fetched dynamically via live HTTP requests to the <i>World Bank Public API</i>.</li>
            <li><b>Simulated Data:</b> Emergency Relief Funds and Social Sentiment indices are mock-injected to demonstrate predictive capability mapping without requiring authenticated private UN/Social Media API keys.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        # Economic Correlation Bubble
        fig_econ = px.scatter(df_recent, x='gdp_b', y='coping_capacity_clean', size='pred_recovery_months',
                              color='country', hover_name='country',
                              labels={'gdp_b': 'Macroeconomic Base - GDP ($ Billions)', 'coping_capacity_clean': 'Systemic Fragility (0-10)', 'pred_recovery_months': 'Predicted Recovery (Months)'},
                              color_discrete_map={'Philippines': COLOR_ORANGE, 'Afghanistan': COLOR_RED})
        
        fig_econ.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color=COLOR_DARK),
            xaxis=dict(showgrid=True, gridcolor='#E5E7EB', type='log'),
            yaxis=dict(showgrid=True, gridcolor='#E5E7EB'),
            title="Economic Base vs. Fragmented Vulnerability"
        )
        st.plotly_chart(fig_econ, use_container_width=True)
        
    with col2:
        st.markdown("<h3 style='margin-bottom: 24px;'>Predicted Recovery Horizons</h3>", unsafe_allow_html=True)
        for idx, row in df_recent.iterrows():
            st.markdown(f"<div style='margin-bottom: 8px;'><span style='font-weight: 600; color: #111827;'>{row['country']}</span></div>", unsafe_allow_html=True)
            st.progress(min(int(row['pred_recovery_months'] * 4), 100))
            st.caption(f"Estimated Projection: {row['pred_recovery_months']} Months | Base Fragility: {row['coping_capacity_clean']} | Incoming Relief Aid: ${row['relief_funds_m']}M")

    # Sentiment Evolution mock
    st.markdown("---")
    st.markdown("### 📡 Real-Time Social Sentiment Pipeline")
    st.markdown("<span style='color: #6B7280; font-size: 0.9em; margin-bottom: 10px; display: inline-block;'>(Simulated extraction from X/Reddit comparing localized public alarmism indices over time).</span>", unsafe_allow_html=True)
    
    mock_sentiment = pd.DataFrame({
        'Timeline (Days)': [1,2,3,4,5],
        'Philippines (Public Discourse)': [80, 75, 50, 30, 15],
        'Afghanistan (Public Discourse)': [90, 85, 80, 70, 65]
    }).melt(id_vars='Timeline (Days)', var_name='Region Scope', value_name='Panic/Alarm Index (%)')
    
    fig_sent = px.area(mock_sentiment, x='Timeline (Days)', y='Panic/Alarm Index (%)', color='Region Scope', line_group='Region Scope',
                       color_discrete_sequence=[COLOR_ORANGE, COLOR_RED])
    fig_sent.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(family='Inter', color=COLOR_DARK),
        xaxis=dict(gridcolor='#E5E7EB'), yaxis=dict(gridcolor='#E5E7EB')
    )
    st.plotly_chart(fig_sent, use_container_width=True)

elif menu == "📑 Intelligence Brief":
    st.markdown("<h1>Intelligence Brief</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='mission-panel' style='font-size: 1.05rem;'>
    <h3 style='margin-top:0; color:#111827; border-bottom: 1px solid #E5E7EB; padding-bottom: 12px; margin-bottom: 16px;'>[4.2] Final Report: The Evolution of Disaster Reporting</h3>
    
    <p><b>Technical Friction in Data Collection:</b></p>
    <ul style='color: #374151; margin-top: 8px; padding-left: 20px; line-height: 1.8;'>
        <li><b>"False Tabs" Architecture:</b> GDACS tabs ("Summary", "Impact", "Media") are separate pages, not JS tabs. Tab-clicking triggered <code>StaleElementReferenceExceptions</code> in Selenium. Fix: rewrote scraper to construct each tab's URL and use direct <code>driver.get(url)</code> calls.</li>
        <li><b>Inconsistent DOM Hierarchies:</b> Data buried in legacy <code>&lt;table&gt;</code> layouts, not semantic tags. GDACS/Tsunami scores hidden in <code>tbody#tableScoreMain</code> with dynamic columns (Tsunami column absent for landlocked Afghanistan). Alert Level not in plaintext—extracted via regex on the <code>&lt;title&gt;</code> tag (e.g., "Overall Orange Earthquake").</li>
        <li><b>Tooltip Parsing for Media Data:</b> Daily article counts were rendered as a visual <code>&lt;div&gt;</code> bar chart, not a table. Numbers extracted from hidden <code>title</code> hover-attributes (e.g., <code>"2020-08-21T00:00:00: 3"</code>) using a custom regex pattern.</li>
        <li><b>Highly Variable String Formats:</b> Population fields inconsistent across eras—older events used <i>"About 11000 people in MMI"</i>, newer ones <i>"860 thousand (in MMI>=VII)"</i>, requiring a waterfall of regex normalization. Secondary Risk fields (e.g., landslides) were unstructured prose and dropped entirely due to unreliable extraction.</li>
    </ul>
    
    <p><b>Disaster Evolution & The Attention Economy:</b><br>
    When analyzing the Dual-Timeline and the "Media Lag" Theory visual plots, a critical sociological question arises: Are global systems and media responding faster today than in the past? From a strictly data-driven volume perspective, the answer is undoubtedly yes. In contemporary events, news volume indices spike almost instantaneously, compressing the traditional reporting delay into a matter of hours compared to historical event timelines that took days to peak. However, this hyper-fast reporting cycle also brings an accelerated "Compassion Fatigue" or Attention Entropy, where the decay rate of coverage drops steeply and exponentially after just 48 hours.</p>
    
    <p>Beyond timeline speed, the extracted data conclusively reveals that international media focus and global alert systems do not align symmetrically with pure population exposure bounds or even raw seismic magnitude. Our localized <b>Forgotten Crisis Index</b>—which calculates the ratio of media articles published per affected person—exposes a distinct asymmetry in how different crises attract global reporting coverage.</p>
    
    <ul style='color: #374151; list-style-type: none; padding-left: 0; margin-top: 16px;'>
        <li style='margin-bottom: 12px; padding-left: 16px; border-left: 3px solid #F59E0B; background:#FFFBEB; padding:12px; border-radius: 4px;'><b>Philippines Case Study (The Under-Reported):</b> The data indicates a 6.9M magnitude event that triggered massive localized population exposure (over 860,000 individuals). Yet, it yielded significantly lower media coverage mapped <i>per individual person</i>. Relative to its absolute physiological scale and population density, it qualified mathematically as a "forgotten" or under-reported macro-event. Its lower overall Alert Level (Orange) stems primarily from the region's moderate, established coping capacity.</li>
        <li style='padding-left: 16px; border-left: 3px solid #EF4444; background:#FEF2F2; padding:12px; border-radius: 4px;'><b>Afghanistan Case Study (The Highly-Tracked Vulnerability):</b> Conversely, this event presented far lower absolute population exposure numbers (70,000) and a lower raw magnitude (6.3M). However, the region carries a devastating vulnerability profile defined by critically low intrinsic coping capacities (an INFORM score of 7.5). Global analytic systems correctly elevated this to a maximum Red Alert, and correlated international media reporting was proportionally massive per displaced person, recognizing the systemic fragility. A massive ~57% of these articles actively focused on casualties, compared to just ~33% in the Philippines.</li>
    </ul>
    
    <p style='margin-top: 24px; padding-top: 16px; border-top: 1px solid #E5E7EB; color: #111827;'><b>Strategic Conclusion:</b> <br>
    <span style='color: #4B5563;'>The Vulnerability Benchmark clearly proves that raw seismic magnitude alone does not dictate the severity of a global alert; pre-existing infrastructure and economic fragility (Coping Capacity) are the dominant driving factors. Furthermore, modern reporting cycles are categorically faster, but the global attention economy remains deeply erratic. It frequently prioritizes regions with systemic infrastructure fragility, acute geopolitical contexts, or catastrophic potential over purely objective metrics of physiological exposure. To truly build resilient global response systems, aid organizations must rely on integrated predictive modeling—tracking macroeconomic context and historical structural deficits alongside real-time tremor data.</span></p>
    </div>
    """, unsafe_allow_html=True)
