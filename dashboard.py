"""
dashboard.py -- Streamlit + Plotly dashboard for the Disaster Evolution pipeline.

Views:
1. Overview (All 6 Events) - Data from pipeline_cache.json
2. Comparative Analysis (Event A vs B) - Data from nlp_results.json

Usage:
    streamlit run dashboard.py
"""

import json
import os
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -- Page Config ---------------------------------------------------------------

st.set_page_config(
    page_title="Disaster Evolution & Intelligence",
    page_icon="🌪️",
    layout="wide",
)

CACHE_FILE = "pipeline_cache.json"
NLP_FILE = "nlp_results.json"
COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]

# -- CSS -----------------------------------------------------------------------

_CSS = """
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    padding: 2rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;
}
.main-header h1 { margin: 0; font-size: 2rem; }
.main-header p  { margin: 0.3rem 0 0 0; opacity: 0.8; }
.insight-box {
    background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6;
    padding: 1rem; border-radius: 4px; margin: 1rem 0;
}
.finding-box {
    background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981;
    padding: 1rem; border-radius: 4px; margin: 1rem 0; font-weight: bold; font-size: 1.1rem;
}
</style>
"""


# -- Data Loading --------------------------------------------------------------

@st.cache_data(ttl=60)
def load_pipeline_data():
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(ttl=60)
def load_nlp_data():
    if not os.path.exists(NLP_FILE):
        return {}
    with open(NLP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# -- View 1: Overview (All Events) ---------------------------------------------

def render_overview():
    data = load_pipeline_data()
    if not data:
        st.warning(f"Cache {CACHE_FILE} not found. Run pipeline first.")
        return

    event_labels = list(data.keys())
    event_names = {k: v.get("event_name", k) for k, v in data.items()}
    color_map = {event_names[k]: COLORS[i % len(COLORS)] for i, k in enumerate(event_labels)}

    hist_labels = [k for k in event_labels if "Historical" in k or "Event_A" in k]
    recent_labels = [k for k in event_labels if "Recent" in k or "Event_B" in k]

    st.markdown(_CSS, unsafe_allow_html=True)
    _events_str = " | ".join(event_names[k] for k in event_labels)
    st.markdown(
        f'<div class="main-header">'
        f'<h1>Pipeline Overview: 6 Events Dataset</h1>'
        f'<p>Analysing: {_events_str}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 1. Summary Comparison Table
    st.subheader("Summary Comparison & New Metrics")
    rows = []
    for label in event_labels:
        ev = data[label]
        cap = ev.get("cap", {})
        met = ev.get("metrics", {})
        cs = ev.get("corpus_summary", {})
        sv = met.get("sentiment", {})
        nr = met.get("ner", {})
        fc = met.get("forgotten_crisis", {})
        vb = met.get("vulnerability_benchmark", {})
        rd = met.get("response_delta", {})

        tone_str = f"{sv.get('tone_alarmist_pct', 0)}% Alrm / {sv.get('tone_analytical_pct', 0)}% Analy"
        
        rows.append({
            "Event": event_names[label],
            "Cat": "Hist" if label in hist_labels else "Rec",
            "Articles": ev.get("media_count", 0),
            "Corpus Vol": cs.get("total_words", 0),
            "dT (h)": rd.get("delta_hours", ""),
            "FCI": fc.get("index", ""),
            "Tone": tone_str,
            "Coping": vb.get("coping_capacity", ""),
            "Alert": vb.get("alertlevel", ""),
            "Deaths": nr.get("deaths_max_reported", 0),
        })

    df_compare = pd.DataFrame(rows)
    st.dataframe(df_compare, use_container_width=True, hide_index=True)
    st.markdown("---")

    # 2. News Volume Timeline
    st.subheader("News Volume Timeline")
    
    def build_timeline_df(event_data, event_name):
        articles = event_data.get("media", [])
        dates = []
        for art in articles:
            pd_str = art.get("pubdate", "")
            try:
                dt = datetime.fromisoformat(pd_str)
                dates.append(dt.strftime("%Y-%m-%d"))
            except ValueError:
                pass
        counts = Counter(dates)
        df = pd.DataFrame(list(counts.items()), columns=["Date", "Count"])
        df["Event"] = event_name
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date")

    dfs = [build_timeline_df(data[k], event_names[k]) for k in event_labels]
    df_timeline = pd.concat(dfs, ignore_index=True)

    if not df_timeline.empty:
        for ev_name in event_names.values():
            mask = df_timeline["Event"] == ev_name
            if mask.any():
                min_date = df_timeline.loc[mask, "Date"].min()
                df_timeline.loc[mask, "Relative_Day"] = (
                    (df_timeline.loc[mask, "Date"] - min_date).dt.days
                )

        tab1, tab2 = st.tabs(["Absolute Dates", "Relative Timeline (Day 0 aligned)"])
        with tab1:
            fig1 = px.bar(
                df_timeline, x="Date", y="Count", color="Event",
                barmode="group", color_discrete_map=color_map,
                labels={"Count": "Articles", "Date": "Publication Date"},
            )
            fig1.update_layout(template="plotly_dark", height=400, **bg_color() )
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            fig1r = px.line(
                df_timeline, x="Relative_Day", y="Count", color="Event",
                markers=True, color_discrete_map=color_map,
                labels={"Count": "Articles", "Relative_Day": "Days Since First Article"},
            )
            fig1r.update_layout(template="plotly_dark", height=400, **bg_color())
            st.plotly_chart(fig1r, use_container_width=True)
    else:
        st.info("No timeline data.")
    st.markdown("---")

    # 3. Radar Chart
    st.subheader("Normalized Event Comparison Radar")
    def get_radar_values(event_data):
        cap = event_data.get("cap", {})
        return {
            "Magnitude": cap.get("max_wind_kmh", 0) or 0,
            "Exposure": cap.get("population_millions", 0) or 0,
            "News Count": event_data.get("media_count", 0),
            "Vuln Score": {"high": 3, "medium": 2, "low": 1}.get(cap.get("vulnerability", "").lower(), 0),
        }

    all_vals = {k: get_radar_values(data[k]) for k in event_labels}
    categories = list(next(iter(all_vals.values())).keys())
    max_vals = {cat: max(all_vals[k][cat] for k in event_labels) or 1 for cat in categories}

    fig_radar = go.Figure()
    for i, label in enumerate(event_labels):
        vals = all_vals[label]
        norm = [vals[c] / max_vals[c] for c in categories]
        fig_radar.add_trace(go.Scatterpolar(
            r=norm + [norm[0]], theta=categories + [categories[0]],
            fill="toself", name=event_names[label],
            line=dict(color=COLORS[i % len(COLORS)]),
        ))
    fig_radar.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=450,
    )
    col_r, col_t = st.columns([3, 2])
    with col_r:
        st.plotly_chart(fig_radar, use_container_width=True)
    with col_t:
        st.markdown("**Raw Values**")
        raw_df = pd.DataFrame({
            "Metric": categories,
            **{event_names[k]: [all_vals[k][c] for c in categories] for k in event_labels}
        })
        st.dataframe(raw_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 4. NER Extraction
    st.subheader("NER Intelligence")
    for i in range(0, len(event_labels), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(event_labels): break
            label = event_labels[idx]
            ev = data[label]
            with col:
                st.markdown(f"**{event_names[label]}**")
                ner = ev.get("metrics", {}).get("ner", {})
                deaths = ner.get("deaths_max_reported", 0)
                damage = ner.get("damage_mentions", [])
                
                st.caption(f"Deaths: **{deaths}** | Damage: **{'; '.join(damage) if damage else 'None'}**")
                
                orgs = ner.get("top_organisations", [])
                if orgs:
                    org_df = pd.DataFrame(orgs, columns=["Organisation", "Mentions"]).head(5)
                    fig_org = px.bar(org_df, x="Mentions", y="Organisation", orientation="h",
                                     color_discrete_sequence=[COLORS[idx % len(COLORS)]])
                    fig_org.update_layout(template="plotly_dark", height=200, yaxis=dict(autorange="reversed"), **bg_color())
                    st.plotly_chart(fig_org, use_container_width=True)

    # 5. Text Corpus Validation
    st.markdown("---")
    st.subheader("Text Corpus Validation")
    corpus_rows = []
    for label in event_labels:
        cs = data[label].get("corpus_summary", {})
        sources = cs.get("sources", {})
        corpus_rows.append({
            "Event": event_names[label],
            "Corpus Words": cs.get("total_words", 0),
            "ReliefWeb": sources.get("reliefweb", {}).get("words", 0),
            "Scraped News": sources.get("article_scrape", {}).get("words", 0),
            "Wiki": sources.get("wikipedia", {}).get("words", 0),
        })
    st.dataframe(pd.DataFrame(corpus_rows), use_container_width=True, hide_index=True)

def bg_color():
    return dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


# -- View 2: Comparative Analysis ----------------------------------------------

def render_comparative():
    nlp = load_nlp_data()
    if not nlp:
        st.warning(f"{NLP_FILE} not found. Run `python nlp_analysis.py` first.")
        return

    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="main-header">'
        f'<h1>Intelligence Brief: Comparative View</h1>'
        f'<p>Event A (Historical: Amphan 2020) vs Event B (Recent: Gezani 2026)</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    ev_a = nlp.get("Event_A_Amphan", {})
    ev_b = nlp.get("Event_B_Gezani", {})
    comp = nlp.get("comparative_analysis", {})

    if not ev_a or not ev_b:
        st.error("Missing event data in NLP results.")
        return

    # Key Finding Banner
    st.markdown(f'<div class="finding-box">Key Finding:<br>{comp.get("answer", "")}</div>', unsafe_allow_html=True)

    # Top-line Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Response Delta (Amphan)", f"{ev_a['response_delta']['delta_hours']}h")
    c2.metric("Response Delta (Gezani)", f"{ev_b['response_delta']['delta_hours']}h", delta=f"{ev_b['response_delta']['delta_hours'] - ev_a['response_delta']['delta_hours']}h", delta_color="inverse")
    c3.metric("FCI (Amphan)", ev_a['forgotten_crisis']['index'])
    c4.metric("FCI (Gezani)", ev_b['forgotten_crisis']['index'])

    st.markdown("---")
    
    colA, colB = st.columns(2)
    
    # Event A Column
    with colA:
        st.subheader("🅰️ Event A: Amphan (Hist)")
        st.caption(f"{ev_a['country']} | {ev_a['period']}")
        st.markdown(f"**Deaths:** {ev_a['ner']['casualties']['max_reported']:,}")
        st.markdown(f"**Displaced:** {ev_a['ner']['displaced']['max_reported']:,}")
        st.markdown(f"**Coping Capacity:** {ev_a['vulnerability_benchmark']['coping_capacity']} (Alert: {ev_a['vulnerability_benchmark']['alert_level']})")
        
        # Tone donut chart
        tone_a = [ev_a['sentiment']['tone_alarmist_pct'], ev_a['sentiment']['tone_analytical_pct'], 100 - ev_a['sentiment']['tone_alarmist_pct'] - ev_a['sentiment']['tone_analytical_pct']]
        fig_tone_a = px.pie(values=tone_a, names=["Alarmist", "Analytical", "Neutral"], hole=0.6,
                            color_discrete_sequence=["#ef4444", "#3b82f6", "#64748b"], title="Tone Distribution")
        fig_tone_a.update_layout(template="plotly_dark", height=300, margin=dict(t=30, b=0, l=0, r=0), **bg_color())
        st.plotly_chart(fig_tone_a, use_container_width=True)

    # Event B Column
    with colB:
        st.subheader("🅱️ Event B: Gezani (Rec)")
        st.caption(f"{ev_b['country']} | {ev_b['period']}")
        st.markdown(f"**Deaths:** {ev_b['ner']['casualties']['max_reported']:,}")
        st.markdown(f"**Displaced:** {ev_b['ner']['displaced']['max_reported']:,}")
        st.markdown(f"**Coping Capacity:** {ev_b['vulnerability_benchmark']['coping_capacity']} (Alert: {ev_b['vulnerability_benchmark']['alert_level']})")
        
        # Tone donut chart
        tone_b = [ev_b['sentiment']['tone_alarmist_pct'], ev_b['sentiment']['tone_analytical_pct'], 100 - ev_b['sentiment']['tone_alarmist_pct'] - ev_b['sentiment']['tone_analytical_pct']]
        fig_tone_b = px.pie(values=tone_b, names=["Alarmist", "Analytical", "Neutral"], hole=0.6,
                            color_discrete_sequence=["#ef4444", "#3b82f6", "#64748b"], title="Tone Distribution")
        fig_tone_b.update_layout(template="plotly_dark", height=300, margin=dict(t=30, b=0, l=0, r=0), **bg_color())
        st.plotly_chart(fig_tone_b, use_container_width=True)

    st.markdown("---")
    st.subheader("Comparative Insights")
    st.markdown(f'<div class="insight-box"><b>⏱️ Response Speed:</b> {comp.get("response_speed", {}).get("insight", "")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box"><b>📢 Coverage Gap (FCI):</b> {comp.get("forgotten_crisis", {}).get("insight", "")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box"><b>🎭 Tone Evolution:</b> {comp.get("tone_evolution", {}).get("insight", "")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box"><b>🛡️ Vulnerability:</b> {comp.get("vulnerability", {}).get("insight", "")}</div>', unsafe_allow_html=True)


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    st.sidebar.title("Navigation")
    view_mode = st.sidebar.radio("Select View:", [
        "Comparative Analysis (Event A vs B)",
        "Overview (All 6 Events)"
    ])
    
    if view_mode == "Overview (All 6 Events)":
        render_overview()
    else:
        render_comparative()
