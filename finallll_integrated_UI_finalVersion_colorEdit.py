import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta

# ── API URLs ──────────────────────────────────────────────────────
CLUSTERING_API = "https://mazenmaher26-aroundu-location-clustering.hf.space"
ANOMALY_API    = "https://mazenmaher26-aroundu-anomaly-detection.hf.space"
SENTIMENT_API  = "https://mazenmaher26-aroundu-sentiment.hf.space"
PLACE_ID       = 5   # Owner's place_id — change per owner

# --- PAGE SETUP ---
st.set_page_config(page_title="AroundU | Owner Dashboard", layout="wide")

st.markdown("""
<style>

.stApp {
    background-color: #FFFFFF;
}

h1, h2, h3 {
    color: #1D3143;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1D3143;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* KPI Cards — auto light/dark mode */
.kpi-card {
    background: var(--background-color, white);
    padding: 22px;
    border-radius: 14px;
    border-left: 6px solid #2F5C85;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    transition: all 0.25s ease;
    cursor: pointer;
}

.kpi-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    border-left: 6px solid #61A3BB;
}

.kpi-title { font-size: 14px; color: var(--text-color-muted, #65797E); }
.kpi-value { font-size: 34px; font-weight: bold; color: var(--text-color, inherit); }
.kpi-delta { font-size: 14px; color: #61A3BB; }

/* Light mode */
@media (prefers-color-scheme: light) {
    .kpi-card  { background: #FFFFFF; }
    .kpi-title { color: #65797E; }
    .kpi-value { color: #1D3143; }
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    .kpi-card  { background: #1E293B; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    .kpi-title { color: #94A3B8; }
    .kpi-value { color: #F1F5F9; }
}

/* DATE INPUT */
[data-baseweb="input"] input {
    color: #1D3143 !important;
    font-weight: 500 !important;
}
div[data-baseweb="input"] span {
    color: #1D3143 !important;
}

/* CALENDAR - أسماء أيام الأسبوع */
[data-baseweb="calendar"] [role="columnheader"] {
    color: #65797E !important;
    font-weight: 600 !important;
}

/* اليوم الحالي */
[data-baseweb="calendar"] [aria-current="date"] {
    border: 2px solid #2F5C85 !important;
    border-radius: 50% !important;
}

/* إزالة outline */
button:focus, button:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)


# --- MOCK DATA ENGINE ---
@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-03-01", periods=365, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Visits': np.random.randint(100, 500, size=365),
        'Saves': np.random.randint(10, 60, size=365),
        'Directions': np.random.randint(20, 100, size=365),
        'Calls': np.random.randint(5, 40, size=365),
        'Orders': np.random.randint(30, 150, size=365),
        'Chat_Queries': np.random.randint(50, 200, size=365),
        'Bot_Success_Rate': np.random.uniform(70, 95, size=365),
        'Review_Sentiment': np.random.choice(['Positive', 'Negative'], size=365, p=[0.8, 0.2])
    })
    return data

df_raw = load_data()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("🏙️ AroundU")
    st.caption("Beni Suef Business Intelligence")

    selected = option_menu(
        "Main Menu",
        ["Dashboard", "Customer Insights", "Operations", "Anomaly Detection", "Location Logic"],
        icons=['speedometer2', 'chat-heart', 'clock-history', 'exclamation-triangle', 'geo-alt'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#1D3143", "padding": "5px"},
            "menu-title": {"color": "#FFFFFF", "font-weight": "bold", "font-size": "15px"},
            "icon": {"color": "#61A3BB", "font-size": "18px"},
            "nav-link": {
                "color": "white", "font-size": "16px",
                "text-align": "left", "margin": "2px",
                "--hover-color": "#619FB8",
            },
            "nav-link-selected": {
                "background-color": "#2F5C85",
                "color": "white", "font-weight": "bold"
            }
        }
    )

    st.markdown("---")
    st.write(f"Logged in as: **Puffy and Fluffy**")

    # Anomaly alert badge
    st.markdown("---")
    st.markdown("### ⚠️ Alerts")
    try:
        resp = requests.post(f"{ANOMALY_API}/detect",
            json={"visits": [{"user_id":1,"place_id":PLACE_ID,"user_lat":29.0661,
                "user_lon":31.0994,"visited_at":"2026-03-15 10:00:00","cluster":0}]},
            timeout=5)
        if resp.status_code == 200:
            count = resp.json()["total_anomalies"]
            if count > 0:
                st.error(f"🔴 {count} anomal{'y' if count==1 else 'ies'} detected for your place")
            else:
                st.success("✅ No anomalies detected")
        else:
            st.info("🔵 Anomaly service unavailable")
    except Exception:
        st.info("🔵 Anomaly service unavailable")

    st.markdown("### 📅 Select Date Range")
    min_date = df_raw['Date'].min().to_pydatetime()
    max_date = df_raw['Date'].max().to_pydatetime()

    date_range = st.date_input(
        "Choose period:",
        value=(max_date - timedelta(days=30), max_date),
        min_value=min_date,
        max_value=max_date
    )

# =========================
# FILTER LOGIC
# =========================
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    filtered_df = df_raw[(df_raw['Date'] >= start_date) & (df_raw['Date'] <= end_date)]
    period_days = (end_date - start_date).days + 1
    prev_start = start_date - timedelta(days=period_days)
    prev_end = start_date - timedelta(days=1)
    prev_df = df_raw[(df_raw['Date'] >= prev_start) & (df_raw['Date'] <= prev_end)]
else:
    st.warning("Please select a valid date range in the sidebar.")
    st.stop()

# =========================
# 1️⃣ DASHBOARD
# =========================
if selected == "Dashboard":
    st.title("📊 Business Performance Overview")

    m1, m2, m3, m4 = st.columns(4)

    def get_delta_val(col):
        if prev_df.empty:
            return None
        curr = filtered_df[col].sum()
        prev = prev_df[col].sum()
        if prev == 0:
            return "0%"
        diff = ((curr - prev) / prev) * 100
        return f"{int(diff)}%"

    m1.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Visits</div>
    <div class="kpi-value">{int(filtered_df['Visits'].sum())}</div>
    <div class="kpi-delta">{get_delta_val('Visits')}</div></div>""", unsafe_allow_html=True)

    m2.markdown(f"""<div class="kpi-card"><div class="kpi-title">Place Saved</div>
    <div class="kpi-value">{int(filtered_df['Saves'].sum())}</div>
    <div class="kpi-delta">{get_delta_val('Saves')}</div></div>""", unsafe_allow_html=True)

    m3.markdown(f"""<div class="kpi-card"><div class="kpi-title">Direction Clicks</div>
    <div class="kpi-value">{int(filtered_df['Directions'].sum())}</div>
    <div class="kpi-delta">{get_delta_val('Directions')}</div></div>""", unsafe_allow_html=True)

    m4.markdown(f"""<div class="kpi-card"><div class="kpi-title">Call Clicks</div>
    <div class="kpi-value">{int(filtered_df['Calls'].sum())}</div>
    <div class="kpi-delta">{get_delta_val('Calls')}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🚀 Growth Analysis (Current vs Previous Period)")
        metrics = ['Visits', 'Saves', 'Calls']
        curr_vals = [filtered_df[m].sum() for m in metrics]
        prev_vals = [prev_df[m].sum() for m in metrics]
        growth_data = pd.DataFrame({
            'Metric': metrics * 2,
            'Value': curr_vals + prev_vals,
            'Period': ['Selected Period'] * 3 + ['Previous Period'] * 3
        })
        fig_growth = px.bar(growth_data, x='Metric', y='Value', color='Period',
            barmode='group', text='Value', text_auto='.2s',
            color_discrete_map={'Selected Period': '#2F5C85', 'Previous Period': '#61A3BB'},
            template="plotly_white")
        fig_growth.update_traces(textposition='outside')
        fig_growth.update_layout(yaxis_title='Total Count', xaxis_title='Metric',
            margin=dict(t=40, b=40, l=40, r=40))
        st.plotly_chart(fig_growth, use_container_width=True)

    with col_right:
        st.subheader("🤖 Chatbot Stats")
        st.metric("Bot Resolution Rate", f"{filtered_df['Bot_Success_Rate'].mean():.1f}%")
        query_types = pd.DataFrame({'Type': ['Menu', 'Hours', 'Location', 'Pricing'], 'Val': [40, 30, 20, 10]})
        fig_pie = px.pie(query_types, values='Val', names='Type', hole=0.5, color='Type',
            color_discrete_map={'Menu': '#2F5C85', 'Hours': '#61A3BB', 'Location': '#65797E', 'Pricing': '#1D3143'})
        fig_pie.update_traces(textinfo='percent+label', textfont_size=14,
            pull=[0.02, 0.04, 0.02, 0.04], marker=dict(line=dict(color='#FFFFFF', width=2)))
        fig_pie.update_layout(legend_title_text='Query Type', margin=dict(t=20, b=20, l=20, r=20), height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

# =========================
# 2️⃣ CUSTOMER INSIGHTS
# =========================
elif selected == "Customer Insights":
    st.title("🤖 Customer & Review Analysis")

    # ── KPI row ───────────────────────────────────────────────────
    pos_count = (filtered_df["Review_Sentiment"] == "Positive").sum()
    neg_count = (filtered_df["Review_Sentiment"] == "Negative").sum()
    total_rev = pos_count + neg_count
    pos_pct   = f"{int(pos_count/total_rev*100)}%" if total_rev else "N/A"
    neg_pct   = f"{int(neg_count/total_rev*100)}%" if total_rev else "N/A"
    bot_avg   = f"{filtered_df['Bot_Success_Rate'].mean():.1f}%"

    ci1, ci2, ci3, ci4 = st.columns(4)
    with ci1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Reviews</div>
        <div class="kpi-value">{total_rev}</div>
        <div class="kpi-delta">&nbsp;</div></div>""", unsafe_allow_html=True)
    with ci2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Positive</div>
        <div class="kpi-value">{pos_count}</div>
        <div class="kpi-delta">{pos_pct}</div></div>""", unsafe_allow_html=True)
    with ci3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Negative</div>
        <div class="kpi-value">{neg_count}</div>
        <div class="kpi-delta" style="color:#EF4444;">{neg_pct}</div></div>""", unsafe_allow_html=True)
    with ci4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Avg Bot Success</div>
        <div class="kpi-value">{bot_avg}</div>
        <div class="kpi-delta">&nbsp;</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("😊 Sentiment Over Time")
        fig_reviews = px.bar(filtered_df, x="Date", y="Visits", color="Review_Sentiment",
            color_discrete_map={"Positive": "#61A3BB", "Negative": "#EF4444"},
            barmode="stack", template="plotly_white")
        st.plotly_chart(fig_reviews, use_container_width=True)

    with c2:
        st.subheader("⭐ Review Ratings Distribution")
        ratings = np.random.choice([1, 2, 3, 4, 5], size=100, p=[0.05, 0.05, 0.1, 0.3, 0.5])
        fig_rate = px.histogram(x=ratings, nbins=5, color_discrete_sequence=["#2F5C85"],
            template="plotly_white")
        fig_rate.update_layout(xaxis_title="Star Rating", yaxis_title="Count")
        st.plotly_chart(fig_rate, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("🔁 Returning vs New Visitors")
        ret_pct = np.random.randint(55, 75)
        ret_df  = pd.DataFrame({"Type": ["Returning", "New"], "Count": [ret_pct, 100 - ret_pct]})
        fig_ret = px.pie(ret_df, values="Count", names="Type", hole=0.55,
            color_discrete_sequence=["#2F5C85", "#61A3BB"], template="plotly_white")
        st.plotly_chart(fig_ret, use_container_width=True)

    with c4:
        st.subheader("💬 Chat Queries Over Time")
        fig_chat = px.area(filtered_df, x="Date", y="Chat_Queries",
            color_discrete_sequence=["#2F5C85"], template="plotly_white")
        st.plotly_chart(fig_chat, use_container_width=True)

    # ── Real Sentiment from API ───────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 Live Sentiment Analysis")
    sample_reviews = [
        "The food was amazing and the service was great!",
        "Very slow service and the food was cold.",
        "Nice atmosphere but a bit pricey.",
        "Best restaurant in Beni Suef, highly recommended!",
        "Disappointing experience, won't come back.",
    ]
    st.caption("Analyzing sample reviews via Sentiment API...")
    results = []
    for review in sample_reviews:
        try:
            resp = requests.post(
                f"{SENTIMENT_API}/predict",
                json={"text": review},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                results.append({
                    "Review":    review,
                    "Sentiment": data.get("label", "N/A"),
                    "Score":     round(data.get("score", 0), 3),
                })
            else:
                results.append({"Review": review, "Sentiment": "API Error", "Score": 0})
        except Exception:
            results.append({"Review": review, "Sentiment": "Unavailable", "Score": 0})

    if results:
        res_df = pd.DataFrame(results)
        for _, row in res_df.iterrows():
            icon = "😊" if row["Sentiment"] == "Positive" else "😞" if row["Sentiment"] == "Negative" else "😐"
            color = "#D1FAE5" if row["Sentiment"] == "Positive" else "#FEE2E2" if row["Sentiment"] == "Negative" else "#F3F4F6"
            st.markdown(
                f'''<div style="background:{color};padding:10px 16px;border-radius:8px;margin-bottom:8px;">
                {icon} <b>{row["Sentiment"]}</b> ({row["Score"]}) — {row["Review"]}
                </div>''', unsafe_allow_html=True
            )

# =========================
# 3️⃣ OPERATIONS
# =========================
elif selected == "Operations":
    st.title("⏰ Operational Efficiency")

    # ── KPI row ───────────────────────────────────────────────────
    total_calls = f"{filtered_df['Calls'].sum():,}"
    bot_success = f"{filtered_df['Bot_Success_Rate'].mean():.1f}%"

    op1, op2 = st.columns(2)
    with op1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Calls</div>
        <div class="kpi-value">{total_calls}</div>
        <div class="kpi-delta">&nbsp;</div></div>""", unsafe_allow_html=True)
    with op2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Avg Bot Success</div>
        <div class="kpi-value">{bot_success}</div>
        <div class="kpi-delta">&nbsp;</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Visiting hours heatmap ────────────────────────────────────
    st.subheader("⏰ Visiting Hours Heatmap")
    hours     = [f"{i}:00" for i in range(24)]
    days      = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heat_data = np.random.randint(10, 100, size=(7, 24))
    heat_data[:, 12:14] += 50   # Lunch peak
    heat_data[:, 19:22] += 70   # Evening peak
    fig_heat  = px.imshow(heat_data, x=hours, y=days,
        color_continuous_scale=["#ECECEC", "#61A3BB", "#2F5C85"],
        aspect="auto", template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📞 Peak Hours — Calls & Visits")
        peak_hours = pd.DataFrame({
            "Hour":   list(range(8, 23)),
            "Visits": np.random.randint(10, 80, 15),
            "Calls":  np.random.randint(2,  20, 15),
        })
        fig_peak = px.line(peak_hours, x="Hour", y=["Visits", "Calls"],
            color_discrete_map={"Visits": "#2F5C85", "Calls": "#61A3BB"},
            template="plotly_white", markers=True)
        fig_peak.update_layout(xaxis_title="Hour of Day", yaxis_title="Count")
        st.plotly_chart(fig_peak, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🤖 Bot Success Rate Over Time")
        fig_bot = px.line(filtered_df, x="Date", y="Bot_Success_Rate",
            color_discrete_sequence=["#61A3BB"], template="plotly_white")
        fig_bot.update_layout(yaxis_title="Success Rate (%)", yaxis_range=[0, 100])
        st.plotly_chart(fig_bot, use_container_width=True)

    with col4:
        st.subheader("💬 Chat Queries Trend")
        fig_cq = px.bar(filtered_df, x="Date", y="Chat_Queries",
            color_discrete_sequence=["#1D3143"], template="plotly_white")
        st.plotly_chart(fig_cq, use_container_width=True)

# =========================
# 4️⃣ LOCATION LOGIC
# =========================
elif selected == "Anomaly Detection":
    st.title("🚨 Anomaly Detection — Your Place")

    @st.cache_data(ttl=120)
    def fetch_place_anomalies():
        try:
            np.random.seed(99)
            visits = []
            base = datetime(2026, 1, 1, 8, 0, 0)
            for _ in range(300):
                visits.append({
                    "user_id":    int(np.random.randint(1, 200)),
                    "place_id":   int(np.random.randint(1, 50)),
                    "user_lat":   round(29.0661 + np.random.normal(0, 0.005), 6),
                    "user_lon":   round(31.0994 + np.random.normal(0, 0.005), 6),
                    "visited_at": str(base + timedelta(
                        days=int(np.random.randint(0, 60)),
                        hours=int(np.random.randint(8, 23)),
                        minutes=int(np.random.randint(0, 60)))),
                    "cluster": 0,
                })
            # Inject spike for this place
            spike = datetime(2026, 2, 15, 19, 0, 0)
            for i in range(30):
                visits.append({
                    "user_id": int(np.random.randint(200, 400)),
                    "place_id": PLACE_ID,
                    "user_lat": 29.0661, "user_lon": 31.0994,
                    "visited_at": str(spike + timedelta(minutes=int(np.random.randint(0, 60)))),
                    "cluster": 0,
                })
            # Detect
            resp_d = requests.post(f"{ANOMALY_API}/detect",
                json={"visits": visits}, timeout=30)
            if resp_d.status_code != 200:
                return [], []
            all_anomalies = resp_d.json()["anomalies"]
            # Filter for this place
            resp_p = requests.post(f"{ANOMALY_API}/place-anomalies",
                json={"place_id": PLACE_ID, "anomalies": all_anomalies}, timeout=15)
            if resp_p.status_code == 200:
                return resp_p.json()["anomalies"], all_anomalies
        except Exception:
            pass
        return [], []

    with st.spinner("Fetching anomaly data for your place..."):
        place_anomalies, _ = fetch_place_anomalies()

    high   = [a for a in place_anomalies if a["severity"] == "High"]
    medium = [a for a in place_anomalies if a["severity"] == "Medium"]

    # ── High severity alert banner ────────────────────────────────
    if high:
        st.error(f"🔴 **{len(high)} High Severity Anomal{'y' if len(high)==1 else 'ies'} Detected for Your Place!** Review immediately.")

    # ── KPI cards ─────────────────────────────────────────────────
    an1, an2, an3 = st.columns(3)
    with an1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Anomalies</div>
        <div class="kpi-value">{len(place_anomalies)}</div>
        <div class="kpi-delta">&nbsp;</div></div>""", unsafe_allow_html=True)
    with an2:
        st.markdown(f"""<div class="kpi-card" style="border-left:6px solid #EF4444;">
        <div class="kpi-title">High Severity</div>
        <div class="kpi-value" style="color:#EF4444;">{len(high)}</div>
        <div class="kpi-delta">&nbsp;</div></div>""", unsafe_allow_html=True)
    with an3:
        st.markdown(f"""<div class="kpi-card" style="border-left:6px solid #F59E0B;">
        <div class="kpi-title">Medium Severity</div>
        <div class="kpi-value" style="color:#F59E0B;">{len(medium)}</div>
        <div class="kpi-delta">&nbsp;</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    if not place_anomalies:
        st.success("✅ No anomalies detected for your place.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Anomalies by Type")
            type_df = pd.DataFrame(place_anomalies)["anomaly_type"].value_counts().reset_index()
            type_df.columns = ["Type", "Count"]
            fig_at = px.bar(type_df, x="Count", y="Type", orientation="h",
                color="Count", color_continuous_scale="Reds",
                text_auto=True, template="plotly_white")
            fig_at.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_at, use_container_width=True)

        with col2:
            st.subheader("🔴 Severity Distribution")
            sev_df = pd.DataFrame(place_anomalies)["severity"].value_counts().reset_index()
            sev_df.columns = ["Severity", "Count"]
            fig_sev = px.pie(sev_df, values="Count", names="Severity", hole=0.55,
                color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B"},
                template="plotly_white")
            st.plotly_chart(fig_sev, use_container_width=True)

        # ── Live anomaly feed ─────────────────────────────────────
        st.subheader("📋 Your Place Anomaly Feed")
        for a in place_anomalies:
            col_info, col_act = st.columns([5, 1])
            badge = "🔴" if a["severity"] == "High" else "🟡"
            with col_info:
                st.markdown(
                    f"{badge} **{a['anomaly_type'].replace('_',' ').title()}** "
                    f"&nbsp;·&nbsp; Score: `{a['score']}` "
                    f"&nbsp;·&nbsp; User: `{a['user_id']}`"
                )
                st.caption(f"_{a['details']}_")
            with col_act:
                if a["severity"] == "High":
                    st.button("🚫 Report",  key=f"rep_{a['user_id']}_{a['anomaly_type']}")
                st.button("✅ Dismiss", key=f"dis_{a['user_id']}_{a['anomaly_type']}")
            st.divider()

elif selected == "Location Logic":
    st.title("📍 Location Analysis: Beni Suef")
    BS_LAT, BS_LON = 29.0661, 31.0994

    @st.cache_data(ttl=300)
    def fetch_owner_heatmap():
        try:
            np.random.seed(42)
            visits = []
            for _ in range(80):
                visits.append({
                    "lat":      round(BS_LAT + np.random.normal(0, 0.005), 6),
                    "lon":      round(BS_LON + np.random.normal(0, 0.005), 6),
                    "cluster":  0,
                    "district": "City Center",
                })
            resp = requests.post(f"{CLUSTERING_API}/heatmap",
                json={"visits": visits}, timeout=15)
            if resp.status_code == 200:
                return resp.json()["hotspots"]
        except Exception:
            pass
        return None

    hotspots = fetch_owner_heatmap()

    if hotspots:
        map_data = pd.DataFrame(hotspots)
        st.caption("🟢 Live data from Location Clustering API")
    else:
        st.caption("⚠️ Using fallback data — API unavailable")
        map_data = pd.DataFrame({
            "lat":       np.random.uniform(BS_LAT - 0.015, BS_LAT + 0.015, 200),
            "lon":       np.random.uniform(BS_LON - 0.015, BS_LON + 0.015, 200),
            "intensity": np.random.randint(1, 100, 200),
        })

    fig_map = px.density_mapbox(map_data, lat="lat", lon="lon", z="intensity",
        radius=15, center=dict(lat=BS_LAT, lon=BS_LON), zoom=12.5,
        mapbox_style="open-street-map", height=700)
    fig_map.update_layout(
        coloraxis_colorbar=dict(title="Intensity",
            title_font=dict(color="#1D3143"), tickfont=dict(color="#1D3143")),
        margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)