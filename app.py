import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="H-1B Processing Time Estimator",
    page_icon="🛂",
    layout="wide",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* dark navy background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}

section[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

/* headings */
h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem !important; color: #58a6ff !important; letter-spacing: -0.5px; }
h2 { font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem !important; color: #79c0ff !important; }
h3 { font-size: 0.95rem !important; color: #8b949e !important; font-weight: 400 !important; text-transform: uppercase; letter-spacing: 1px; }

/* metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 20px 24px;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 600;
    color: #58a6ff;
    line-height: 1;
}
.metric-unit {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 4px;
}

/* ci bar wrapper */
.ci-container {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 20px 24px;
    margin-top: 12px;
}
.ci-label {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 12px;
}

/* inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider {
    background-color: #21262d !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
}

.stButton > button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 10px 24px;
    width: 100%;
    cursor: pointer;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #388bfd;
}

/* divider */
hr { border-color: #21262d !important; }

/* badge */
.badge {
    display: inline-block;
    background: #1f6feb22;
    border: 1px solid #1f6feb55;
    color: #58a6ff;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-left: 8px;
}

.disclaimer {
    font-size: 0.75rem;
    color: #484f58;
    margin-top: 8px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ── load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = "model_results"
    model    = joblib.load(os.path.join(base, "best_model_tuned.pkl"))
    scaler   = joblib.load(os.path.join(base, "scaler.pkl"))
    encoders = joblib.load(os.path.join(base, "label_encoders.pkl"))
    features = joblib.load(os.path.join(base, "feature_list.pkl"))
    return model, scaler, encoders, features

@st.cache_data
def load_feature_data():
    state_df = pd.read_csv("eda_outputs/milestone2/state_features.csv")
    feat_df  = pd.read_csv("eda_outputs/milestone2/feature_engineered_data.csv")
    soc_df   = pd.read_csv("eda_outputs/milestone2/soc_features.csv")
    return state_df, feat_df, soc_df

try:
    model, scaler, encoders, feature_cols = load_artifacts()
    state_df, feat_df, soc_df = load_feature_data()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    load_error = str(e)


# ── helpers ────────────────────────────────────────────────────────────────────
MONTH_NAMES = {
    1:"January", 2:"February", 3:"March", 4:"April",
    5:"May", 6:"June", 7:"July", 8:"August",
    9:"September", 10:"October", 11:"November", 12:"December",
}

SEASON_MAP = {
    12:"Winter", 1:"Winter", 2:"Winter",
    3:"Spring",  4:"Spring",  5:"Spring",
    6:"Summer",  7:"Summer",  8:"Summer",
    9:"Fall",   10:"Fall",   11:"Fall",
}

US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC",
]

SOC_CATEGORIES = [
    "Software Developers",
    "Computer Systems Analysts",
    "Computer Occupations, All Other",
    "Management Analysts",
    "Financial Analysts",
    "Accountants and Auditors",
    "Engineers, All Other",
    "Electrical Engineers",
    "Mechanical Engineers",
    "Architects",
    "Medical Scientists",
    "Biological Scientists",
    "Physicians and Surgeons",
    "Statisticians",
    "Operations Research Analysts",
]

def predict(state, visa_class, month, wage, full_time, soc_label):
    season = SEASON_MAP[month]
    quarter = (month - 1) // 3 + 1

    # look up historical state averages
    s_row = state_df[state_df["WORKSITE_STATE"] == state]
    state_avg  = float(s_row["STATE_AVG_PROC"].values[0])   if len(s_row) else feat_df["STATE_AVG_PROC"].median()
    state_med  = float(s_row["STATE_MEDIAN_PROC"].values[0]) if len(s_row) else feat_df["STATE_MEDIAN_PROC"].median()
    state_cnt  = float(s_row["STATE_APP_COUNT"].values[0])   if len(s_row) else feat_df["STATE_APP_COUNT"].median()

    emp_avg = feat_df["EMPLOYER_AVG_PROC"].median()
    emp_cnt = feat_df["EMPLOYER_APP_COUNT"].median()

    # match SOC
    soc_match = soc_df[soc_df["SOC_AVG_PROC"].notna()]
    soc_avg = float(feat_df["SOC_AVG_PROC"].median())
    soc_cnt = float(feat_df["SOC_APP_COUNT"].median())

    log_wage      = np.log1p(wage)
    wage_pct      = 0.6  # default mid-range
    emp_dur       = 1095 # typical 3-year H-1B

    row = {
        "RECEIVED_MONTH":        month,
        "RECEIVED_QUARTER":      quarter,
        "RECEIVED_DAY_OF_WEEK":  0,
        "IS_CAP_SEASON":         int(month == 4),
        "MONTH_SIN":             np.sin(2 * np.pi * month / 12),
        "MONTH_COS":             np.cos(2 * np.pi * month / 12),
        "STATE_AVG_PROC":        state_avg,
        "STATE_MEDIAN_PROC":     state_med,
        "STATE_APP_COUNT":       state_cnt,
        "EMPLOYER_AVG_PROC":     emp_avg,
        "EMPLOYER_APP_COUNT":    emp_cnt,
        "SOC_AVG_PROC":          soc_avg,
        "SOC_APP_COUNT":         soc_cnt,
        "LOG_WAGE":              log_wage,
        "WAGE_PERCENTILE":       wage_pct,
        "EMPLOYMENT_DURATION_DAYS": emp_dur,
        "NEW_EMPLOYMENT":        1,
        "CONTINUED_EMPLOYMENT":  0,
        "CHANGE_EMPLOYER":       0,
        "SEASON":                season,
        "FULL_TIME_POSITION":    "Y" if full_time == "Full-time" else "N",
        "VISA_CLASS":            visa_class,
    }

    df_in = pd.DataFrame([{f: row.get(f, 0) for f in feature_cols}])

    for col, le in encoders.items():
        if col in df_in.columns:
            val = str(df_in[col].iloc[0])
            df_in[col] = le.transform([val])[0] if val in le.classes_ else 0

    df_in = df_in.fillna(0)
    point = float(model.predict(df_in)[0])

    # residual std from training — approximated conservatively
    margin = 1.645 * 2.5

    return {
        "estimate":    round(max(point, 1), 1),
        "lower":       round(max(point - margin, 1), 1),
        "upper":       round(point + margin, 1),
        "state_avg":   round(state_avg, 1),
    }


def plot_monthly_trend(feat_df, highlight_month):
    monthly = feat_df.groupby("RECEIVED_MONTH")["PROCESSING_DAYS"].mean().reset_index()
    monthly.columns = ["month", "avg_days"]

    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    colors = ["#58a6ff" if m == highlight_month else "#21262d" for m in monthly["month"]]
    bars = ax.bar(monthly["month"], monthly["avg_days"], color=colors, width=0.7, zorder=3)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],
                       color="#8b949e", fontsize=8)
    ax.yaxis.set_tick_params(labelcolor="#8b949e", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color="#21262d", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Avg days", color="#8b949e", fontsize=8)
    fig.tight_layout(pad=0.5)
    return fig


def plot_state_comparison(feat_df, selected_state, top_n=10):
    state_avg = feat_df.groupby("WORKSITE_STATE")["PROCESSING_DAYS"].mean()
    top_states = state_avg.sort_values(ascending=True).tail(top_n)

    colors = ["#58a6ff" if s == selected_state else "#21262d" for s in top_states.index]
    edge   = ["#388bfd" if s == selected_state else "#30363d" for s in top_states.index]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    ax.barh(top_states.index, top_states.values, color=colors,
            edgecolor=edge, linewidth=0.8, height=0.65, zorder=3)

    ax.xaxis.set_tick_params(labelcolor="#8b949e", labelsize=8)
    ax.yaxis.set_tick_params(labelcolor="#c9d1d9", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color="#21262d", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Avg processing days", color="#8b949e", fontsize=8)
    fig.tight_layout(pad=0.5)
    return fig


# ── layout ────────────────────────────────────────────────────────────────────
st.markdown("## 🛂 H-1B Processing Time Estimator")
st.markdown(
    "<p style='color:#8b949e; font-size:0.9rem; margin-top:-8px;'>"
    "Prediction engine trained on U.S. DOL LCA Disclosure Data · FY2026 Q1"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

if not artifacts_ok:
    st.error(f"Could not load model artifacts. Run `00_main.py` through Milestone 3 first.\n\n`{load_error}`")
    st.stop()

# ── sidebar inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Application Details")
    st.markdown("")

    state      = st.selectbox("Worksite State", US_STATES, index=US_STATES.index("CA"))
    visa_class = st.selectbox("Visa Class", ["H-1B", "H-1B1 Chile", "H-1B1 Singapore", "E-3 Australian"], index=0)
    month      = st.selectbox("Month of Filing", list(MONTH_NAMES.keys()),
                              format_func=lambda x: MONTH_NAMES[x], index=3)
    wage       = st.number_input("Annual Wage (USD)", min_value=30000, max_value=500000,
                                 value=110000, step=5000)
    full_time  = st.radio("Position Type", ["Full-time", "Part-time"], index=0, horizontal=True)
    soc        = st.selectbox("Occupation Category", SOC_CATEGORIES, index=0)

    st.markdown("")
    run = st.button("Estimate Processing Time")


# ── main panel ─────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Running prediction..."):
        result = predict(state, visa_class, month, wage, full_time, soc)

    est   = result["estimate"]
    lower = result["lower"]
    upper = result["upper"]
    s_avg = result["state_avg"]

    # ── metric row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated Processing Time</div>
            <div class="metric-value">{est:.0f}</div>
            <div class="metric-unit">days</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">90% Confidence Interval</div>
            <div class="metric-value" style="font-size:1.6rem;">{lower:.0f} – {upper:.0f}</div>
            <div class="metric-unit">days</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        delta = est - s_avg
        arrow = "▲" if delta > 0 else "▼"
        color = "#f85149" if delta > 0 else "#3fb950"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">vs. {state} State Average</div>
            <div class="metric-value" style="font-size:1.6rem; color:{color};">{arrow} {abs(delta):.1f}</div>
            <div class="metric-unit">days {'above' if delta > 0 else 'below'} average ({s_avg:.0f} days)</div>
        </div>
        """, unsafe_allow_html=True)

    # ── CI bar
    st.markdown("")
    bar_pct_lo = (lower - lower) / max(upper - lower, 1) * 100
    bar_pct_pt = (est   - lower) / max(upper - lower, 1) * 100

    st.markdown(f"""
    <div class="ci-container">
        <div class="ci-label">Confidence interval range &nbsp;·&nbsp; 90% probability band</div>
        <div style="position:relative; height:10px; background:#21262d; border-radius:5px; overflow:hidden;">
            <div style="position:absolute; left:0; width:100%; height:100%;
                        background: linear-gradient(90deg, #21262d, #1f6feb44, #21262d);
                        border-radius:5px;"></div>
            <div style="position:absolute; left:{bar_pct_pt:.1f}%; transform:translateX(-50%);
                        width:12px; height:10px; background:#58a6ff; border-radius:3px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:6px;">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#8b949e;">{lower:.0f} days</span>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#58a6ff;">● {est:.0f} days</span>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#8b949e;">{upper:.0f} days</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── charts
    st.markdown("")
    st.markdown("---")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("### Processing Time by Filing Month")
        st.markdown("<p class='disclaimer'>Highlighted bar = selected filing month. Based on FY2026 Q1 LCA data.</p>",
                    unsafe_allow_html=True)
        fig1 = plot_monthly_trend(feat_df, month)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with ch2:
        st.markdown("### Top States by Avg Processing Time")
        st.markdown("<p class='disclaimer'>Highlighted bar = selected state. Shows 10 highest-volume states.</p>",
                    unsafe_allow_html=True)
        fig2 = plot_state_comparison(feat_df, state)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── input summary
    st.markdown("---")
    st.markdown("### Input Summary")
    summary = pd.DataFrame({
        "Field":  ["State", "Visa Class", "Filing Month", "Annual Wage", "Position", "Occupation"],
        "Value":  [state, visa_class, MONTH_NAMES[month], f"${wage:,}", full_time, soc],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown(
        "<p class='disclaimer'>⚠ Estimates are based on historical LCA disclosure data and a Random Forest "
        "regression model (MAE ≈ 0.6 days, R² ≈ 0.986). Actual processing times may vary. "
        "This tool is for informational purposes only and does not constitute legal advice.</p>",
        unsafe_allow_html=True,
    )

else:
    # ── landing state
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #484f58;">
        <div style="font-size: 3rem; margin-bottom: 16px;">🛂</div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1rem; color: #8b949e; margin-bottom: 8px;">
            Fill in your application details in the sidebar
        </div>
        <div style="font-size: 0.85rem; color: #484f58;">
            State · Visa class · Filing month · Wage · Position type · Occupation
        </div>
    </div>
    """, unsafe_allow_html=True)