"""
WattWise - 7-Day Building Energy Consumption Forecasting
Multi-Page Dark Blue & White Edition

Run with:
    python -m streamlit run app.py
"""

from pathlib import Path
import holidays
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WattWise | Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM DARK BLUE & WHITE UI DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Color Palette: Dark Blue & Pure White */
    :root {
        --bg-deep: #0a192f;
        --bg-surface: #112240;
        --bg-surface-elevated: #172a45;
        --border-white: rgba(255, 255, 255, 0.14);
        --border-white-hover: rgba(255, 255, 255, 0.35);
        --text-pure-white: #ffffff;
        --text-muted-white: #e2e8f0;
        --text-sub: #94a3b8;
        --accent-blue: #38bdf8;
        --accent-glow: rgba(56, 189, 248, 0.25);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--text-pure-white);
    }

    /* Main background */
    .stApp {
        background: radial-gradient(circle at 10% 10%, #0f2744 0%, #0a192f 55%, #060e1a 100%) fixed;
    }

    .block-container {
        max-width: 1420px;
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
    }

    /* Sidebar Dark Blue Styling */
    [data-testid="stSidebar"] {
        background-color: #0b1d35 !important;
        border-right: 1px solid var(--border-white) !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
        margin: 1.2rem 0;
    }

    /* Hero Header */
    .hero-box {
        background: linear-gradient(135deg, rgba(23, 42, 69, 0.95) 0%, rgba(13, 33, 55, 0.95) 100%);
        border: 1px solid var(--border-white);
        border-radius: 20px;
        padding: 26px 32px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
    }

    .hero-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #ffffff 0%, #38bdf8 50%, #ffffff 100%);
    }

    .hero-eyebrow {
        color: #38bdf8;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .hero-box h1 {
        color: #ffffff;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.15;
    }

    .hero-box p {
        color: var(--text-muted-white);
        font-size: 0.98rem;
        margin-top: 8px;
        max-width: 880px;
        line-height: 1.5;
    }

    /* Top Overview Cards */
    .overview-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }

    .overview-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-white);
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }

    .overview-card-label {
        color: var(--text-sub);
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .overview-card-value {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Stat Cards */
    .stat-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-white);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .stat-card:hover {
        transform: translateY(-3px);
        border-color: var(--border-white-hover);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.45);
    }

    .stat-label {
        color: var(--text-sub);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .stat-value {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 800;
        margin-top: 6px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }

    .stat-sub {
        color: #38bdf8;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Section Titles */
    .page-title {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin: 1.2rem 0 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .page-desc {
        color: var(--text-muted-white);
        font-size: 0.92rem;
        margin-bottom: 18px;
    }

    /* Status Badges */
    .badge-white {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.25);
        font-size: 0.78rem;
        font-weight: 600;
    }

    .badge-blue {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: #ffffff !important;
        color: #0a192f !important;
        font-weight: 800 !important;
        border: 1px solid #ffffff !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(255, 255, 255, 0.2) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #e2e8f0 !important;
        color: #060e1a !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 22px rgba(255, 255, 255, 0.35) !important;
    }

    /* Tables & DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--border-white);
    }

    /* Info / Callout Box */
    .info-callout {
        background: var(--bg-surface);
        border: 1px solid var(--border-white);
        border-left: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 14px 18px;
        color: var(--text-muted-white);
        font-size: 0.88rem;
        margin-top: 18px;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS & ARTIFACT LOADING
# ============================================================

EXPORT_DIR = Path("export")
US_HOL = holidays.US()


@st.cache_resource
def load_artifacts():
    building_ids = joblib.load(EXPORT_DIR / "building_ids.pkl")

    models = {}
    for bid in building_ids:
        model_path = EXPORT_DIR / "models" / f"{bid}.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

        model = XGBRegressor()
        model.load_model(str(model_path))
        models[bid] = model

    features = joblib.load(EXPORT_DIR / "features.pkl")

    full_history_path = EXPORT_DIR / "full_history.pkl"
    recent_history_path = EXPORT_DIR / "recent_data.pkl"

    if full_history_path.exists():
        history = joblib.load(full_history_path)
        history_type = "full"
    elif recent_history_path.exists():
        history = joblib.load(recent_history_path)
        history_type = "recent"
    else:
        raise FileNotFoundError(
            "Neither full_history.pkl nor recent_data.pkl was found in export/."
        )

    return models, history, list(features), history_type


# ============================================================
# FEATURE ENGINEERING
# MUST MATCH THE TRAINING FEATURES
# ============================================================

def compute_features(df, require_target=True):
    df = df.copy()

    # Calendar features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_holiday"] = [int(d in US_HOL) for d in df.index.date]

    # Cyclical features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Consumption memory features
    df["lag_24"] = df["kwh"].shift(24)
    df["lag_168"] = df["kwh"].shift(168)
    df["lag_336"] = df["kwh"].shift(336)
    df["lag_504"] = df["kwh"].shift(504)

    # Historical rolling patterns
    df["roll_24_mean"] = df["kwh"].shift(24).rolling(24).mean()
    df["roll_168_mean"] = df["kwh"].shift(24).rolling(168).mean()

    if require_target:
        return df.dropna()

    feature_cols = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dayofweek", "is_weekend", "is_holiday",
        "lag_24", "lag_168", "lag_336", "lag_504",
        "roll_24_mean", "roll_168_mean",
        "airTemperature", "dewTemperature", "windSpeed",
    ]

    return df.dropna(subset=feature_cols)


# ============================================================
# WEATHER PROXY
# ============================================================

def get_future_weather(ts, base, working):
    reference_ts = ts - pd.Timedelta(hours=168)

    if reference_ts in working.index:
        row = working.loc[reference_ts]
    elif reference_ts in base.index:
        row = base.loc[reference_ts]
    else:
        same_hour = base[base.index.hour == ts.hour]
        if not same_hour.empty:
            row = same_hour.iloc[-1]
        else:
            row = base.iloc[-1]

    return row[["airTemperature", "dewTemperature", "windSpeed"]]


# ============================================================
# RECURSIVE FORECAST (OPTIMIZED TAIL FOR HIGH PERFORMANCE)
# ============================================================

def forecast_beyond_dataset(bid, models, history, features, horizon_hours=168):
    required_history = 504 + 168

    full_base = history[bid][
        ["kwh", "airTemperature", "dewTemperature", "windSpeed"]
    ].copy().sort_index()

    if len(full_base) < required_history:
        raise ValueError(
            f"{bid} has only {len(full_base)} historical rows available. "
            f"At least {required_history} hourly rows are required."
        )

    # Keep last 720 rows for speed while maintaining all lags
    base = full_base.iloc[-720:].copy()
    last_ts = base.index.max()

    future_index = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="h",
    )

    working = base.copy()
    predictions = []
    model = models[bid]

    for ts in future_index:
        working.loc[ts, "kwh"] = np.nan
        weather = get_future_weather(ts, base, working)

        working.loc[ts, "airTemperature"] = weather["airTemperature"]
        working.loc[ts, "dewTemperature"] = weather["dewTemperature"]
        working.loc[ts, "windSpeed"] = weather["windSpeed"]

        featured = compute_features(working, require_target=False)

        if ts not in featured.index:
            raise ValueError(f"Could not compute forecasting features for {ts}.")

        current_features = featured.loc[[ts], features]
        prediction = float(model.predict(current_features)[0])
        prediction = max(0.0, prediction)

        predictions.append(prediction)
        working.loc[ts, "kwh"] = prediction

    forecast = pd.DataFrame(
        {"predicted_kwh": predictions},
        index=future_index,
    )
    forecast["date"] = forecast.index.date
    forecast["hour"] = forecast.index.hour

    return forecast


# ============================================================
# HISTORICAL BACKTEST
# ============================================================

def recursive_backtest_7_days(bid, start_date, models, history, features):
    df = history[bid][
        ["kwh", "airTemperature", "dewTemperature", "windSpeed"]
    ].copy().sort_index()

    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(hours=167)

    if start not in df.index or end not in df.index:
        return None

    # Keep last 720 rows of known history before start
    known = df.loc[df.index < start].iloc[-720:].copy()
    actual = df.loc[start:end].copy()

    if len(known) < 672:
        return None

    working = known.copy()
    predictions = []
    model = models[bid]

    for ts in actual.index:
        working.loc[ts, "kwh"] = np.nan
        weather = get_future_weather(ts, known, working)

        working.loc[ts, "airTemperature"] = weather["airTemperature"]
        working.loc[ts, "dewTemperature"] = weather["dewTemperature"]
        working.loc[ts, "windSpeed"] = weather["windSpeed"]

        featured = compute_features(working, require_target=False)

        if ts not in featured.index:
            return None

        x = featured.loc[[ts], features]
        pred = max(0.0, float(model.predict(x)[0]))

        predictions.append(pred)
        working.loc[ts, "kwh"] = pred

    result = actual[["kwh"]].copy()
    result["predicted_kwh"] = predictions

    return result


# ============================================================
# HISTORICAL SUMMARY
# ============================================================

def historical_summary(df):
    monthly = df["kwh"].resample("ME").sum()
    daily = df["kwh"].resample("D").sum()
    return monthly, daily


# ============================================================
# LOAD DATA & VALIDATION
# ============================================================

if not EXPORT_DIR.exists():
    st.error("The export/ folder is missing. Place the export/ folder next to app.py.")
    st.stop()

try:
    models, history, features, history_type = load_artifacts()
except Exception as exc:
    st.error(f"Could not load artifacts: {exc}")
    st.stop()


# ============================================================
# SIDEBAR NAVIGATION & BUILDING CONTROLS
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
            <div style="background:#ffffff; color:#0a192f; width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.3rem; font-weight:900; box-shadow:0 0 14px rgba(255,255,255,0.4);">
                ⚡
            </div>
            <div>
                <h3 style="margin:0; font-size:1.35rem; font-weight:800; color:#ffffff; letter-spacing:-0.02em;">WattWise</h3>
                <div style="font-size:0.75rem; color:#38bdf8; font-weight:700; letter-spacing:0.08em;">ENERGY FORECASTING</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Multi-Page Navigation for the 3 tabs
    st.markdown(
        """
        <div style="font-size:0.76rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
            Select Page
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Page Navigation",
        [
            "🔮 7-Day Forecast",
            "🧪 Historical 7-Day Test",
            "📊 Historical Analytics",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Facility & Horizon Controls
    st.markdown(
        """
        <div style="font-size:0.76rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
            Facility Control
        </div>
        """,
        unsafe_allow_html=True,
    )

    bid = st.selectbox(
        "Select building",
        sorted(models.keys()),
        index=3, # Fox_office_Clayton
    )

    horizon_days = st.slider(
        "Forecast horizon",
        min_value=1,
        max_value=7,
        value=7,
        step=1,
        help="Recursive forecast horizon. The trained model predicts one hour at a time and feeds each prediction into the next hour.",
    )

    generate = st.button(
        "🔮 Generate Forecast",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")

    # Model & Data Badges
    st.markdown("### Model")
    st.markdown('<span class="badge-white">XGBoost • Active</span>', unsafe_allow_html=True)

    st.markdown("### Features")
    st.caption(f"{len(features)} features • daily/weekly lags • rolling history • weather")

    if history_type == "full":
        st.markdown('<span class="badge-blue">Full history loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-white">Recent history only</span>', unsafe_allow_html=True)


# ============================================================
# BUILDING OVERVIEW TOP HEADER
# ============================================================

building_data = history[bid].sort_index()
data_start = building_data.index.min()
data_end = building_data.index.max()

# Hero Header in Dark Blue & Pure White
st.markdown(
    f"""
    <div class="hero-box">
        <div class="hero-eyebrow">Machine Learning • Time Series • Energy Analytics</div>
        <h1>⚡ WattWise Energy Intelligence</h1>
        <p>
            Building-level electricity consumption forecasting powered by XGBoost, historical consumption patterns,
            calendar signals, and weather proxies. Selected Facility: <b style="color:#ffffff;">{bid}</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 4 Stat Cards at the top
st.markdown(
    f"""
    <div class="overview-grid">
        <div class="overview-card">
            <div class="overview-card-label">Monitored Facilities</div>
            <div class="overview-card-value">{len(models)} Buildings</div>
        </div>
        <div class="overview-card">
            <div class="overview-card-label">Active Facility</div>
            <div class="overview-card-value" style="font-size:1.15rem; color:#38bdf8;">{bid}</div>
        </div>
        <div class="overview-card">
            <div class="overview-card-label">Data Coverage</div>
            <div class="overview-card-value" style="font-size:1.05rem;">{data_start:%d %b %Y} → {data_end:%d %b %Y}</div>
        </div>
        <div class="overview-card">
            <div class="overview-card-label">Forecast Horizon</div>
            <div class="overview-card-value">{horizon_days} Day{'s' if horizon_days != 1 else ''} ({horizon_days*24}h)</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1: 🔮 7-DAY FORECAST
# ============================================================

if page == "🔮 7-Day Forecast":

    st.markdown('<div class="page-title">🔮 Future Energy Outlook</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="page-desc">
            Forecast starts after <b style="color:#ffffff;">{data_end:%d %B %Y, %H:%M}</b>.
            The model recursively predicts each hour and uses the previous week's weather pattern as a proxy.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compute or retrieve forecast
    if generate or "forecast_result" not in st.session_state or st.session_state.get("forecast_bid") != bid:
        try:
            with st.spinner("Generating recursive forecast..."):
                forecast_result = forecast_beyond_dataset(
                    bid,
                    models,
                    history,
                    features,
                    horizon_hours=horizon_days * 24,
                )
            st.session_state["forecast_result"] = forecast_result
            st.session_state["forecast_bid"] = bid
        except Exception as exc:
            st.error(f"Forecast could not be generated: {exc}")
            forecast_result = None
    else:
        forecast_result = st.session_state.get("forecast_result")

    if forecast_result is not None:
        total = forecast_result["predicted_kwh"].sum()
        avg_hourly = forecast_result["predicted_kwh"].mean()
        peak_idx = forecast_result["predicted_kwh"].idxmax()
        peak_value = forecast_result["predicted_kwh"].max()
        daily = forecast_result["predicted_kwh"].resample("D").sum()

        # 4 Stat Cards in Dark Blue & White
        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Predicted Consumption</div>
                    <div class="stat-value">{total:,.1f} <span style="font-size:1.1rem; color:#94a3b8; font-weight:500;">kWh</span></div>
                    <div class="stat-sub">Total over {horizon_days} day{'s' if horizon_days != 1 else ''}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Average / Hour</div>
                    <div class="stat-value">{avg_hourly:,.2f} <span style="font-size:1.1rem; color:#94a3b8; font-weight:500;">kWh</span></div>
                    <div class="stat-sub">Hourly mean baseline</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Peak Load</div>
                    <div class="stat-value">{peak_value:,.2f} <span style="font-size:1.1rem; color:#94a3b8; font-weight:500;">kWh</span></div>
                    <div class="stat-sub">Maximum forecast demand</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k4:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Peak Time</div>
                    <div class="stat-value" style="font-size:1.6rem;">{peak_idx.strftime('%d %b • %H:%M')}</div>
                    <div class="stat-sub">Expected peak timestamp</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Hourly Forecast Line Chart
        st.markdown('<div class="page-title">📈 Hourly Forecast</div>', unsafe_allow_html=True)

        chart_df = forecast_result[["predicted_kwh"]].rename(
            columns={"predicted_kwh": "Predicted kWh"}
        )

        st.line_chart(
            chart_df,
            use_container_width=True,
            height=440,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Daily Forecast Data Table
        st.markdown('<div class="page-title">📅 Daily Forecast Data</div>', unsafe_allow_html=True)

        daily_df = pd.DataFrame(
            {
                "Date": daily.index.strftime("%a, %d %b %Y"),
                "Predicted kWh": daily.values,
            }
        )

        daily_df["Predicted kWh"] = daily_df["Predicted kWh"].map(
            lambda x: f"{x:,.2f}"
        )

        st.dataframe(
            daily_df,
            use_container_width=True,
            hide_index=True,
            height=min(360, 58 + len(daily_df) * 38),
        )

        csv = forecast_result[
            ["predicted_kwh"]
        ].to_csv(index_label="timestamp").encode("utf-8")

        st.download_button(
            "⬇️ Download 7-Day Forecast CSV",
            data=csv,
            file_name=f"{bid}_7_day_forecast.csv",
            mime="text/csv",
        )

        st.markdown(
            """
            <div class="info-callout">
                <b>Weather Proxy Notice:</b> Future weather is approximated using the same hour from the previous week.
                For a production system, replace this proxy with a live weather forecast API.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div class="stat-card" style="text-align:center; padding:50px;">
                <div style="font-size:3rem;">🔮</div>
                <h3 style="color:#ffffff;">Ready to Forecast</h3>
                <p style="color:#94a3b8;">Select a building in the sidebar and click <b>Generate Forecast</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE 2: 🧪 HISTORICAL 7-DAY TEST
# ============================================================

elif page == "🧪 Historical 7-Day Test":

    st.markdown('<div class="page-title">🧪 7-Day Recursive Backtest</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="page-desc">
            Simulates the real forecasting situation on a historical date.
            Only data before the selected date is used to generate predictions.
            The following 168 actual observations are compared with predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if history_type != "full":
        st.warning(
            "The current export contains only recent_data.pkl. "
            "Historical 7-day backtesting requires full_history.pkl."
        )
    else:
        earliest = building_data.index.min() + pd.Timedelta(hours=672)
        latest = building_data.index.max() - pd.Timedelta(hours=167)

        if earliest.date() <= latest.date():
            c_date, c_btn = st.columns([1.5, 1])

            with c_date:
                chosen = st.date_input(
                    "Choose a historical forecast start date",
                    value=latest.date(),
                    min_value=earliest.date(),
                    max_value=latest.date(),
                )

            with c_btn:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                run_backtest = st.button(
                    "🧪 Run 7-Day Backtest",
                    type="primary",
                    use_container_width=True,
                )

            if run_backtest or "backtest_data" in st.session_state and st.session_state.get("bt_bid") == bid:
                if run_backtest:
                    with st.spinner("Running 168-hour recursive backtest..."):
                        result = recursive_backtest_7_days(
                            bid,
                            chosen,
                            models,
                            history,
                            features,
                        )
                    st.session_state["backtest_data"] = result
                    st.session_state["bt_bid"] = bid
                else:
                    result = st.session_state["backtest_data"]

                if result is None:
                    st.error("Not enough continuous historical data for this date.")
                else:
                    mae = mean_absolute_error(
                        result["kwh"],
                        result["predicted_kwh"],
                    )

                    nmae = (
                        mae / result["kwh"].mean() * 100
                        if result["kwh"].mean() != 0
                        else np.nan
                    )

                    total_actual = result["kwh"].sum()
                    total_pred = result["predicted_kwh"].sum()

                    # 4 Scorecard Cards in Dark Blue & White
                    b1, b2, b3, b4 = st.columns(4)

                    with b1:
                        st.markdown(
                            f"""
                            <div class="stat-card">
                                <div class="stat-label">MAE</div>
                                <div class="stat-value">{mae:.2f} <span style="font-size:1.1rem; color:#94a3b8; font-weight:500;">kWh</span></div>
                                <div class="stat-sub">Mean Absolute Error</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with b2:
                        st.markdown(
                            f"""
                            <div class="stat-card">
                                <div class="stat-label">nMAE</div>
                                <div class="stat-value">{nmae:.2f}%</div>
                                <div class="stat-sub">Normalized MAE</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with b3:
                        st.markdown(
                            f"""
                            <div class="stat-card">
                                <div class="stat-label">Actual Total</div>
                                <div class="stat-value">{total_actual:,.1f} <span style="font-size:1.1rem; color:#94a3b8; font-weight:500;">kWh</span></div>
                                <div class="stat-sub">Ground truth 7-day total</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with b4:
                        st.markdown(
                            f"""
                            <div class="stat-card">
                                <div class="stat-label">Predicted Total</div>
                                <div class="stat-value">{total_pred:,.1f} <span style="font-size:1.1rem; color:#94a3b8; font-weight:500;">kWh</span></div>
                                <div class="stat-sub">Model 7-day total</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    compare = result.rename(
                        columns={
                            "kwh": "Actual kWh",
                            "predicted_kwh": "Predicted kWh",
                        }
                    )

                    st.markdown('<div class="page-title">📈 Actual vs. Predicted Curve</div>', unsafe_allow_html=True)
                    st.line_chart(
                        compare,
                        use_container_width=True,
                        height=430,
                    )

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="page-title">📋 Comparison Data Table</div>', unsafe_allow_html=True)
                    st.dataframe(
                        compare.round(2),
                        use_container_width=True,
                    )


# ============================================================
# PAGE 3: 📊 HISTORICAL ANALYTICS
# ============================================================

elif page == "📊 Historical Analytics":

    st.markdown('<div class="page-title">📊 Historical Consumption Profile</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="page-desc">
            Historical energy signature and consumption patterns for <b style="color:#ffffff;">{bid}</b>
            across 17,544 continuous hourly measurements.
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw = building_data
    monthly, daily = historical_summary(raw)

    # 4 Historical Stat Cards in Dark Blue & White
    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Average Hourly</div>
                <div class="stat-value">{raw['kwh'].mean():,.2f} <span style="font-size:1.1rem; color:#94a3b8; font-weight:500;">kWh</span></div>
                <div class="stat-sub">Baseline load</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Peak Historical</div>
                <div class="stat-value">{raw['kwh'].max():,.2f} <span style="font-size:1.1rem; color:#94a3b8; font-weight:500;">kWh</span></div>
                <div class="stat-sub">Maximum recorded demand</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Average Daily</div>
                <div class="stat-value">{daily.mean():,.1f} <span style="font-size:1.1rem; color:#94a3b8; font-weight:500;">kWh</span></div>
                <div class="stat-sub">Mean 24h consumption</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Total Observations</div>
                <div class="stat-value">{len(raw):,}</div>
                <div class="stat-sub">Hourly time-series rows</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly Bar Chart
    st.markdown('<div class="page-title">📅 Monthly Consumption</div>', unsafe_allow_html=True)
    st.bar_chart(
        monthly.rename("Monthly kWh"),
        use_container_width=True,
        height=380,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Diurnal Hourly Average Profile
    st.markdown('<div class="page-title">⏱️ Average Consumption by Hour of Day</div>', unsafe_allow_html=True)
    hourly_profile = raw.groupby(raw.index.hour)["kwh"].mean()

    st.line_chart(
        hourly_profile.rename("Average kWh"),
        use_container_width=True,
        height=330,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-text">
        WattWise • XGBoost Time-Series Forecasting • BDG2 Building Energy Data • Dark Blue & White Edition
    </div>
    """,
    unsafe_allow_html=True,
)
