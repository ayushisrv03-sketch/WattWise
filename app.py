"""
WattWise - 7-Day Building Energy Consumption Forecasting
Run with:
    streamlit run app.py

Expected export folder:
    export/models/*.json
    export/building_ids.pkl
    export/features.pkl
    export/full_history.pkl        # preferred
    OR export/recent_data.pkl      # must contain >= 672 hourly rows/building
"""

import os
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="WattWise | 7-Day Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #0b1728 48%, #101c2f 100%);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #081321;
        border-right: 1px solid rgba(255,255,255,.08);
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.08);
        padding: 18px;
        border-radius: 18px;
    }

    .hero {
        padding: 30px 34px;
        border-radius: 24px;
        background:
            radial-gradient(circle at 85% 15%, rgba(56,189,248,.18), transparent 32%),
            radial-gradient(circle at 10% 90%, rgba(34,197,94,.12), transparent 30%),
            rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.09);
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #7dd3fc;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero h1 {
        color: white;
        font-size: 2.6rem;
        line-height: 1.1;
        margin: 0;
    }

    .hero p {
        color: #a8b6c8;
        font-size: 1rem;
        margin-top: 12px;
        max-width: 850px;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 1.4rem 0 .75rem;
    }

    .small-note {
        color: #94a3b8;
        font-size: .88rem;
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.08);
    }

    .status-pill {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: rgba(34,197,94,.12);
        color: #86efac;
        border: 1px solid rgba(34,197,94,.25);
        font-size: .8rem;
        font-weight: 600;
    }

    .warning-pill {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: rgba(245,158,11,.12);
        color: #fcd34d;
        border: 1px solid rgba(245,158,11,.25);
        font-size: .8rem;
        font-weight: 600;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: .78rem;
        margin-top: 35px;
        padding-top: 18px;
        border-top: 1px solid rgba(255,255,255,.07);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS / HOLIDAYS
# ============================================================

EXPORT_DIR = Path("export")
US_HOL = holidays.US()


# ============================================================
# ARTIFACT LOADING
# ============================================================

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
# MUST MATCH THE COLAB TRAINING FEATURES
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

    # During forecasting, kwh for the current future row is NaN.
    # We only require the feature columns themselves to be available.
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
    """
    No real weather forecast exists beyond the BDG2 dataset.

    For each future hour, use the weather from the same hour
    one week earlier. This preserves weekday/hour seasonality
    better than repeating the final observed weather value.
    """

    reference_ts = ts - pd.Timedelta(hours=168)

    if reference_ts in working.index:
        row = working.loc[reference_ts]
    elif reference_ts in base.index:
        row = base.loc[reference_ts]
    else:
        # Fallback: same hour from the most recent available day.
        same_hour = base[base.index.hour == ts.hour]
        if not same_hour.empty:
            row = same_hour.iloc[-1]
        else:
            row = base.iloc[-1]

    return row[["airTemperature", "dewTemperature", "windSpeed"]]


# ============================================================
# RECURSIVE 7-DAY FORECAST
# ============================================================

def forecast_beyond_dataset(bid, models, history, features, horizon_hours=168):
    required_history = 504 + 168

    base = history[bid][
        ["kwh", "airTemperature", "dewTemperature", "windSpeed"]
    ].copy()

    base = base.sort_index()

    if len(base) < required_history:
        raise ValueError(
            f"{bid} has only {len(base)} historical rows available. "
            f"At least {required_history} hourly rows are required for "
            f"lag_504 + roll_168_mean. Re-export recent_data.pkl with "
            f"at least {required_history} rows per building, or export "
            f"full_history.pkl."
        )

    last_ts = base.index.max()

    future_index = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="h",
    )

    working = base.copy()
    predictions = []

    for ts in future_index:
        # Add future row first.
        working.loc[ts, "kwh"] = np.nan

        # Use same hour from the previous week as the weather proxy.
        weather = get_future_weather(ts, base, working)

        working.loc[ts, "airTemperature"] = weather["airTemperature"]
        working.loc[ts, "dewTemperature"] = weather["dewTemperature"]
        working.loc[ts, "windSpeed"] = weather["windSpeed"]

        featured = compute_features(
            working,
            require_target=False,
        )

        if ts not in featured.index:
            raise ValueError(
                f"Could not compute forecasting features for {ts}. "
                f"Check that enough historical rows are available."
            )

        current_features = featured.loc[[ts], features]

        prediction = float(models[bid].predict(current_features)[0])

        # Consumption cannot be negative.
        prediction = max(0.0, prediction)

        predictions.append(prediction)

        # CRITICAL: feed the prediction back into history.
        # This is what makes the forecast recursive.
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
    """
    Simulates the real forecasting situation on a historical date.

    Only data before start_date is used to generate predictions.
    The following 168 actual observations are returned for comparison.
    """

    df = history[bid][
        ["kwh", "airTemperature", "dewTemperature", "windSpeed"]
    ].copy().sort_index()

    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(hours=167)

    if start not in df.index or end not in df.index:
        return None

    known = df.loc[df.index < start].copy()
    actual = df.loc[start:end].copy()

    if len(known) < 672:
        return None

    working = known.copy()
    predictions = []

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
        pred = max(0.0, float(models[bid].predict(x)[0]))

        predictions.append(pred)
        working.loc[ts, "kwh"] = pred

    result = actual[["kwh"]].copy()
    result["predicted_kwh"] = predictions

    return result


# ============================================================
# BASIC HISTORICAL SUMMARY
# ============================================================

def historical_summary(df):
    monthly = df["kwh"].resample("ME").sum()
    daily = df["kwh"].resample("D").sum()

    return monthly, daily


# ============================================================
# STARTUP / VALIDATION
# ============================================================

if not EXPORT_DIR.exists():
    st.error(
        "The export/ folder is missing. Download your Colab export.zip, "
        "extract it, and place the export/ folder next to app.py."
    )
    st.stop()

try:
    models, history, features, history_type = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the forecasting artifacts: {exc}")
    st.stop()


# Check that the exported feature list matches what this app can calculate.
supported_features = {
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "dayofweek", "is_weekend", "is_holiday",
    "lag_24", "lag_168", "lag_336", "lag_504",
    "roll_24_mean", "roll_168_mean",
    "airTemperature", "dewTemperature", "windSpeed",
}

unsupported = [f for f in features if f not in supported_features]

if unsupported:
    st.error(
        "The exported model expects unsupported features: "
        + ", ".join(unsupported)
    )
    st.stop()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine Learning • Time Series • Energy Analytics</div>
        <h1>⚡ WattWise</h1>
        <p>
            Building-level electricity consumption forecasting powered by
            XGBoost, historical consumption patterns, calendar signals and weather.
            Generate a recursive <b>7-day / 168-hour</b> forecast beyond the dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚡ WattWise")
    st.caption("Forecast control center")

    bid = st.selectbox(
        "Select building",
        sorted(models.keys()),
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

    st.divider()

    st.markdown("### Model")
    st.markdown(
        '<span class="status-pill">XGBoost • Active</span>',
        unsafe_allow_html=True,
    )

    st.markdown("### Features")
    st.caption(
        f"{len(features)} exported features • "
        "daily + weekly lags • rolling history • weather"
    )

    if history_type == "full":
        st.markdown(
            '<span class="status-pill">Full history loaded</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="warning-pill">Recent history only</span>',
            unsafe_allow_html=True,
        )


# ============================================================
# DATA INFO
# ============================================================

building_data = history[bid].sort_index()

data_start = building_data.index.min()
data_end = building_data.index.max()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Buildings", f"{len(models)}")
c2.metric("Selected building", str(bid))
c3.metric("Data available", f"{data_start:%d %b %Y} → {data_end:%d %b %Y}")
c4.metric("Forecast horizon", f"{horizon_days} day{'s' if horizon_days != 1 else ''}")


# ============================================================
# TABS
# ============================================================

tab_forecast, tab_backtest, tab_history = st.tabs(
    [
        "🔮 7-Day Forecast",
        "🧪 Historical 7-Day Test",
        "📊 Historical Analytics",
    ]
)


# ============================================================
# FORECAST TAB
# ============================================================

with tab_forecast:

    st.markdown(
        '<div class="section-title">Future Energy Outlook</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="small-note">
        Forecast starts after <b>{data_end:%d %B %Y, %H:%M}</b>.
        The model recursively predicts each hour and uses the previous week's
        weather pattern as a proxy for future weather.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if generate or "forecast_result" not in st.session_state:
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
        forecast_result = st.session_state["forecast_result"]

        if st.session_state.get("forecast_bid") != bid:
            forecast_result = None

    if forecast_result is not None:

        # ---- KPI cards ----
        total = forecast_result["predicted_kwh"].sum()
        avg_hourly = forecast_result["predicted_kwh"].mean()
        peak_idx = forecast_result["predicted_kwh"].idxmax()
        peak_value = forecast_result["predicted_kwh"].max()

        daily = forecast_result["predicted_kwh"].resample("D").sum()

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Predicted consumption",
            f"{total:,.1f} kWh",
        )

        k2.metric(
            "Average / hour",
            f"{avg_hourly:,.2f} kWh",
        )

        k3.metric(
            "Peak load",
            f"{peak_value:,.2f} kWh",
        )

        k4.metric(
            "Peak time",
            peak_idx.strftime("%d %b • %H:%M"),
        )

        st.markdown(
            '<div class="section-title">Hourly Forecast</div>',
            unsafe_allow_html=True,
        )

        # Interactive hourly forecast using Streamlit's built-in chart.
        # No additional plotting dependency is required.
        chart_df = forecast_result[["predicted_kwh"]].rename(
            columns={"predicted_kwh": "Predicted kWh"}
        )

        st.line_chart(
            chart_df,
            use_container_width=True,
            height=440,
        )

        # The daily table is intentionally visible immediately below the graph.
        # No expander/tab/click is required.
        st.markdown(
            '<div class="section-title">Daily Forecast Data</div>',
            unsafe_allow_html=True,
        )

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

        st.info(
            "Future weather is approximated using the same hour from the previous "
            "week. For a production system, replace this proxy with a weather "
            "forecast API. This is the main external-data limitation of the current model."
        )

    else:
        st.markdown(
            """
            <div style="
                padding: 50px;
                text-align: center;
                border-radius: 20px;
                background: rgba(255,255,255,.04);
                border: 1px dashed rgba(255,255,255,.12);
            ">
                <div style="font-size: 3rem;">🔮</div>
                <h3 style="color:white;">Ready to forecast</h3>
                <p style="color:#94a3b8;">
                    Select a building and click <b>Generate Forecast</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# HISTORICAL 7-DAY BACKTEST TAB
# ============================================================

with tab_backtest:

    st.markdown(
        '<div class="section-title">7-Day Recursive Backtest</div>',
        unsafe_allow_html=True,
    )

    if history_type != "full":
        st.warning(
            "The current export contains only recent_data.pkl. "
            "Historical 7-day backtesting requires full_history.pkl. "
            "Use the full-history export cell in Colab to enable this tab."
        )
    else:
        earliest = building_data.index.min() + pd.Timedelta(hours=672)
        latest = building_data.index.max() - pd.Timedelta(hours=167)

        if earliest.date() <= latest.date():

            chosen = st.date_input(
                "Choose a historical forecast start date",
                value=latest.date(),
                min_value=earliest.date(),
                max_value=latest.date(),
            )

            run_backtest = st.button(
                "🧪 Run 7-Day Backtest",
                type="primary",
            )

            if run_backtest:

                with st.spinner("Running 168-hour recursive backtest..."):
                    result = recursive_backtest_7_days(
                        bid,
                        chosen,
                        models,
                        history,
                        features,
                    )

                if result is None:
                    st.error(
                        "Not enough continuous historical data for this date."
                    )
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

                    b1, b2, b3, b4 = st.columns(4)

                    b1.metric(
                        "MAE",
                        f"{mae:.2f} kWh",
                    )

                    b2.metric(
                        "nMAE",
                        f"{nmae:.2f}%",
                    )

                    b3.metric(
                        "Actual total",
                        f"{total_actual:,.1f} kWh",
                    )

                    b4.metric(
                        "Predicted total",
                        f"{total_pred:,.1f} kWh",
                    )

                    compare = result.rename(
                        columns={
                            "kwh": "Actual kWh",
                            "predicted_kwh": "Predicted kWh",
                        }
                    )

                    st.line_chart(
                        compare,
                        use_container_width=True,
                        height=430,
                    )

                    st.dataframe(
                        compare.round(2),
                        use_container_width=True,
                    )


# ============================================================
# HISTORICAL ANALYTICS TAB
# ============================================================

with tab_history:

    st.markdown(
        '<div class="section-title">Historical Consumption</div>',
        unsafe_allow_html=True,
    )

    raw = building_data

    monthly, daily = historical_summary(raw)

    h1, h2, h3, h4 = st.columns(4)

    h1.metric(
        "Average hourly",
        f"{raw['kwh'].mean():,.2f} kWh",
    )

    h2.metric(
        "Peak historical",
        f"{raw['kwh'].max():,.2f} kWh",
    )

    h3.metric(
        "Average daily",
        f"{daily.mean():,.1f} kWh",
    )

    h4.metric(
        "Historical observations",
        f"{len(raw):,}",
    )

    st.markdown(
        '<div class="section-title">Monthly Consumption</div>',
        unsafe_allow_html=True,
    )

    st.bar_chart(
        monthly.rename("Monthly kWh"),
        use_container_width=True,
        height=380,
    )

    st.markdown(
        '<div class="section-title">Average Consumption by Hour</div>',
        unsafe_allow_html=True,
    )

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
    <div class="footer">
        WattWise • XGBoost Time-Series Forecasting • BDG2 Building Energy Data
    </div>
    """,
    unsafe_allow_html=True,
)
