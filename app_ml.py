import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "models/model_bundle.joblib"
METRICS_PATH = "models/validation_metrics.csv"
PREDICTIONS_PATH = "models/validation_predictions.csv"

st.set_page_config(
    page_title="Petrodollar AI — De-dollarization Intelligence",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Visual system — dark executive analytics dashboard
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
  --bg: #07101d;
  --panel: #0d1a2a;
  --panel2: #102238;
  --line: #1f3550;
  --text: #f5f7fb;
  --muted: #91a4bb;
  --cyan: #36b9ff;
  --blue: #5b8cff;
  --green: #29d391;
  --amber: #ffbf5b;
  --purple: #a970ff;
  --red: #ff6b7a;
}

.stApp {
  background:
    radial-gradient(circle at 84% 4%, rgba(36,126,255,.14), transparent 28%),
    radial-gradient(circle at 10% 88%, rgba(82,45,180,.10), transparent 30%),
    var(--bg);
}

.block-container { padding-top: 1.35rem; padding-bottom: 1rem; max-width: 1500px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#091423 0%,#0b1625 100%); border-right:1px solid #152a41; }
[data-testid="stSidebar"] * { color:#eaf1fa; }

.hero {
  border: 1px solid #183553;
  border-radius: 20px;
  padding: 1.15rem 1.35rem 1.05rem 1.35rem;
  margin-bottom: 1rem;
  background:
    linear-gradient(120deg, rgba(16,40,66,.96), rgba(8,18,32,.94)),
    radial-gradient(circle at 90% 0%, rgba(54,185,255,.22), transparent 32%);
  box-shadow: 0 14px 40px rgba(0,0,0,.20);
}
.hero-title { font-size:2.35rem; font-weight:800; letter-spacing:-.035em; line-height:1.05; margin:0; }
.hero-subtitle { color:#9db1c8; margin:.35rem 0 0; font-size:.98rem; }
.eyebrow { color:#5fc5ff; font-size:.74rem; font-weight:800; text-transform:uppercase; letter-spacing:.12em; margin-bottom:.35rem; }

.section-title { font-size:1.38rem; font-weight:800; margin:.25rem 0 .15rem; }
.section-note { color:var(--muted); font-size:.86rem; margin-bottom:.7rem; }

.kpi {
  min-height:132px;
  border:1px solid #1c3550;
  border-radius:16px;
  padding:1rem 1rem .9rem;
  background:linear-gradient(145deg,rgba(17,36,59,.96),rgba(10,23,39,.96));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.025), 0 10px 28px rgba(0,0,0,.12);
}
.kpi.cyan { border-top:2px solid var(--cyan); }
.kpi.green { border-top:2px solid var(--green); }
.kpi.amber { border-top:2px solid var(--amber); }
.kpi.purple { border-top:2px solid var(--purple); }
.kpi.red { border-top:2px solid var(--red); }
.kpi-label { color:#9fb2c7; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.055em; }
.kpi-value { color:#fff; font-size:2rem; font-weight:800; margin-top:.25rem; letter-spacing:-.025em; }
.kpi-delta { color:#54e5aa; font-size:.78rem; font-weight:700; margin-top:.12rem; }
.kpi-delta.neutral { color:#9fb2c7; }

.insight {
  min-height:125px;
  border:1px solid #1b344e;
  border-radius:15px;
  padding:1rem;
  background:linear-gradient(145deg,rgba(13,31,51,.95),rgba(8,20,34,.95));
}
.insight-title { font-weight:800; font-size:.9rem; margin-bottom:.4rem; }
.insight-text { color:#b7c5d5; font-size:.82rem; line-height:1.45; }

.panel {
  border:1px solid #1b344e;
  border-radius:17px;
  padding:1rem 1.05rem;
  background:linear-gradient(145deg,rgba(13,29,48,.93),rgba(8,20,34,.93));
  margin-bottom:.8rem;
}

.badge {
  display:inline-block; padding:.24rem .55rem; border-radius:999px;
  background:rgba(41,211,145,.12); color:#55e5ae; border:1px solid rgba(41,211,145,.25);
  font-size:.72rem; font-weight:800;
}
.badge-blue { background:rgba(54,185,255,.11); color:#62c8ff; border-color:rgba(54,185,255,.25); }
.badge-amber { background:rgba(255,191,91,.10); color:#ffd083; border-color:rgba(255,191,91,.24); }

div[data-testid="stMetric"] { background:transparent; }
div[data-testid="stMetricLabel"] { color:#9fb2c7; }
div[data-testid="stMetricValue"] { color:#fff; }

.stTabs [data-baseweb="tab-list"] { gap:.25rem; background:transparent; }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:.7rem .95rem; color:#aabbd0; }
.stTabs [aria-selected="true"] { color:#fff; background:#102238; border-bottom:2px solid #36b9ff; }

div[data-testid="stDataFrame"] { border:1px solid #1b344e; border-radius:14px; overflow:hidden; }
.stDownloadButton button { border:1px solid #2a5276; background:#0d2238; color:#eaf5ff; }

.footer-disclaimer {
  margin-top:1.25rem;
  padding:.85rem 1rem;
  border:1px solid rgba(255,191,91,.35);
  border-radius:13px;
  background:linear-gradient(90deg,rgba(88,61,16,.52),rgba(45,35,16,.32));
  color:#e7d39c;
  font-size:.78rem;
  line-height:1.45;
}
.footer-disclaimer strong { color:#ffd16e; }

.chart-caption { color:#71879e; font-size:.75rem; margin-top:-.35rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


bundle = load_bundle()
if bundle is None:
    st.error("Trained model artifact not found.")
    st.code("python train_model.py")
    st.stop()

history = bundle["history"].copy()
history["date"] = pd.to_datetime(history["date"])
history = history.sort_values("date").reset_index(drop=True)
latest_date = pd.Timestamp(bundle["latest_date"])
latest_index = float(bundle["latest_index"])
feature_cols = bundle["feature_cols"]
features = pd.DataFrame([bundle["latest_features"]])[feature_cols]
slope = float(bundle["latest_features"]["idx_trend_slope_12"])
horizons = list(bundle["horizons"])


def forecast_for_horizon(horizon):
    model = bundle["models"][horizon]
    trend_component = latest_index + slope * horizon
    residual = float(model.predict(features)[0])
    forecast = trend_component + residual
    unc = float(bundle.get("residual_std", {}).get(horizon, 0.0))
    future_date = latest_date + pd.DateOffset(months=horizon)
    return forecast, trend_component, residual, unc, future_date


forecasts = {h: forecast_for_horizon(h) for h in horizons}

# -----------------------------------------------------------------------------
# Freshness calculation — used only for the final disclaimer.
# -----------------------------------------------------------------------------
today = pd.Timestamp.today().normalize()
months_old = max(0, (today.to_period("M") - latest_date.to_period("M")).n)

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🎛️ Dashboard controls")
st.sidebar.caption("Use these controls to explore the model output without changing the trained artifact.")
selected_horizon = st.sidebar.selectbox("Detailed forecast horizon", horizons, index=0)
history_window = st.sidebar.radio("History window", ["5Y", "10Y", "20Y", "All"], index=2, horizontal=True)
show_uncertainty = st.sidebar.toggle("Show uncertainty", value=True)
show_trend = st.sidebar.toggle("Show trend baseline", value=True)

if history_window == "All":
    history_view = history.copy()
else:
    years = int(history_window.replace("Y", ""))
    history_view = history[history["date"] >= latest_date - pd.DateOffset(years=years)].copy()

forecast, trend_component, residual, unc, future_date = forecasts[selected_horizon]

recent_12 = history.tail(12)["dedollarization_index"]
change_12m = float(recent_12.iloc[-1] - recent_12.iloc[0]) if len(recent_12) >= 2 else np.nan
median_index = float(history["dedollarization_index"].median())
percentile = float((history["dedollarization_index"] <= latest_index).mean() * 100)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div class="eyebrow">GLOBAL ENERGY • CURRENCY • ANALYTICS</div>
  <div class="hero-title">💵 Petrodollar AI — De-dollarization Intelligence</div>
  <div class="hero-subtitle">Track the analytical index, inspect model drivers, validate historical performance, and explore scenario horizons.</div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Executive KPI strip
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">Executive dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="section-note">A compact view of the latest index and the four model scenario endpoints.</div>', unsafe_allow_html=True)

kpi_specs = [
    ("Latest index", f"{latest_index:.2f}", f"{change_12m:+.2f} over recent 12M" if pd.notna(change_12m) else "Recent change unavailable", "cyan"),
    ("12M slope / month", f"{slope:+.4f}", "Index trend input", "green"),
]
for h in horizons:
    val = forecasts[h][0]
    kpi_specs.append((f"{h}M scenario", f"{val:.2f}", f"{val-latest_index:+.2f} vs latest", "purple" if h == 36 else "amber" if h >= 24 else "blue"))

cols = st.columns(6)
for col, (label, value, delta, tone) in zip(cols, kpi_specs):
    col.markdown(
        f'<div class="kpi {tone}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-delta">{delta}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# -----------------------------------------------------------------------------
# Main navigation tabs
# -----------------------------------------------------------------------------
executive_tab, forecast_tab, validation_tab, drivers_tab, methodology_tab = st.tabs(
    ["🏠 Executive view", "🔮 Forecast", "📊 Validation", "🧠 Drivers", "📘 Methodology"]
)


def history_chart_df(df):
    return df[["date", "dedollarization_index"]].rename(columns={"date": "Date", "dedollarization_index": "Index"})


def render_history_forecast_chart(df, horizon, include_all_scenarios=False):
    hist = history_chart_df(df)
    rows = []
    for h in horizons:
        f, _, _, u, fd = forecasts[h]
        if include_all_scenarios:
            rows.append({"Date": fd, "Index": f, "Series": f"{h}M scenario", "Horizon": h})
    scenario_df = pd.DataFrame(rows)

    layers = []
    hist_line = {
        "mark": {"type": "line", "color": "#43b8ff", "strokeWidth": 2.5},
        "encoding": {
            "x": {"field": "Date", "type": "temporal", "axis": {"title": None, "labelColor": "#8ea4bc", "gridColor": "#18304a"}},
            "y": {"field": "Index", "type": "quantitative", "axis": {"title": "De-dollarization index", "titleColor": "#9db1c8", "labelColor": "#8ea4bc", "gridColor": "#18304a"}},
            "tooltip": [{"field": "Date", "type": "temporal", "title": "Date"}, {"field": "Index", "type": "quantitative", "format": ".2f"}],
        },
    }
    layers.append({"data": {"values": hist.to_dict("records")}, **hist_line})

    if show_trend:
        trend_dates = pd.date_range(latest_date, future_date, periods=50)
        trend_vals = np.linspace(latest_index, forecasts[horizon][1], len(trend_dates))
        trend_df = pd.DataFrame({"Date": trend_dates, "Index": trend_vals})
        layers.append({
            "data": {"values": trend_df.to_dict("records")},
            "mark": {"type": "line", "strokeDash": [7, 5], "color": "#ffbf5b", "strokeWidth": 2},
            "encoding": {"x": {"field": "Date", "type": "temporal"}, "y": {"field": "Index", "type": "quantitative"},
                         "tooltip": [{"field": "Index", "type": "quantitative", "format": ".2f"}]},
        })

    f, _, _, u, fd = forecasts[horizon]
    bridge = pd.DataFrame({"Date": [latest_date, fd], "Index": [latest_index, f]})
    layers.append({
        "data": {"values": bridge.to_dict("records")},
        "mark": {"type": "line", "strokeDash": [3, 4], "color": "#8b6cff", "strokeWidth": 3},
        "encoding": {"x": {"field": "Date", "type": "temporal"}, "y": {"field": "Index", "type": "quantitative"}},
    })

    if show_uncertainty and u > 0:
        band_dates = pd.date_range(latest_date, fd, periods=50)
        progress = np.linspace(0, 1, len(band_dates))
        center = np.linspace(latest_index, f, len(band_dates))
        band = pd.DataFrame({"Date": band_dates, "lower": center - u * progress, "upper": center + u * progress})
        layers.insert(1, {
            "data": {"values": band.to_dict("records")},
            "mark": {"type": "area", "color": "#7e6cff", "opacity": 0.12},
            "encoding": {"x": {"field": "Date", "type": "temporal"}, "y": {"field": "lower", "type": "quantitative"}, "y2": {"field": "upper"}},
        })

    point = pd.DataFrame({"Date": [fd], "Index": [f], "Scenario": [f"{horizon}-month scenario"]})
    layers.append({
        "data": {"values": point.to_dict("records")},
        "mark": {"type": "point", "filled": True, "size": 170, "color": "#29d391", "stroke": "#ffffff", "strokeWidth": 1.5},
        "encoding": {"x": {"field": "Date", "type": "temporal"}, "y": {"field": "Index", "type": "quantitative"},
                     "tooltip": [{"field": "Scenario"}, {"field": "Date", "type": "temporal"}, {"field": "Index", "type": "quantitative", "format": ".2f"}]},
    })

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": "container",
        "height": 410,
        "background": "transparent",
        "layer": layers,
        "config": {"view": {"stroke": "#1a344e"}, "axis": {"domainColor": "#2a4560"}, "legend": {"labelColor": "#9db1c8", "titleColor": "#9db1c8"}},
    }
    st.vega_lite_chart(spec, use_container_width=True)


with executive_tab:
    left, right = st.columns([2.3, 1])
    with left:
        st.markdown('<div class="section-title">De-dollarization index — history & scenario path</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Use the sidebar to change the history window and scenario horizon; hover over the line for exact observations.</div>', unsafe_allow_html=True)
        render_history_forecast_chart(history_view, selected_horizon)
    with right:
        st.markdown('<div class="section-title">Scenario breakdown</div>', unsafe_allow_html=True)
        f, t, r, u, fd = forecasts[selected_horizon]
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.metric("Scenario value", f"{f:.2f}", f"{f-latest_index:+.2f} vs latest")
        st.metric("Trend component", f"{t:.2f}")
        st.metric("ML residual", f"{r:+.2f}")
        st.metric("Endpoint uncertainty", f"±{u:.2f}")
        st.markdown(f'<span class="badge badge-blue">Forecast date • {fd:%b %Y}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key signals</div>', unsafe_allow_html=True)
    insight_cols = st.columns(4)
    insights = [
        ("📈 Index position", f"Current index is {percentile:.0f}th historical percentile.", "cyan"),
        ("🧭 Recent direction", f"Recent 12-month change is {change_12m:+.2f}.", "green"),
        ("🔮 Scenario spread", f"The model provides {len(horizons)} forecast horizons for comparison.", "purple"),
        ("🧮 Model structure", "Trend baseline + Random Forest residual adjustment.", "amber"),
    ]
    for col, (title, text, tone) in zip(insight_cols, insights):
        col.markdown(f'<div class="insight"><div class="insight-title">{title}</div><div class="insight-text">{text}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model information</div>', unsafe_allow_html=True)
    info_cols = st.columns(4)
    info_cols[0].metric("Training rows", f"{bundle['training_rows']:,}")
    info_cols[1].metric("Training through", latest_date.strftime("%b %Y"))
    info_cols[2].metric("Validation cutoff", pd.Timestamp(bundle["validation_cutoff"]).strftime("%b %Y"))
    info_cols[3].metric("PCA variance explained", f"{bundle['pca_explained_variance']*100:.1f}%")

with forecast_tab:
    st.markdown('<div class="section-title">Scenario forecast explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">The model estimates an endpoint for the selected horizon; the connecting path is a visual bridge, not a monthly prediction sequence.</div>', unsafe_allow_html=True)
    f, t, r, u, fd = forecasts[selected_horizon]
    a, b, c, d = st.columns(4)
    a.metric("Scenario value", f"{f:.2f}", f"{f-latest_index:+.2f} vs latest")
    b.metric("Trend baseline", f"{t:.2f}")
    c.metric("ML residual", f"{r:+.2f}")
    d.metric("Endpoint uncertainty", f"±{u:.2f}")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    render_history_forecast_chart(history_view, selected_horizon)
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Forecast decomposition</div>', unsafe_allow_html=True)
        decomposition = pd.DataFrame({
            "Component": ["Trend baseline", "ML residual", "Final scenario"],
            "Value": [t, r, f],
        })
        st.dataframe(decomposition.round(3), use_container_width=True, hide_index=True)
    with c2:
        st.markdown('<div class="section-title">Scenario comparison</div>', unsafe_allow_html=True)
        comp = pd.DataFrame([{"Horizon": f"{h}M", "Scenario": forecasts[h][0], "Δ vs latest": forecasts[h][0]-latest_index, "Uncertainty": forecasts[h][3]} for h in horizons])
        st.dataframe(comp.round(3), use_container_width=True, hide_index=True)

with validation_tab:
    st.markdown('<div class="section-title">Historical validation lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Chronological out-of-sample evaluation. Metrics describe historical test performance and are not guarantees of future accuracy.</div>', unsafe_allow_html=True)

    metrics = pd.read_csv(METRICS_PATH) if os.path.exists(METRICS_PATH) else pd.DataFrame()
    preds = pd.read_csv(PREDICTIONS_PATH) if os.path.exists(PREDICTIONS_PATH) else pd.DataFrame()
    if not preds.empty:
        preds["Forecast Date"] = pd.to_datetime(preds["Forecast Date"])
        validation_horizon = st.selectbox("Validation horizon", horizons, index=0, key="validation_horizon_polished")
        pv = preds[preds["Horizon (months)"] == validation_horizon].sort_values("Forecast Date").copy()

        if not metrics.empty:
            row = metrics[metrics["Horizon (months)"] == validation_horizon]
            if not row.empty:
                rr = row.iloc[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Test months", f"{int(rr['Test months'])}")
                m2.metric("MAE", f"{rr['MAE']:.4f}")
                m3.metric("RMSE", f"{rr['RMSE']:.4f}")
                m4.metric("R²", f"{rr['R2']:.4f}")

        chart_df = pv[["Forecast Date", "Actual Index", "Forecast Index"]].copy()
        long_df = chart_df.melt("Forecast Date", var_name="Series", value_name="Index")
        actual = alt = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": "container", "height": 390, "background": "transparent",
            "data": {"values": long_df.to_dict("records")},
            "mark": {"type": "line", "point": {"filled": True, "size": 28}, "strokeWidth": 2.5},
            "encoding": {
                "x": {"field": "Forecast Date", "type": "temporal", "axis": {"title": None, "labelColor": "#8ea4bc", "gridColor": "#18304a"}},
                "y": {"field": "Index", "type": "quantitative", "axis": {"title": "Index", "titleColor": "#9db1c8", "labelColor": "#8ea4bc", "gridColor": "#18304a"}},
                "color": {"field": "Series", "type": "nominal", "scale": {"domain": ["Actual Index", "Forecast Index"], "range": ["#43b8ff", "#a970ff"]}, "legend": {"labelColor": "#aabbd0"}},
                "tooltip": [{"field": "Forecast Date", "type": "temporal"}, {"field": "Series"}, {"field": "Index", "type": "quantitative", "format": ".3f"}],
            },
        }
        v1, v2 = st.columns([2.2, 1])
        with v1:
            st.vega_lite_chart(actual, use_container_width=True)
        with v2:
            residual_df = pv.assign(Residual=pv["Actual Index"] - pv["Forecast Index"])
            hist_spec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "width": "container", "height": 300, "background": "transparent",
                "data": {"values": residual_df[["Residual"]].to_dict("records")},
                "mark": {"type": "bar", "color": "#36b9ff", "opacity": .78},
                "encoding": {"x": {"field": "Residual", "bin": {"maxbins": 18}, "axis": {"labelColor": "#8ea4bc"}}, "y": {"aggregate": "count", "axis": {"title": "Count", "labelColor": "#8ea4bc", "gridColor": "#18304a"}}, "tooltip": [{"aggregate": "count", "type": "quantitative", "title": "Observations"}]},
                "config": {"view": {"stroke": "#1a344e"}},
            }
            st.markdown('<div class="section-title">Residual distribution</div>', unsafe_allow_html=True)
            st.vega_lite_chart(hist_spec, use_container_width=True)

        with st.expander("View validation observations"):
            st.dataframe(pv.round(4), use_container_width=True, hide_index=True)
        st.download_button("⬇ Download validation predictions", preds.to_csv(index=False), "validation_predictions.csv", "text/csv")
    else:
        st.info("Validation predictions file not found. Add models/validation_predictions.csv to enable this panel.")

with drivers_tab:
    st.markdown('<div class="section-title">Index drivers & PCA explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">These values describe mathematical contribution to the composite PCA score; they should not be interpreted as causal effects.</div>', unsafe_allow_html=True)
    raw_latest = bundle.get("latest_index_components")
    components = bundle["idx_components"]
    loading = bundle["pca"].components_[0] * bundle["index_sign"]
    if raw_latest:
        raw = pd.Series(raw_latest).reindex(components).astype(float)
        transformed = raw.copy()
        for c in bundle["negative_components"]:
            transformed[c] = -transformed[c]
        z = pd.Series(bundle["idx_scaler"].transform(pd.DataFrame([transformed], columns=components))[0], index=components)
        contribution = z * pd.Series(loading, index=components)
        driver_df = pd.DataFrame({"Feature": components, "Latest value": raw.values, "Standardized value": z.values, "PCA loading": loading, "Index contribution": contribution.values}).sort_values("Index contribution", key=lambda s: s.abs(), ascending=False)
        st.dataframe(driver_df.round(4), use_container_width=True, hide_index=True)
        bar_df = driver_df.sort_values("Index contribution")
        driver_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json", "width": "container", "height": 300, "background": "transparent",
            "data": {"values": bar_df[["Feature", "Index contribution"]].to_dict("records")},
            "mark": {"type": "bar", "cornerRadiusEnd": 5},
            "encoding": {"y": {"field": "Feature", "type": "nominal", "sort": None, "axis": {"labelColor": "#aabbd0"}}, "x": {"field": "Index contribution", "type": "quantitative", "axis": {"title": "Contribution", "labelColor": "#8ea4bc", "gridColor": "#18304a"}}, "color": {"condition": {"test": "datum['Index contribution'] >= 0", "value": "#29d391"}, "value": "#ff6b7a"}, "tooltip": [{"field": "Feature"}, {"field": "Index contribution", "type": "quantitative", "format": ".4f"}]},
            "config": {"view": {"stroke": "#1a344e"}},
        }
        st.vega_lite_chart(driver_spec, use_container_width=True)
    else:
        weights = pd.DataFrame({"Feature": components, "PCA loading": loading})
        st.dataframe(weights.round(4), use_container_width=True, hide_index=True)
        st.info("Retrain with the latest training script to enable current-value contribution analysis.")

with methodology_tab:
    st.markdown('<div class="section-title">Methodology & model notes</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="panel">
<b>1. Data preparation</b><br>
Five project datasets are aggregated and joined at monthly frequency.<br><br>
<b>2. De-dollarization index</b><br>
Selected reserve-share, oil-invoicing and petroyuan variables are standardized and reduced to one PCA component. Component signs are oriented so that a higher index represents the project's defined de-dollarization direction.<br><br>
<b>3. Feature engineering</b><br>
The forecasting model uses index lags, rolling statistics, trend slope, market/macro variables and event flags.<br><br>
<b>4. Forecast model</b><br>
A linear trend provides the baseline path. A Random Forest predicts the residual adjustment around that trend for 6, 12, 24 and 36 months.<br><br>
<b>5. Validation</b><br>
The historical sample is split chronologically. Transformations and model fitting for validation are based on the historical training window before later observations are evaluated.<br><br>
<b>6. Production refit</b><br>
After validation, the production model is refit on the full available project history so the deployed artifact uses all available observations.
</div>
""",
        unsafe_allow_html=True,
    )
    info = pd.DataFrame({
        "Item": ["Model version", "Data start", "Data end", "Forecast horizons", "Model type"],
        "Value": [bundle.get("version", "N/A"), pd.Timestamp(bundle.get("data_start", history["date"].iloc[0])).strftime("%B %Y"), latest_date.strftime("%B %Y"), ", ".join(map(str, horizons)) + " months", bundle["model_type"]],
    })
    st.dataframe(info, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Data freshness disclaimer — intentionally LAST on the page.
# -----------------------------------------------------------------------------
if months_old >= 1:
    freshness_text = (
        f"Model data ends in <strong>{latest_date:%B %Y}</strong>, approximately <strong>{months_old} months</strong> behind the current date. "
        "Forecasts are scenario outputs based on the available project dataset, not live market forecasts."
    )
else:
    freshness_text = f"Model data currently runs through <strong>{latest_date:%B %Y}</strong>. Forecasts are scenario outputs based on the available project dataset, not live market forecasts."

st.markdown(
    f'<div class="footer-disclaimer">⚠️ <strong>Data freshness & use note:</strong> {freshness_text} The project source data may include synthetic-filled values; verify underlying data with authoritative sources before using results for business, investment, or financial decisions.</div>',
    unsafe_allow_html=True,
)
