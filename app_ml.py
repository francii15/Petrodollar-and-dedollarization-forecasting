import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

MODEL_PATH = 'models/model_bundle.joblib'
METRICS_PATH = 'models/validation_metrics.csv'
PREDICTIONS_PATH = 'models/validation_predictions.csv'

st.set_page_config(
    page_title='Petrodollar AI — De-dollarization Intelligence',
    page_icon='💵',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ---------- Styling ----------
st.markdown('''
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {padding: 0.25rem 0 0.75rem 0;}
.hero h1 {font-size: 2.6rem; margin-bottom: 0.2rem;}
.hero p {font-size: 1rem; color: #9aa0aa; margin-top: 0;}
.section-note {color:#9aa0aa; font-size:0.9rem;}
.small-label {font-size:0.78rem; color:#9aa0aa; text-transform:uppercase; letter-spacing:.06em;}
</style>
''', unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


bundle = load_bundle()

if bundle is None:
    st.error('Trained model artifact not found.')
    st.markdown('Run the training pipeline once:')
    st.code('python train_model.py')
    st.stop()

history = bundle['history'].copy()
history['date'] = pd.to_datetime(history['date'])
latest_date = pd.Timestamp(bundle['latest_date'])
latest_index = float(bundle['latest_index'])
feature_cols = bundle['feature_cols']
features = pd.DataFrame([bundle['latest_features']])[feature_cols]
slope = float(bundle['latest_features']['idx_trend_slope_12'])
horizons = bundle['horizons']


def forecast_for_horizon(horizon):
    model = bundle['models'][horizon]
    trend_component = latest_index + slope * horizon
    residual = float(model.predict(features)[0])
    forecast = trend_component + residual
    unc = float(bundle.get('residual_std', {}).get(horizon, 0.0))
    future_date = latest_date + pd.DateOffset(months=horizon)
    return forecast, trend_component, residual, unc, future_date


forecasts = {h: forecast_for_horizon(h) for h in horizons}
selected_horizon = st.sidebar.selectbox('Forecast horizon (months)', horizons, index=0)
forecast, trend_component, residual, unc, future_date = forecasts[selected_horizon]

# ---------- Data freshness ----------
today = pd.Timestamp.today().normalize()
months_old = (today.to_period('M') - latest_date.to_period('M')).n

# ---------- Derived dashboard signals ----------
history_sorted = history.sort_values('date').reset_index(drop=True)
recent_12 = history_sorted.tail(12)['dedollarization_index']
prev_12 = history_sorted.iloc[-24:-12]['dedollarization_index'] if len(history_sorted) >= 24 else pd.Series(dtype=float)
change_12m = float(recent_12.iloc[-1] - recent_12.iloc[0]) if len(recent_12) >= 2 else np.nan
median_index = float(history_sorted['dedollarization_index'].median())
percentile = float((history_sorted['dedollarization_index'] <= latest_index).mean() * 100)

# ---------- Header ----------
st.markdown('''
<div class="hero">
<h1>💵 Petrodollar AI — De-dollarization Intelligence</h1>
<p>PCA-based composite index • Trend + Random Forest residual model • 6/12/24/36-month scenario forecasting</p>
</div>
''', unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.markdown('### Forecast controls')
st.sidebar.caption('Choose the scenario horizon used for the detailed forecast below.')

if months_old >= 6:
    st.warning(
        f'**Data freshness:** model data ends in **{latest_date:%B %Y}**, about **{months_old} months** behind the current date. '
        'The forecasts are scenario outputs based on the available project dataset, not live market forecasts.'
    )
else:
    st.success(f'Data through **{latest_date:%B %Y}** is loaded.')

# ---------- Executive dashboard ----------
st.subheader('Executive dashboard')
cols = st.columns(6)
cols[0].metric('Latest index', f'{latest_index:.2f}')
cols[1].metric('12M slope / month', f'{slope:+.4f}')
for i, h in enumerate(horizons, start=2):
    value = forecasts[h][0]
    cols[i].metric(f'{h}M scenario', f'{value:.2f}', delta=f'{value - latest_index:+.2f}')

st.caption('Scenario values are model outputs. The composite index is an analytical measure and is not a market price or causal estimate.')

# ---------- Main tabs ----------
executive_tab, forecast_tab, validation_tab, drivers_tab, methodology_tab = st.tabs([
    '🏠 Executive view', '🔮 Forecast', '📊 Validation', '🧠 Drivers', '📘 Methodology'
])

with executive_tab:
    st.subheader('Current index snapshot')
    a, b, c, d = st.columns(4)
    a.metric('12-month index change', f'{change_12m:+.2f}' if pd.notna(change_12m) else 'N/A')
    b.metric('Historical median', f'{median_index:.2f}')
    c.metric('Current historical percentile', f'{percentile:.0f}%')
    d.metric('Latest data point', latest_date.strftime('%b %Y'))

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.plot(history_sorted['date'], history_sorted['dedollarization_index'], label='Historical index', linewidth=2)
    ax.axhline(median_index, linestyle=':', linewidth=1, label='Historical median')
    ax.axvline(latest_date, linestyle='--', linewidth=1, label='Latest data')
    ax.set_title('De-dollarization composite index — historical path')
    ax.set_xlabel('Date')
    ax.set_ylabel('Composite index')
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.subheader('Scenario range')
    scenario_df = pd.DataFrame([
        {'Horizon': f'{h}M', 'Scenario': forecasts[h][0], 'Change vs latest': forecasts[h][0] - latest_index,
         'Residual uncertainty': forecasts[h][3]}
        for h in horizons
    ])
    st.dataframe(scenario_df.round(3), use_container_width=True, hide_index=True)

with forecast_tab:
    st.subheader(f'{selected_horizon}-month scenario forecast')
    f1, f2, f3, f4 = st.columns(4)
    f1.metric('Scenario value', f'{forecast:.2f}')
    f2.metric('Change vs latest', f'{forecast - latest_index:+.2f}')
    f3.metric('Trend component', f'{trend_component:.2f}')
    f4.metric('ML residual', f'{residual:+.2f}')

    # The model produces an endpoint forecast. The connecting scenario line is explicitly illustrative.
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(history_sorted['date'], history_sorted['dedollarization_index'], label='Historical index', linewidth=2)
    trend_dates = pd.date_range(latest_date, future_date, periods=30)
    trend_path = np.linspace(latest_index, trend_component, len(trend_dates))
    scenario_path = np.linspace(latest_index, forecast, len(trend_dates))
    ax.plot(trend_dates, trend_path, '--', label='Trend baseline')
    ax.plot(trend_dates, scenario_path, ':', linewidth=2, label='Scenario bridge (illustrative)')
    ax.scatter([future_date], [forecast], s=100, label='ML scenario')
    if unc > 0:
        ax.errorbar([future_date], [forecast], yerr=[unc], fmt='none', capsize=7, label='Residual uncertainty')
    ax.axvline(latest_date, linestyle=':', linewidth=1, label='Forecast origin')
    ax.set_title(f'{selected_horizon}-month forecast endpoint')
    ax.set_xlabel('Date')
    ax.set_ylabel('De-dollarization index')
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    d1, d2 = st.columns(2)
    with d1:
        st.subheader('Forecast decomposition')
        st.write(f'**Forecast date:** {future_date:%B %Y}')
        st.write(f'**Trend baseline:** {trend_component:.2f}')
        st.write(f'**RF residual adjustment:** {residual:+.2f}')
        st.write(f'**Residual uncertainty:** ±{unc:.2f}')
    with d2:
        st.subheader('Model status')
        st.write(f"**Model:** {bundle['model_type']}")
        st.write(f"**Training rows:** {bundle['training_rows']:,}")
        st.write(f"**Training/data window:** {pd.Timestamp(bundle.get('data_start', history_sorted['date'].iloc[0])):%b %Y} → {latest_date:%b %Y}")
        st.write(f"**Validation cutoff:** {pd.Timestamp(bundle['validation_cutoff']):%b %Y}")
        st.write(f"**PCA variance explained:** {bundle['pca_explained_variance']*100:.1f}%")

with validation_tab:
    st.subheader('Out-of-sample validation')
    st.caption('Chronological validation: later observations are evaluated after the historical training window. Metrics describe historical test performance, not guaranteed future accuracy.')

    if os.path.exists(METRICS_PATH):
        metrics = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)

    if os.path.exists(PREDICTIONS_PATH):
        preds = pd.read_csv(PREDICTIONS_PATH)
        preds['Forecast Date'] = pd.to_datetime(preds['Forecast Date'])
        plot_h = st.selectbox('Validation horizon', horizons, index=0, key='validation_horizon')
        pv = preds[preds['Horizon (months)'] == plot_h].sort_values('Forecast Date')
        fig2, ax2 = plt.subplots(figsize=(14, 5.2))
        ax2.plot(pv['Forecast Date'], pv['Actual Index'], label='Actual index')
        ax2.plot(pv['Forecast Date'], pv['Forecast Index'], '--', label='Model forecast')
        ax2.set_title(f'{plot_h}-month out-of-sample validation')
        ax2.set_xlabel('Forecast date')
        ax2.set_ylabel('De-dollarization index')
        ax2.grid(True, alpha=0.25)
        ax2.legend()
        st.pyplot(fig2, clear_figure=True)
        st.download_button('Download validation predictions', preds.to_csv(index=False), 'validation_predictions.csv', 'text/csv')

with drivers_tab:
    st.subheader('What mathematically contributes to the index?')
    st.caption("These are PCA contributions, not causal effects. Positive/negative direction refers to the project's index construction.")

    # Prefer stored latest component values from the retrained bundle.
    raw_latest = bundle.get('latest_index_components')
    components = bundle['idx_components']
    loading = bundle['pca'].components_[0] * bundle['index_sign']

    if raw_latest:
        raw = pd.Series(raw_latest).reindex(components).astype(float)
        transformed = raw.copy()
        for c in bundle['negative_components']:
            transformed[c] = -transformed[c]
        z = pd.Series(bundle['idx_scaler'].transform(pd.DataFrame([transformed], columns=components))[0], index=components)
        contribution = z * pd.Series(loading, index=components)
        driver_df = pd.DataFrame({
            'Feature': components,
            'Latest value': raw.values,
            'Standardized value': z.values,
            'PCA loading': loading,
            'Index contribution': contribution.values,
        }).sort_values('Index contribution', key=lambda s: s.abs(), ascending=False)
        st.dataframe(driver_df.round(4), use_container_width=True, hide_index=True)

        fig3, ax3 = plt.subplots(figsize=(12, 5))
        plot_df = driver_df.sort_values('Index contribution')
        ax3.barh(plot_df['Feature'], plot_df['Index contribution'])
        ax3.axvline(0, linewidth=1)
        ax3.set_xlabel('Contribution to current PCA score')
        ax3.set_title('Current index contribution by component')
        ax3.grid(True, axis='x', alpha=0.25)
        st.pyplot(fig3, clear_figure=True)
    else:
        weights = pd.DataFrame({'Feature': components, 'PCA loading': loading})
        st.dataframe(weights.round(4), use_container_width=True, hide_index=True)
        st.info('Retrain with the latest training script to enable current-value contribution analysis.')

    st.subheader('PCA loadings')
    weights = pd.DataFrame({'Feature': components, 'Weight': loading})
    weights['Direction in index'] = np.where(weights['Weight'] >= 0, 'Positive', 'Negative')
    st.dataframe(weights.round(4), use_container_width=True, hide_index=True)

with methodology_tab:
    st.subheader('How the model works')
    st.markdown('''
**1. Data preparation**  
Five project datasets are aggregated and joined at monthly frequency.

**2. De-dollarization index**  
Selected reserve-share, oil-invoicing and petroyuan variables are standardized and reduced to one PCA component. Component signs are oriented so that a higher index represents the project's defined de-dollarization direction.

**3. Feature engineering**  
The forecasting model uses index lags, rolling statistics, trend slope, market/macro variables and event flags.

**4. Forecast model**  
A linear trend provides the baseline path. A Random Forest predicts the residual adjustment around that trend for 6, 12, 24 and 36 months.

**5. Validation**  
The historical sample is split chronologically. Transformations and model fitting for validation are based on the historical training window before later observations are evaluated.

**6. Production refit**  
After validation, the production model is refit on the full available project history so the deployed artifact uses all available observations.
''')
    st.write(f"**Model version:** {bundle.get('version', 'N/A')}")
    st.write(f"**Data start:** {pd.Timestamp(bundle.get('data_start', history_sorted['date'].iloc[0])):%B %Y}")
    st.write(f"**Data end:** {latest_date:%B %Y}")
    st.write(f"**Forecast horizons:** {', '.join(map(str, horizons))} months")

# ---------- Footer ----------
st.divider()
st.caption(
    '⚠ Model-based scenario analysis only. The project source data may include synthetic-filled values. '
    'Verify underlying data with authoritative sources before using the results for business, investment or financial decisions.'
)
