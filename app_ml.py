import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

MODEL_PATH = 'models/model_bundle.joblib'
METRICS_PATH = 'models/validation_metrics.csv'

st.set_page_config(page_title='Petrodollar AI', page_icon='💵', layout='wide')

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

bundle = load_bundle()
st.title('💵 Petrodollar AI — De-dollarization Intelligence')
st.caption('Trained ML inference • PCA index • 6/12/24/36-month forecasting')

if bundle is None:
    st.error('Trained model artifact not found.')
    st.markdown('Run the training pipeline once:')
    st.code('python train_model.py')
    st.stop()

history = bundle['history'].copy()
history['date'] = pd.to_datetime(history['date'])
horizon = st.sidebar.selectbox('Forecast horizon (months)', bundle['horizons'], index=2)
model = bundle['models'][horizon]
latest_index = float(bundle['latest_index'])
features = pd.DataFrame([bundle['latest_features']])[bundle['feature_cols']]
slope = float(bundle['latest_features']['idx_trend_slope_12'])
trend_component = latest_index + slope * horizon
residual = float(model.predict(features)[0])
forecast = trend_component + residual
unc = float(bundle.get('residual_std', {}).get(horizon, 0.0))
future_date = pd.Timestamp(bundle['latest_date']) + pd.DateOffset(months=horizon)

c1, c2, c3, c4 = st.columns(4)
c1.metric('Latest index', f'{latest_index:.2f}')
c2.metric('Trend / month', f'{slope:.4f}')
c3.metric(f'{horizon}-month forecast', f'{forecast:.2f}')
c4.metric('PCA variance explained', f"{bundle['pca_explained_variance']*100:.1f}%")

st.subheader('Forecast path')
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(history['date'], history['dedollarization_index'], label='Historical index')
ax.plot([history['date'].iloc[-1], future_date], [latest_index, trend_component], '--', label='Trend component')
ax.scatter([future_date], [forecast], s=80, label='ML forecast')
if unc > 0:
    ax.errorbar([future_date], [forecast], yerr=[unc], fmt='none', capsize=6, label='Residual uncertainty')
ax.set_xlabel('Date')
ax.set_ylabel('De-dollarization index')
ax.grid(True)
ax.legend()
st.pyplot(fig, clear_figure=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader('Forecast decomposition')
    st.metric('Trend component', f'{trend_component:.2f}')
    st.metric('ML residual adjustment', f'{residual:+.2f}')
    st.metric('Forecast date', future_date.strftime('%B %Y'))
with c2:
    st.subheader('Model information')
    st.write(f"**Model:** {bundle['model_type']}")
    st.write(f"**Training rows:** {bundle['training_rows']:,}")
    st.write(f"**Training through:** {pd.Timestamp(bundle['latest_date']):%Y-%m}")
    st.write(f"**Validation cutoff:** {pd.Timestamp(bundle['validation_cutoff']):%Y-%m}")

st.subheader('Validation performance')
if os.path.exists(METRICS_PATH):
    metrics = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)

st.subheader('PCA feature weights')
weights = pd.DataFrame({
    'Feature': bundle['idx_components'],
    'Weight': bundle['pca'].components_[0] * bundle['index_sign']
})
st.dataframe(weights.round(4), use_container_width=True, hide_index=True)

st.info('This is a model-based scenario forecast, not a guarantee. The project source data includes synthetic-filled values; verified official data should be used before business/financial decisions.')
