import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = os.getenv('DATA_DIR', 'data')
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    'oil': 'petrodollar_1_oil_production_trade.csv',
    'opec': 'petrodollar_2_opec_quotas_events.csv',
    'swf': 'petrodollar_3_recycling_swf.csv',
    'brics': 'petrodollar_4_dedollarization.csv',
    'prices': 'petrodollar_5_daily_prices_fx.csv',
}
HORIZONS = [6, 12, 24, 36]
POSITIVE_COMPONENTS = ['cny_global_fx_reserves_pct','oil_trade_non_usd_pct','ine_petroyuan_volume_klots']
NEGATIVE_COMPONENTS = ['usd_global_fx_reserves_pct','oil_trade_usd_invoicing_pct']
MACRO_FEATURES = ['brent_crude_usd','usd_dxy_index','opec_compliance_pct','spare_capacity','global_swf_total_est_usd_bn','total_opec_oil_revenue_usd_bn','non_usd_trade_ratio','usd_cny_gap','petroyuan_change','opec_event_flag','brics_event_flag']
FEATURE_COLS = ['idx_lag1','idx_lag3','idx_lag6','idx_lag12','idx_rolling_mean_6','idx_rolling_mean_12','idx_rolling_std_6','idx_trend_slope_12'] + MACRO_FEATURES


def load_raw():
    data, missing = {}, []
    for key, fn in FILES.items():
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path): missing.append(fn)
        else: data[key] = pd.read_csv(path)
    if missing:
        raise FileNotFoundError('Missing files: ' + ', '.join(missing))
    for key in ['oil','opec','brics','prices']:
        data[key]['date'] = pd.to_datetime(data[key]['date'], errors='coerce')
    data['brics']['key_event'] = data['brics']['key_event'].fillna('No_Event')
    data['opec']['key_event'] = data['opec']['key_event'].fillna('No_Event')
    return data


def build_base(data):
    oil, opec, swf = data['oil'].copy(), data['opec'].copy(), data['swf'].copy()
    brics, prices = data['brics'].copy(), data['prices'].copy()
    oil_monthly = oil.groupby(['year','month'], as_index=False).agg({'production_mbd':'sum','exports_mbd':'sum','world_demand_mbd':'mean','oil_revenue_usd_bn':'sum'})
    prices_monthly = prices.drop(columns=['weekday','data_source'], errors='ignore').groupby(['year','month'], as_index=False).mean(numeric_only=True)
    df = prices_monthly.merge(oil_monthly, on=['year','month'], how='inner')
    df = df.merge(opec.drop(columns=['date','data_source','brent_price_usd'], errors='ignore'), on=['year','month'], how='left')
    df = df.merge(brics.drop(columns=['date','data_source','brent_price_usd'], errors='ignore'), on=['year','month'], how='left', suffixes=('_opec','_brics'))
    df = df.merge(swf.drop(columns=['data_source','brent_avg_usd'], errors='ignore'), on='year', how='left')
    df = df[df['year'] >= 2005].sort_values(['year','month']).reset_index(drop=True)
    df['spare_capacity'] = df['opec_quota_mbd'] - df['opec_actual_production_mbd']
    df['non_usd_trade_ratio'] = df['oil_trade_non_usd_pct'] / df['oil_trade_usd_invoicing_pct']
    df['usd_cny_gap'] = df['usd_global_fx_reserves_pct'] - df['cny_global_fx_reserves_pct']
    df['petroyuan_change'] = df['ine_petroyuan_volume_klots'].diff().fillna(0)
    df['opec_event_flag'] = (df['key_event_opec'] != 'No_Event').astype(int)
    df['brics_event_flag'] = (df['key_event_brics'] != 'No_Event').astype(int)
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
    return df.reset_index(drop=True)


def fit_index_transform(df_train):
    components = POSITIVE_COMPONENTS + NEGATIVE_COMPONENTS
    comp = df_train[components].copy()
    for c in NEGATIVE_COMPONENTS: comp[c] = -comp[c]
    scaler = StandardScaler().fit(comp)
    pca = PCA(n_components=1, random_state=42).fit(scaler.transform(comp))
    raw = pca.transform(scaler.transform(comp)).ravel()
    sign = 1.0 if raw[-24:].mean() >= raw[:24].mean() else -1.0
    return scaler, pca, sign


def transform_features(df, scaler, pca, sign):
    out = df.copy()
    components = POSITIVE_COMPONENTS + NEGATIVE_COMPONENTS
    comp = out[components].copy()
    for c in NEGATIVE_COMPONENTS: comp[c] = -comp[c]
    out['dedollarization_index'] = pca.transform(scaler.transform(comp)).ravel() * sign
    out['idx_lag1'] = out['dedollarization_index'].shift(1)
    out['idx_lag3'] = out['dedollarization_index'].shift(3)
    out['idx_lag6'] = out['dedollarization_index'].shift(6)
    out['idx_lag12'] = out['dedollarization_index'].shift(12)
    out['idx_rolling_mean_6'] = out['dedollarization_index'].shift(1).rolling(6).mean()
    out['idx_rolling_mean_12'] = out['dedollarization_index'].shift(1).rolling(12).mean()
    out['idx_rolling_std_6'] = out['dedollarization_index'].shift(1).rolling(6).std()
    out['idx_trend_slope_12'] = out['dedollarization_index'].diff().shift(1).rolling(12).mean()
    out[FEATURE_COLS] = out[FEATURE_COLS].replace([np.inf,-np.inf], np.nan)
    return out


def metrics(y, p):
    return {'MAE': float(mean_absolute_error(y,p)), 'RMSE': float(np.sqrt(mean_squared_error(y,p))), 'R2': float(r2_score(y,p))}


def train_for_horizon(df, train_cutoff, horizon):
    # Fit PCA using only the historical training window.
    train_rows = df['date'] <= train_cutoff
    idx_scaler, pca, sign = fit_index_transform(df.loc[train_rows])
    transformed = transform_features(df, idx_scaler, pca, sign)
    d = transformed.copy()
    d['Target'] = d['dedollarization_index'].shift(-horizon)
    d['TargetDate'] = d['date'].shift(-horizon)
    d = d.dropna(subset=['Target'] + FEATURE_COLS).reset_index(drop=True)
    train = d[d['TargetDate'] <= train_cutoff]
    test = d[d['TargetDate'] > train_cutoff]
    Xtr, ytr = train[FEATURE_COLS], train['Target']
    Xte, yte = test[FEATURE_COLS], test['Target']
    trend_tr = train['dedollarization_index'] + train['idx_trend_slope_12'] * horizon
    trend_te = test['dedollarization_index'] + test['idx_trend_slope_12'] * horizon
    rf = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr - trend_tr)
    pred = trend_te + rf.predict(Xte)
    return transformed, d, train, test, rf, metrics(yte,pred), (idx_scaler,pca,sign)


def main():
    raw = load_raw()
    base = build_base(raw)
    # One fixed chronological cutoff for all horizons.
    cutoff = base['date'].iloc[int(len(base)*0.80)-1]
    results = []
    validation = {}
    for h in HORIZONS:
        transformed, d, train, test, rf, m, transformers = train_for_horizon(base, cutoff, h)
        results.append({'Horizon (months)':h,'Model':'Trend + RF residual','Test months':len(test),**m})
        validation[h] = {'metrics':m,'test_months':len(test)}
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(os.path.join(MODEL_DIR,'validation_metrics.csv'), index=False)

    # Production refit: now that evaluation is complete, use all available history.
    scaler, pca, sign = fit_index_transform(base)
    full = transform_features(base, scaler, pca, sign)
    models = {}
    residual_std = {}
    for h in HORIZONS:
        d = full.copy()
        d['Target'] = d['dedollarization_index'].shift(-h)
        d = d.dropna(subset=['Target'] + FEATURE_COLS).reset_index(drop=True)
        trend = d['dedollarization_index'] + d['idx_trend_slope_12'] * h
        rf = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(d[FEATURE_COLS], d['Target'] - trend)
        models[h] = rf
        residual_std[h] = float(np.std((d['Target'] - trend) - rf.predict(d[FEATURE_COLS])))

    bundle = {
        'version':'2.0', 'model_type':'Trend + Random Forest residual', 'horizons':HORIZONS,
        'feature_cols':FEATURE_COLS, 'idx_components':POSITIVE_COMPONENTS+NEGATIVE_COMPONENTS,
        'positive_components':POSITIVE_COMPONENTS, 'negative_components':NEGATIVE_COMPONENTS,
        'idx_scaler':scaler, 'pca':pca, 'index_sign':sign, 'models':models,
        'latest_date':full['date'].iloc[-1], 'latest_index':float(full['dedollarization_index'].iloc[-1]),
        'latest_features':full[FEATURE_COLS].iloc[-1].to_dict(), 'history':full[['date','dedollarization_index']].copy(),
        'pca_explained_variance':float(pca.explained_variance_ratio_[0]), 'training_rows':int(len(full)),
        'validation_cutoff':cutoff, 'validation':validation, 'residual_std':residual_std,
        'source_note':'Project source data may contain synthetic-filled values; replace with verified official data before production/business use.'
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR,'model_bundle.joblib'), compress=3)
    print(metrics_df.to_string(index=False))
    print('Saved model bundle to', os.path.join(MODEL_DIR,'model_bundle.joblib'))

if __name__ == '__main__':
    main()
