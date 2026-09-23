
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Petrodollar & De-dollarization",
    page_icon="💵",
    layout="wide"
)

st.title("💵 Petrodollar & De-dollarization Analysis")
st.caption("EDA, a PCA-based de-dollarization index, and multi-horizon forecasting")


FILES = {
    "oil": "petrodollar_1_oil_production_trade.csv",
    "opec": "petrodollar_2_opec_quotas_events.csv",
    "swf": "petrodollar_3_recycling_swf.csv",
    "brics": "petrodollar_4_dedollarization.csv",
    "prices": "petrodollar_5_daily_prices_fx.csv",
}


@st.cache_data
def load_data():
    data = {}
    missing = []

    for key, filename in FILES.items():
        path = "data/" + filename
        try:
            data[key] = pd.read_csv(path)
        except FileNotFoundError:
            missing.append(filename)

    if missing:
        return None, missing

    for key in ["oil", "opec", "brics", "prices"]:
        data[key]["date"] = pd.to_datetime(data[key]["date"], errors="coerce")

    data["brics"]["key_event"] = data["brics"]["key_event"].fillna("No_Event")
    data["opec"]["key_event"] = data["opec"]["key_event"].fillna("No_Event")

    return data, []


def prepare_data(data):
    oil = data["oil"].copy()
    opec = data["opec"].copy()
    swf = data["swf"].copy()
    brics = data["brics"].copy()
    prices = data["prices"].copy()

    oil_monthly = (
        oil.groupby(["year", "month"], as_index=False)
        .agg({
            "production_mbd": "sum",
            "exports_mbd": "sum",
            "world_demand_mbd": "mean",
            "oil_revenue_usd_bn": "sum",
        })
    )

    prices_monthly = (
        prices.drop(columns=["weekday", "data_source"], errors="ignore")
        .groupby(["year", "month"], as_index=False)
        .mean(numeric_only=True)
    )

    df = prices_monthly.merge(
        oil_monthly, on=["year", "month"], how="inner"
    )

    df = df.merge(
        opec.drop(
            columns=["date", "data_source", "brent_price_usd"],
            errors="ignore"
        ),
        on=["year", "month"],
        how="left"
    )

    df = df.merge(
        brics.drop(
            columns=["date", "data_source", "brent_price_usd"],
            errors="ignore"
        ),
        on=["year", "month"],
        how="left",
        suffixes=("_opec", "_brics")
    )

    df = df.merge(
        swf.drop(
            columns=["data_source", "brent_avg_usd"],
            errors="ignore"
        ),
        on="year",
        how="left"
    )

    df = df[df["year"] >= 2005].reset_index(drop=True)

    # Derived features from the original notebook
    df["spare_capacity"] = (
        df["opec_quota_mbd"] - df["opec_actual_production_mbd"]
    )

    df["non_usd_trade_ratio"] = (
        df["oil_trade_non_usd_pct"] /
        df["oil_trade_usd_invoicing_pct"]
    )

    df["usd_cny_gap"] = (
        df["usd_global_fx_reserves_pct"] -
        df["cny_global_fx_reserves_pct"]
    )

    df["petroyuan_change"] = (
        df["ine_petroyuan_volume_klots"].diff().fillna(0)
    )

    df["opec_event_flag"] = (
        df["key_event_opec"] != "No_Event"
    ).astype(int)

    df["brics_event_flag"] = (
        df["key_event_brics"] != "No_Event"
    ).astype(int)

    df = df.sort_values(["year", "month"]).reset_index(drop=True)

    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" +
        df["month"].astype(str) + "-01"
    )

    # Construct the PCA index
    positive_components = [
        "cny_global_fx_reserves_pct",
        "oil_trade_non_usd_pct",
        "ine_petroyuan_volume_klots",
    ]

    negative_components = [
        "usd_global_fx_reserves_pct",
        "oil_trade_usd_invoicing_pct",
    ]

    idx_components = positive_components + negative_components

    comp_df = df[idx_components].copy()

    for col in negative_components:
        comp_df[col] = -comp_df[col]

    idx_scaler = StandardScaler()
    comp_scaled = idx_scaler.fit_transform(comp_df)

    pca = PCA(n_components=1, random_state=42)
    df["dedollarization_index"] = (
        pca.fit_transform(comp_scaled).flatten()
    )

    IDX = "dedollarization_index"

    if df[IDX].tail(24).mean() < df[IDX].head(24).mean():
        df[IDX] = -df[IDX]

    # Lag and rolling features
    df["idx_lag1"] = df[IDX].shift(1)
    df["idx_lag3"] = df[IDX].shift(3)
    df["idx_lag6"] = df[IDX].shift(6)
    df["idx_lag12"] = df[IDX].shift(12)

    df["idx_rolling_mean_6"] = (
        df[IDX].shift(1).rolling(6).mean()
    )
    df["idx_rolling_mean_12"] = (
        df[IDX].shift(1).rolling(12).mean()
    )
    df["idx_rolling_std_6"] = (
        df[IDX].shift(1).rolling(6).std()
    )
    df["idx_trend_slope_12"] = (
        df[IDX].diff().shift(1).rolling(12).mean()
    )

    macro_features = [
        "brent_crude_usd",
        "usd_dxy_index",
        "opec_compliance_pct",
        "spare_capacity",
        "global_swf_total_est_usd_bn",
        "total_opec_oil_revenue_usd_bn",
        "non_usd_trade_ratio",
        "usd_cny_gap",
        "petroyuan_change",
        "opec_event_flag",
        "brics_event_flag",
    ]

    feature_cols = [
        "idx_lag1",
        "idx_lag3",
        "idx_lag6",
        "idx_lag12",
        "idx_rolling_mean_6",
        "idx_rolling_mean_12",
        "idx_rolling_std_6",
        "idx_trend_slope_12",
    ] + macro_features

    # Match the notebook's final null-free modeling dataset
    df[feature_cols] = df[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    return df, pca, idx_components, feature_cols


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


@st.cache_data
def run_forecasts(df, feature_cols):
    IDX = "dedollarization_index"
    all_results = []
    artifacts = {}

    for horizon in [6, 12, 24, 36]:
        d = df.copy()
        d["Target"] = d[IDX].shift(-horizon)

        d = d.dropna(
            subset=["Target"] + feature_cols
        ).reset_index(drop=True)

        X = d[feature_cols]
        y = d["Target"]

        split = int(len(d) * 0.8)

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        naive_pred = d[IDX].iloc[split:]

        trend_train = (
            d[IDX].iloc[:split] +
            d["idx_trend_slope_12"].iloc[:split] * horizon
        )

        trend_test = (
            d[IDX].iloc[split:] +
            d["idx_trend_slope_12"].iloc[split:] * horizon
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ridge = Ridge(alpha=10, random_state=42)
        ridge.fit(X_train_scaled, y_train)
        pred_ridge = ridge.predict(X_test_scaled)

        resid_train = y_train - trend_train

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=5,
            random_state=42,
        )

        rf.fit(X_train, resid_train)
        resid_pred = rf.predict(X_test)

        pred_rf = trend_test + resid_pred

        models = [
            ("Naive persistence", naive_pred),
            ("Trend extrapolation", trend_test),
            ("Ridge (direct)", pred_ridge),
            ("Trend + RF residual", pred_rf),
        ]

        for name, pred in models:
            metrics = evaluate(y_test, pred)
            all_results.append({
                "Horizon (months)": horizon,
                "Model": name,
                "Test months": len(y_test),
                **metrics,
            })

        artifacts[horizon] = {
            "d": d,
            "split": split,
            "X_test": X_test,
            "y_test": y_test,
            "rf": rf,
            "ridge": ridge,
            "trend_pred_test": trend_test,
            "pred_rf_trend": pred_rf,
        }

    return pd.DataFrame(all_results), artifacts


data, missing = load_data()

if data is None:
    st.error("The five required CSV files were not found.")
    st.markdown("### Put the datasets in this folder:")
    st.code("data/")
    for filename in missing:
        st.write("•", filename)
    st.info(
        "Download the five CSV files used by the notebook and place them "
        "inside the data folder with exactly these filenames."
    )
    st.stop()

try:
    df, pca, idx_components, feature_cols = prepare_data(data)
except Exception as e:
    st.error("The dataset structure does not match the notebook.")
    st.exception(e)
    st.stop()

summary, artifacts = run_forecasts(df, feature_cols)

# Sidebar
st.sidebar.header("Dashboard")
page = st.sidebar.radio(
    "Select section",
    ["Overview", "EDA", "De-dollarization Index", "Forecasting", "Feature Importance"]
)

st.sidebar.markdown("---")
st.sidebar.write(f"Monthly observations: **{len(df):,}**")
st.sidebar.write(
    f"Period: **{df['date'].min():%Y-%m} → {df['date'].max():%Y-%m}**"
)


if page == "Overview":
    st.header("Project Overview")

    latest = df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Latest Index",
        f"{latest['dedollarization_index']:.2f}"
    )
    c2.metric(
        "USD FX Reserve Share",
        f"{latest['usd_global_fx_reserves_pct']:.1f}%"
    )
    c3.metric(
        "CNY FX Reserve Share",
        f"{latest['cny_global_fx_reserves_pct']:.1f}%"
    )
    c4.metric(
        "Non-USD Oil Trade",
        f"{latest['oil_trade_non_usd_pct']:.1f}%"
    )

    st.subheader("De-dollarization Index")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["dedollarization_index"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Index")
    ax.set_title("PCA-based De-dollarization Index")
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Latest Data")
    st.dataframe(
        df.tail(12).sort_values("date", ascending=False),
        use_container_width=True
    )


elif page == "EDA":
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("USD vs CNY Share of Global FX Reserves")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            df["date"],
            df["usd_global_fx_reserves_pct"],
            label="USD"
        )
        ax.plot(
            df["date"],
            df["cny_global_fx_reserves_pct"],
            label="CNY"
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Reserve Share (%)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig, clear_figure=True)

    with col2:
        st.subheader("Oil Trade Invoicing")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.stackplot(
            df["date"],
            df["oil_trade_usd_invoicing_pct"],
            df["oil_trade_non_usd_pct"],
            labels=["USD", "Non-USD"]
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Percentage")
        ax.legend(loc="upper right")
        st.pyplot(fig, clear_figure=True)

    st.subheader("Petroyuan Futures Trading Volume")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["ine_petroyuan_volume_klots"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Volume")
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Brent Crude Price vs Dollar Index")
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(
        df["date"],
        df["brent_crude_usd"],
        label="Brent Price"
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Brent Price")

    ax2 = ax1.twinx()
    ax2.plot(
        df["date"],
        df["usd_dxy_index"],
        label="DXY"
    )
    ax2.set_ylabel("Dollar Index")

    ax1.grid(True)
    st.pyplot(fig, clear_figure=True)


elif page == "De-dollarization Index":
    st.header("PCA De-dollarization Index")

    st.write(
        "The index combines positive de-dollarization components "
        "(CNY reserves, non-USD oil invoicing, and petroyuan volume) "
        "with negative components (USD reserves and USD oil invoicing), "
        "then applies StandardScaler + PCA."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Explained Variance",
            f"{pca.explained_variance_ratio_[0] * 100:.2f}%"
        )

    with c2:
        if "de_dollarization_pressure_index" in df.columns:
            corr = df["dedollarization_index"].corr(
                df["de_dollarization_pressure_index"]
            )
            st.metric("Correlation with Dataset Index", f"{corr:.3f}")

    st.subheader("PCA Feature Weights")

    weights = pd.DataFrame({
        "Feature": idx_components,
        "Weight": pca.components_[0],
    }).sort_values("Weight", ascending=False)

    st.dataframe(weights, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        df["date"],
        df["dedollarization_index"],
        label="Our Index"
    )

    if "de_dollarization_pressure_index" in df.columns:
        ax.plot(
            df["date"],
            df["de_dollarization_pressure_index"],
            label="Dataset Index"
        )

    ax.set_title("De-dollarization Index Comparison")
    ax.set_xlabel("Year")
    ax.set_ylabel("Index")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)


elif page == "Forecasting":
    st.header("Forecasting")

    st.write(
        "The forecasting workflow evaluates 6-, 12-, 24-, and 36-month "
        "horizons using the same chronological 80/20 split as the notebook."
    )

    st.subheader("Model Comparison")
    st.dataframe(
        summary.round(4),
        use_container_width=True
    )

    horizon = st.selectbox(
        "Forecast horizon",
        [6, 12, 24, 36],
        index=2
    )

    art = artifacts[horizon]
    d = art["d"]
    split = art["split"]

    dates_test = d["date"].iloc[split:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        dates_test,
        art["y_test"],
        marker="o",
        ms=3,
        label="Actual"
    )
    ax.plot(
        dates_test,
        art["trend_pred_test"],
        linestyle="--",
        label="Trend extrapolation"
    )
    ax.plot(
        dates_test,
        art["pred_rf_trend"],
        marker="o",
        ms=3,
        label="Trend + RF residual"
    )
    ax.set_title(
        f"{horizon}-month-ahead forecast vs actual"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("De-dollarization Index")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Metrics for Selected Horizon")
    st.dataframe(
        summary[summary["Horizon (months)"] == horizon].round(4),
        use_container_width=True
    )


elif page == "Feature Importance":
    st.header("Random Forest Feature Importance")

    horizon = 24
    art = artifacts[horizon]

    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": art["rf"].feature_importances_,
    }).sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)

    st.subheader("Top 10 Features — 24-month Horizon")
    st.dataframe(
        importances.head(10).round(6),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    top = importances.head(10).sort_values("Importance")
    ax.barh(top["Feature"], top["Importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Random Forest Features")
    st.pyplot(fig, clear_figure=True)


st.markdown("---")
st.caption(
    "Built from the supplied Petrodollar & De-dollarization notebook. "
    "The dashboard reproduces its preprocessing, PCA index construction, "
    "and forecasting workflow."
)
