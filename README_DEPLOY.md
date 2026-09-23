# Petrodollar & De-dollarization — Deployable ML App

This package converts the supplied notebook workflow into a persisted ML model + Streamlit inference application.

## What changed

The original `app.py` retrains the PCA/scaler/Ridge/Random Forest objects every time the Streamlit application runs. That is useful for exploration, but it is not the same as deploying a trained model.

This deployment version separates:

1. **Training** — `train_model.py`
2. **Model artifact** — `models/model_bundle.joblib`
3. **Inference/UI** — `app_ml.py`

The selected production model is **Trend + Random Forest residual**, matching the notebook's model-comparison workflow.

## Required data

Put these five files under `data/`:

- `petrodollar_1_oil_production_trade.csv`
- `petrodollar_2_opec_quotas_events.csv`
- `petrodollar_3_recycling_swf.csv`
- `petrodollar_4_dedollarization.csv`
- `petrodollar_5_daily_prices_fx.csv`

These are the same filenames used by the supplied notebook/app.

## Train

```bash
pip install -r requirements.txt
python train_model.py
```

This creates:

```text
models/model_bundle.joblib
models/validation_metrics.csv
```

The model bundle contains the PCA transformation, index direction, feature list, trained Random Forest models for 6/12/24/36-month horizons, latest feature state and historical index.

## Run locally

```bash
streamlit run app_ml.py
```

## Deploy on Streamlit Community Cloud

Push this project to GitHub with:

```text
app_ml.py
train_model.py
requirements.txt
models/model_bundle.joblib
models/validation_metrics.csv
```

The five source CSVs do not need to be shipped to the inference app once the model bundle has been trained, unless you also want the application to rebuild/retrain the model.

Set the Streamlit entry point to:

```text
app_ml.py
```

## Deploy with Docker

```bash
docker build -t petrodollar-ml .
docker run -p 8501:8501 petrodollar-ml
```

Open:

```text
http://localhost:8501
```

## Important modeling note

The notebook's validation uses an 80/20 chronological split for 6, 12, 24 and 36 month targets. After model selection, `train_model.py` refits the selected Trend + Random Forest residual model on all rows for which the target is known, then persists the trained model.

The deployed app performs inference from the latest feature state. It does not retrain on every page load.

The notebook also explicitly notes that the source datasets contain a mix of real reported statistics and synthetic fill. Before using the model for real financial/policy decisions, replace those inputs with verified official data and retrain.
