# Petrodollar & De-dollarization — ML Deployment

This project separates **research/training** from **production inference**.

### Architecture

`5 CSV sources -> preprocessing -> PCA index -> features -> chronological validation -> Trend + Random Forest residual models -> model_bundle.joblib -> Streamlit inference app`

The Streamlit app loads the persisted model bundle. It does **not** retrain the ML models when a user opens the app.

## Files

- `train_model.py` — trains and persists the production models.
- `app_ml.py` — deployed Streamlit inference application.
- `app_original.py` — original exploratory Streamlit application for reference.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — container deployment option.
- `DEPLOYMENT_CHECKLIST.md` — deployment and retraining checklist.
- `data/` — place the five source CSVs here before training.
- `models/` — generated model bundle and validation metrics.

## Required source data

The trainer expects the same five files used by the supplied notebook:

- `petrodollar_1_oil_production_trade.csv`
- `petrodollar_2_opec_quotas_events.csv`
- `petrodollar_3_recycling_swf.csv`
- `petrodollar_4_dedollarization.csv`
- `petrodollar_5_daily_prices_fx.csv`

## Train the production model

```bash
pip install -r requirements.txt
python train_model.py
```

The trainer evaluates the Trend + Random Forest residual model on a chronological holdout for 6, 12, 24 and 36 months, then refits the selected model on all available historical rows and saves:

```text
models/model_bundle.joblib
models/validation_metrics.csv
```

The bundle includes the PCA transformation, index direction, feature list, four horizon-specific Random Forest models, latest feature state, historical index, validation metadata and residual uncertainty estimates.

## Run locally

```bash
streamlit run app_ml.py
```

## Deploy to Streamlit Community Cloud

1. Push the project to GitHub.
2. Make sure `models/model_bundle.joblib` and `models/validation_metrics.csv` are in the repository.
3. Select `app_ml.py` as the Streamlit entry point.
4. Deploy.

The source CSV files are not required by the inference app after the model bundle has been trained. Keep them private or outside the deployment repository if appropriate.

## Docker

```bash
docker build -t petrodollar-ml .
docker run -p 8501:8501 petrodollar-ml
```

Then open `http://localhost:8501`.

## Important notebook/modeling note

The original notebook is best retained as the research and EDA record. Its original PCA workflow fits the transformation before the chronological forecasting split. For deployment validation, `train_model.py` avoids that leakage by fitting PCA/scaling on the training period and transforming later observations with the fitted objects. After validation, the production artifacts are refit using all available historical data.

Therefore **you do not need to rewrite the notebook to deploy the app**. However, for a stronger portfolio submission, add a short “Deployment & Leakage Control” section to the notebook explaining that the production trainer uses a leakage-safe chronological validation procedure and persists the trained artifacts.

## Data quality

The supplied project materials indicate that some source figures may be synthetic-filled. This deployment should therefore be described as a **portfolio/demo forecasting system** unless and until the source data is replaced with verified official data and the model is retrained and revalidated.
