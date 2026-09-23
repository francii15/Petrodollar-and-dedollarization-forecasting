# Deployment Checklist

## Notebook
- Keep the notebook as the research/EDA document.
- Do not use the notebook's in-memory `artifacts` as the production model.
- The production training logic is in `train_model.py`.
- The deployment trainer fits PCA/scaling only on the chronological training window during validation, then refits on all available history after validation.

## Before training
1. Put all five CSV files in `data/`.
2. Verify the column names match the notebook.
3. Prefer verified official data for any real-world use.

## Train
```bash
python train_model.py
```

Expected outputs:
- `models/model_bundle.joblib`
- `models/validation_metrics.csv`

## Test
```bash
streamlit run app_ml.py
```

## Deploy
Push `app_ml.py`, `requirements.txt`, `models/model_bundle.joblib`, and `models/validation_metrics.csv` to GitHub. Deploy `app_ml.py` as the Streamlit entry point.

## Retraining
Retrain whenever the underlying data is refreshed materially or when model monitoring shows deterioration. Do not retrain inside the Streamlit request path.
