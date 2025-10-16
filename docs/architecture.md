# Architecture

Overview
- Streamlit app (`app.py`) is the frontend and orchestrates model loading, predictions, and analytics.
- Models are trained with `train_india.py` and persisted as `model_<state>.pkl` plus metadata JSON.
- `scripts/compute_centroids.py` computes offline centroids for mapping when geocoding is unavailable.

Key components
- UI: Streamlit tabs for Predict, Analytics, Compare, and Help.
- ML: scikit-learn pipeline (preprocessing + RandomForestRegressor) in `train_india.py`.
- Data: Pickled DataFrames `df_<state>.pkl` saved after cleaning; `data/centroids.json` for offline mapping.
- Caching: Streamlit caching used to avoid reloading large artifacts on each rerun.

Notes
- No HTTP API currently; adding a FastAPI service would allow programmatic access to `/predict`.
