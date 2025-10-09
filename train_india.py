"""Train a baseline house price model from merged_files.csv and save a model file.

Usage:
    python train_india.py --csv merged_files.csv --state Hyderabad --n-estimators 100

This script:
- loads the CSV, performs simple cleaning
- detects binary amenity columns (0/1) automatically
- trains a RandomForest pipeline and saves model_<state>.pkl and df_<state>.pkl
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def find_binary_amenities(df, exclude=None):
    """Return list of columns that look like binary amenity flags (0/1)."""
    if exclude is None:
        exclude = []
    amenities = []
    for col in df.columns:
        if col in exclude:
            continue
        try:
            if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
                vals = df[col].dropna().unique()
                if set(vals).issubset({0, 1}):
                    amenities.append(col)
        except Exception:
            # ignore columns that raise during dtype checks
            continue
    return amenities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', default='india', help='State or tag used in output filenames')
    parser.add_argument('--n-estimators', type=int, default=50, help='Number of trees for RandomForest')
    parser.add_argument('--csv', default='merged_files.csv', help='Path to merged CSV')
    args = parser.parse_args()

    ROOT = Path(__file__).parent
    csv_path = ROOT / args.csv
    if not csv_path.exists():
        raise SystemExit(f"Dataset not found: {csv_path}")

    print('Loading dataset...')
    df = pd.read_csv(csv_path)
    print(f'Rows: {len(df)}, Columns: {len(df.columns)}')

    required = ['Price', 'Area', 'No. of Bedrooms', 'City']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f'Missing required columns in CSV: {missing}')

    orig_len = len(df)
    # Basic cleaning
    df = df[df['Price'].notna()]
    df = df[df['Area'].notna()]
    df = df[(df['Area'] > 0) & (df['Price'] > 0)]
    df['City'] = df['City'].astype(str).str.strip()
    for col in ['Area', 'No. of Bedrooms', 'Price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df['Area'].notna() & df['Price'].notna()]
    print(f'Rows after cleaning: {len(df)} (dropped {orig_len - len(df)})')

    # Detect binary amenity columns automatically and include them
    exclude = set(required)
    amenities = find_binary_amenities(df, exclude=exclude)
    print(f'Detected amenity/binary features: {amenities}')

    # Save cleaned df for UI defaults
    df_path = ROOT / f'df_{args.state}.pkl'
    with open(df_path, 'wb') as fh:
        pickle.dump(df, fh)
    print(f'Saved cleaned df to: {df_path}')

    feature_cols = ['Area', 'No. of Bedrooms', 'City'] + amenities
    X = df[feature_cols]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ['Area', 'No. of Bedrooms'] + [c for c in amenities if pd.api.types.is_numeric_dtype(df[c])]
    cat_features = ['City']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # OneHotEncoder compatibility
    try:
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='(unknown)')),
        ('onehot', onehot_encoder)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', cat_transformer, cat_features)
        ],
        remainder='passthrough'  # amenity columns (0/1) will be passed through
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=args.n_estimators, random_state=42, n_jobs=-1))
    ])

    print('Training model...')
    model.fit(X_train, y_train)

    print('Evaluating...')
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f'MAE: {mae:,.2f}')
    print(f'RMSE: {rmse:,.2f}')
    print(f'R2: {r2:.4f}')

    model_path = ROOT / f'model_{args.state}.pkl'
    with open(model_path, 'wb') as fh:
        pickle.dump(model, fh)
    print(f'Saved model to: {model_path}')

    # Save metadata (metrics, features, importances if available)
    meta = {
        'state': args.state,
        'n_estimators': args.n_estimators,
        'features': feature_cols,
        'metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }
    }
    # Save feature ranges and categories to help the app warn on out-of-range inputs
    feature_ranges = {}
    categories = {}
    for col in feature_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            feature_ranges[col] = {'min': float(df[col].min()), 'max': float(df[col].max()), 'median': float(df[col].median())}
        elif col in df.columns and pd.api.types.is_object_dtype(df[col]):
            categories[col] = sorted(df[col].dropna().unique().tolist())

    meta['feature_ranges'] = feature_ranges
    meta['categories'] = categories
    # Try to extract feature importances from the trained RandomForestRegressor
    try:
        # the regressor is the last step in the pipeline
        rf = model.named_steps['regressor']
        if hasattr(rf, 'feature_importances_'):
            meta['importances_available'] = True
            meta['importances'] = {'raw': rf.feature_importances_.tolist()}
        else:
            meta['importances_available'] = False
    except Exception:
        meta['importances_available'] = False

    # Compute location centroids to speed up runtime map suggestions.
    # Prefer using lat/lon columns if present. Otherwise, attempt optional geocoding
    # (geopy) if available. The result is a mapping {Location: [lat, lon]} stored
    # in the metadata JSON under 'location_centroids'. This avoids runtime
    # geocoding in the Streamlit app and reduces rate-limit issues.
    location_centroids = {}
    if 'Location' in df.columns:
        # If the dataset already contains Latitude/Longitude columns, use medians
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            try:
                cent = df.groupby('Location')[['Latitude', 'Longitude']].median().dropna()
                # convert to simple dict
                for idx, row in cent.iterrows():
                    try:
                        location_centroids[str(idx)] = [float(row['Latitude']), float(row['Longitude'])]
                    except Exception:
                        continue
            except Exception:
                location_centroids = {}
        else:
            # Optional: try geocoding unique Location names if geopy is installed.
            try:
                from geopy.geocoders import Nominatim
                from geopy.extra.rate_limiter import RateLimiter
                geoloc = Nominatim(user_agent='house-price-train')
                geocode = RateLimiter(geoloc.geocode, min_delay_seconds=1, max_retries=2)
                unique_locs = sorted(df['Location'].dropna().unique().tolist())[:500]
                for L in unique_locs:
                    try:
                        q = f"{L}, {df['City'].mode()[0] if 'City' in df.columns else ''}, India"
                        res = geocode(q, timeout=10)
                        if res:
                            location_centroids[L] = [float(res.latitude), float(res.longitude)]
                    except Exception:
                        continue
            except Exception:
                # geopy not available or failed — leave centroids empty
                location_centroids = {}

    meta['location_centroids'] = location_centroids

    import json
    meta_path = ROOT / f'model_{args.state}_meta.json'
    with open(meta_path, 'w', encoding='utf8') as fh:
        json.dump(meta, fh, indent=2)
    print(f'Saved metadata to: {meta_path}')


if __name__ == '__main__':
    main()
