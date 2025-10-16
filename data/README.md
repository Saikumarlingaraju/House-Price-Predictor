# Data

This folder contains auxiliary data used by the app.

- `data/centroids.json` — optional precomputed per-city centroids used as an offline geocoding fallback. Format:

```json
{
  "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "count": 1245}
}
```

How to generate centroids
```powershell
python scripts\compute_centroids.py --input merged_hyderabad.csv --out data\centroids.json
```

Sample dataset
- Consider adding `data/sample_hyderabad.csv` (small subset) for quick local runs and CI tests.