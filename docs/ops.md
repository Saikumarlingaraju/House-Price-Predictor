# Operations / Deployment Notes

Run locally
- Create a Python venv and install requirements.
- Start Streamlit: `python -m streamlit run app.py`.

Dependencies
- Core: streamlit, pandas, numpy, scikit-learn, plotly
- Optional (maps): folium, streamlit-folium, geopy

Docker (suggested)
- A minimal Dockerfile could use `python:3.11-slim` and install `requirements.txt`, then run `streamlit run app.py --server.port $PORT`.

Production notes
- Use a process manager (systemd, supervisord) or container orchestration.
- Persist models in a shared storage or mount `models/` into the container.
- Protect Nominatim API usage and cache geocoding results.

Monitoring
- Add structured logs and a Prometheus exporter if exposing metrics.

Backups
- Don't commit large model files; use Git LFS or an artifact store.
