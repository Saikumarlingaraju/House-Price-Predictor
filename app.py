import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Optional mapping/geocoding libs (used if installed). We import lazily where required and fall back cleanly.
try:
    import folium
    from streamlit_folium import st_folium
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    GEOCODE_AVAILABLE = True
except Exception:
    GEOCODE_AVAILABLE = False

ROOT = Path(__file__).parent
model = None
df = None
data_source = 'Unknown'

# Discover available models (model_*.pkl)
models = sorted([p.name for p in ROOT.glob('model_*.pkl')])
selected_model = None
if models:
    if 'chosen_model' not in st.session_state:
        st.session_state['chosen_model'] = models[0]
else:
    if (ROOT / 'model_india.pkl').exists():
        models = ['model_india.pkl']
        if 'chosen_model' not in st.session_state:
            st.session_state['chosen_model'] = models[0]

if not models and (ROOT / 'model.pkl').exists():
    models = ['model.pkl']
    if 'chosen_model' not in st.session_state:
        st.session_state['chosen_model'] = models[0]


st.set_page_config(page_title='House Price Predictor (INR)', layout='wide')

# Sidebar: model selection and info
with st.sidebar:
    st.title('Model & Data')
    if models:
        chosen = st.selectbox('Choose model', options=models, index=models.index(st.session_state.get('chosen_model')) if st.session_state.get('chosen_model') in models else 0)
        st.session_state['chosen_model'] = chosen
        selected_model = ROOT / chosen
        df_candidate = ROOT / f"df_{chosen.replace('model_', '').replace('.pkl','')}.pkl"
        if df_candidate.exists():
            try:
                df = pickle.load(open(df_candidate, 'rb'))
                data_source = df_candidate.name
            except Exception:
                df = None
        # try load metadata json
        meta_candidate = ROOT / f"{selected_model.stem}_meta.json"
        meta = None
        if meta_candidate.exists():
            try:
                import json
                meta = json.load(open(meta_candidate, 'r', encoding='utf8'))
            except Exception:
                meta = None
        elif (ROOT / 'df_india.pkl').exists():
            try:
                df = pickle.load(open(ROOT / 'df_india.pkl', 'rb'))
                data_source = 'df_india.pkl'
            except Exception:
                df = None
    else:
        st.info('No model_*.pkl files found. Run training to create model files.')

    st.markdown('---')
    st.markdown(f'**Data source:** {data_source}')
    if df is not None:
        st.markdown(f'- Rows (loaded): **{len(df):,}**')
        # list amenity columns
        amen_cols = []
        for c in df.columns:
            if c in ['Price', 'Area', 'No. of Bedrooms', 'City']:
                continue
            try:
                if pd.api.types.is_numeric_dtype(df[c]):
                    vals = set(df[c].dropna().unique())
                    if vals.issubset({0, 1}):
                        amen_cols.append(c)
            except Exception:
                continue
        if amen_cols:
            st.markdown('**Detected amenity flags:**')
            for a in amen_cols:
                st.markdown(f'- {a}')
        # Display metadata metrics if available
        if meta is not None and 'metrics' in meta:
            st.markdown('---')
            st.markdown('**Model metrics (from training)**')
            m = meta['metrics']
            c1, c2, c3 = st.columns(3)
            c1.metric('MAE', f"₹ {m['mae']:,.0f}")
            c2.metric('RMSE', f"₹ {m['rmse']:,.0f}")
            c3.metric('R²', f"{m['r2']:.3f}")
            # feature list
            if 'features' in meta:
                st.markdown('**Features used:** ' + ', '.join(meta['features']))
            # show feature ranges and categories
            if 'feature_ranges' in meta and meta['feature_ranges']:
                st.markdown('**Feature ranges (from training data)**')
                for k, v in meta['feature_ranges'].items():
                    st.markdown(f'- {k}: min={v["min"]:,}, median={v["median"]:,}, max={v["max"]:,}')
            if 'categories' in meta and meta['categories']:
                st.markdown('**Categorical values (sample)**')
                for k, vals in meta['categories'].items():
                    st.markdown(f'- {k}: {vals[:10]}')
            # importances (raw) - show small chart if present and lengths align
            if meta.get('importances_available') and meta.get('features') and meta.get('importances'):
                imps = meta.get('importances', {}).get('raw', [])
                feats = meta.get('features', [])
                if len(imps) == len(feats) and len(imps) > 0:
                    try:
                        df_imp = pd.DataFrame({'feature': feats, 'importance': imps})
                        df_imp = df_imp.sort_values('importance', ascending=False).head(20).set_index('feature')
                        st.bar_chart(df_imp['importance'])
                    except Exception:
                        pass

    # compute quick evaluation (cached) if model and df present
    @st.cache_data(ttl=300)
    def compute_metrics(model_path, df, amen_cols):
        try:
            m = pickle.load(open(model_path, 'rb'))
        except Exception:
            return None
        # build feature DataFrame from df using available columns
        features = ['Area', 'No. of Bedrooms', 'City'] + amen_cols
        missing = [c for c in features if c not in df.columns]
        if missing:
            return None
        X = df[features].copy()
        y = df['Price']
        try:
            preds = m.predict(X)
            mae = float(np.mean(np.abs(y - preds)))
            rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
            # r2 fallback
            ss_res = float(np.sum((y - preds) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
            return {'mae': mae, 'rmse': rmse, 'r2': r2}
        except Exception:
            return None

    st.markdown('---')
    st.markdown('**Utilities**')
    if st.button('Quick test (median values)'):
        if selected_model is not None and selected_model.exists():
            try:
                model = pickle.load(open(selected_model, 'rb'))
            except Exception as e:
                st.error(f'Failed loading model: {e}')
                model = None
        if model is None:
            st.error('No model loaded for quick test')
        else:
            sample = {'Area': df['Area'].median() if (df is not None and 'Area' in df.columns) else 1000,
                      'No. of Bedrooms': int(df['No. of Bedrooms'].median()) if (df is not None and 'No. of Bedrooms' in df.columns) else 2,
                      'City': (df['City'].mode()[0] if (df is not None and 'City' in df.columns and not df['City'].mode().empty) else '')}
            # sample amenity defaults
            for a in amen_cols:
                sample[a] = int(bool(df[a].median() >= 0.5)) if (df is not None and a in df.columns) else 0
            sample_df = pd.DataFrame([sample])
            try:
                pred = model.predict(sample_df)
                st.success(f'Quick test prediction: ₹ {float(pred[0]):,.0f}')
            except Exception as e:
                st.error(f'Quick test failed: {e}')

    # If model and df present, show quick metrics
    if df is not None and selected_model is not None and selected_model.exists():
        metrics = compute_metrics(str(selected_model), df, amen_cols)
        if metrics:
            col1, col2, col3 = st.columns(3)
            col1.metric('MAE', f'₹ {metrics["mae"]:,.0f}')
            col2.metric('RMSE', f'₹ {metrics["rmse"]:,.0f}')
            col3.metric('R²', f'{metrics["r2"]:.3f}')


# Main UI
st.markdown("""
<div style='text-align:center'>
  <h1>₹ House Price Predictor</h1>
  <p style='color:#666'>Enter property details and choose a model from the sidebar.</p>
</div>
""", unsafe_allow_html=True)

def get_median(col, default=0):
    if df is None or col not in df.columns:
        return default
    try:
        return float(df[col].median())
    except Exception:
        return default


def format_inr(x):
    try:
        return f'₹ {int(x):,}'
    except Exception:
        return str(x)


def haversine(lat1, lon1, lat2, lon2):
    # returns distance in kilometers
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))


# Geocoding helpers (cache results in session_state to avoid repeated queries)
def ensure_geocode_cache():
    if 'loc_coords' not in st.session_state:
        st.session_state['loc_coords'] = {}
    if 'geolocator' not in st.session_state and GEOCODE_AVAILABLE:
        try:
            loc = Nominatim(user_agent='house-price-app')
            # RateLimiter to avoid hitting service too fast
            st.session_state['geolocator'] = loc
            st.session_state['geocode'] = RateLimiter(loc.geocode, min_delay_seconds=1, max_retries=2)
            st.session_state['reverse'] = RateLimiter(loc.reverse, min_delay_seconds=1, max_retries=2)
        except Exception:
            st.session_state['geolocator'] = None


def geocode_location(location_name):
    """Return (lat, lon) for a location name. Uses session cache."""
    if not GEOCODE_AVAILABLE:
        return None
    ensure_geocode_cache()
    cache = st.session_state['loc_coords']
    if not location_name:
        return None
    if location_name in cache:
        return cache[location_name]
    try:
        geocode = st.session_state.get('geocode')
        if geocode is None:
            return None
        q = f"{location_name}, Hyderabad, India"
        res = geocode(q, timeout=10)
        if res:
            latlon = (res.latitude, res.longitude)
            cache[location_name] = latlon
            return latlon
    except Exception:
        return None
    return None


def reverse_geocode(lat, lon):
    if not GEOCODE_AVAILABLE:
        return None
    ensure_geocode_cache()
    rev = st.session_state.get('reverse')
    if rev is None:
        return None
    try:
        r = rev((lat, lon), exactly_one=True, timeout=10)
        if r:
            return r.address
    except Exception:
        return None
    return None


def is_streamlit_runtime():
    """Return True when running inside the Streamlit server/runtime.
    This helps avoid performing long-running or network operations when the
    module is merely imported (for example during quick import checks).
    """
    try:
        # Preferred fast-path: Streamlit exposes a helper in some versions
        fn = getattr(st, '_is_running_with_streamlit', None)
        if callable(fn):
            return bool(fn())
    except Exception:
        pass
    try:
        import os
        # Fall back to checking environment variables that Streamlit sets
        if os.environ.get('STREAMLIT_SERVER_PORT') or os.environ.get('STREAMLIT_RUN_MAIN'):
            return True
    except Exception:
        pass
    return False


left, right = st.columns([2, 1])
with left:
    st.header('Property details')
    st.markdown('Enter the key details below. Use the "Advanced amenities" section to toggle many features quickly.')

    # Primary inputs arranged compactly
    c1, c2 = st.columns([2, 1])
    with c1:
        area = st.number_input('Area (sqft)', value=int(get_median('Area', 1000)), min_value=10, step=10)
        # small slider for quick adjustments (same range as data)
        try:
            ar_min = int(meta.get('feature_ranges', {}).get('Area', {}).get('min', max(10, get_median('Area', 1000) - 500)))
            ar_max = int(meta.get('feature_ranges', {}).get('Area', {}).get('max', get_median('Area', 5000)))
        except Exception:
            ar_min, ar_max = 10, max(2000, int(get_median('Area', 1000)))
        area = st.slider('Adjust Area', min_value=max(10, ar_min), max_value=max(ar_min + 1, ar_max), value=int(area))
    with c2:
        bedrooms = st.number_input('Bedrooms', value=int(get_median('No. of Bedrooms', 2)), min_value=0, max_value=20, step=1)

    # Map picker: show interactive map above Location input if mapping libs are available
    suggested_location = None
    picked_latlon = None
    if GEOCODE_AVAILABLE:
        st.markdown('**Pick on map** — click to drop a pin and auto-suggest a Location')
        # prepare centroids for each Location once. Prefer precomputed centroids
        # saved in the model metadata to avoid runtime geocoding. Fallback to
        # existing behaviors (lat/lon in DF, or on-the-fly geocoding) if meta
        # centroids aren't available.
        location_centroids = None
        # try load precomputed centroids from metadata if available
        try:
            if meta is not None and isinstance(meta.get('location_centroids', None), dict) and meta.get('location_centroids'):
                lc = meta.get('location_centroids')
                # convert dict -> DataFrame for consistent handling below
                try:
                    location_centroids = pd.DataFrame.from_dict(lc, orient='index', columns=['Latitude', 'Longitude'])
                except Exception:
                    location_centroids = None
        except Exception:
            location_centroids = None

        # existing fallback: use DF lat/lon if available
        if location_centroids is None and df is not None and 'Location' in df.columns and 'Latitude' in df.columns and 'Longitude' in df.columns:
            try:
                location_centroids = df.groupby('Location')[['Latitude', 'Longitude']].median().dropna()
            except Exception:
                location_centroids = None

        # final fallback: attempt to geocode a limited set of unique locations
        # Only attempt this when running inside the Streamlit runtime. Importing
        # the module (for tests or quick checks) should never trigger network
        # geocoding which can block or fail when offline.
        if is_streamlit_runtime() and location_centroids is None and df is not None and 'Location' in df.columns:
            unique_locs = sorted(df['Location'].dropna().unique().tolist())[:200]
            loc_coords = {}
            # Show a spinner/progress because geocoding may take a while
            try:
                with st.spinner(f'Geocoding up to {len(unique_locs)} locations (may take a while)...'):
                    progress = None
                    if len(unique_locs) > 1:
                        progress = st.progress(0)
                    for i, L in enumerate(unique_locs):
                        latlon = geocode_location(L)
                        if latlon:
                            loc_coords[L] = latlon
                        if progress is not None:
                            try:
                                progress.progress(int((i + 1) / len(unique_locs) * 100))
                            except Exception:
                                pass
            except Exception:
                # ignore UI failures
                pass
            if loc_coords:
                try:
                    location_centroids = pd.DataFrame.from_dict(loc_coords, orient='index', columns=['Latitude', 'Longitude'])
                except Exception:
                    location_centroids = None

        # default center Hyderabad
        center = (17.3850, 78.4867)
        if location_centroids is not None and len(location_centroids):
            try:
                first = location_centroids.iloc[0]
                center = (float(first['Latitude']), float(first['Longitude']))
            except Exception:
                center = center

        try:
            m = folium.Map(location=center, zoom_start=12, tiles=None)
            folium.TileLayer('OpenStreetMap', name='Street').add_to(m)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Esri Satellite',
                overlay=False,
                control=True
            ).add_to(m)
            # show some sample markers (optional): show last predictions from session if available
            preds = st.session_state.get('pred_history', [])
            for ph in preds[:10]:
                try:
                    if ph.get('lat') and ph.get('lon'):
                        folium.CircleMarker(location=[ph['lat'], ph['lon']], radius=4, color='#ff5722', fill=True).add_to(m)
                except Exception:
                    pass
            # Add a layer control so users can toggle basemaps
            folium.LayerControl().add_to(m)
            # Small legend (price marker color)
            legend_html = '''
             <div style="position: fixed; 
                         bottom: 50px; left: 10px; width:150px; height:60px; 
                         border:2px solid grey; z-index:9999; font-size:12px; background:white; padding:6px;">
             <b>Legend</b><br>
             <i style="background:#ff5722;border-radius:50%;display:inline-block;width:10px;height:10px;margin-right:6px"></i> Recent preds
             </div>
             '''
            m.get_root().html.add_child(folium.Element(legend_html))
            m.add_child(folium.LatLngPopup())
            map_data = st_folium(m, height=350, returned_objects=['last_clicked'])
            if map_data and map_data.get('last_clicked'):
                lc = map_data['last_clicked']
                picked_latlon = (lc['lat'], lc['lng'])
                # Reverse geocoding can be slow; provide immediate feedback
                with st.spinner('Reverse-geocoding selected point...'):
                    rev = reverse_geocode(lc['lat'], lc['lng'])
                if rev:
                    st.markdown(f'*Reverse-geocoded address:* {rev}')
                # find nearest known Location from centroids
                if location_centroids is not None:
                    best = None
                    best_dist = None
                    with st.spinner('Finding nearest known Location...'):
                        for L, row in location_centroids.iterrows():
                            try:
                                lat2 = float(row['Latitude'])
                                lon2 = float(row['Longitude'])
                                d = haversine(lc['lat'], lc['lng'], lat2, lon2)
                                if best is None or d < best_dist:
                                    best = L
                                    best_dist = d
                            except Exception:
                                continue
                    if best is not None:
                        suggested_location = f"{best} (≈ {best_dist:.1f} km)"
                        st.success(f"Suggested Location: {suggested_location}")
                        # Allow user to apply the suggestion into the Location selectbox
                        try:
                            if st.button('Apply suggested Location'):
                                # store the raw location name (without distance) so the selectbox can pick it
                                st.session_state['chosen_location_to_apply'] = best
                                st.experimental_rerun()
                        except Exception:
                            pass
                    else:
                        st.warning('No nearby known Location found for the selected point. Try a nearby point or type the Location manually.')
        except Exception as e:
            st.warning('Mapping libraries are not available or failed to load. Install folium, streamlit-folium, geopy to enable map features.')

    # Location / City: prefer a selectbox to reduce typos
    if df is not None and 'Location' in df.columns:
        locations = sorted(df['Location'].dropna().unique().tolist())
        if suggested_location:
            # show suggested_location as help text and leave selection unchanged
            st.markdown(f"**Auto-suggested:** {suggested_location}")
        # If user applied a suggested location via the button, preselect it
        chosen_to_apply = st.session_state.pop('chosen_location_to_apply', None) if 'chosen_location_to_apply' in st.session_state else None
        if chosen_to_apply and chosen_to_apply in locations:
            try:
                idx = locations.index(chosen_to_apply) + 1
            except Exception:
                idx = 0
        else:
            idx = 0
        loc = st.selectbox('Location (area/locality)', options=['(unknown)'] + locations, index=idx)
    else:
        loc = st.text_input('Location (optional)', value='')

    # Resale and a few common categorical flags presented as radios/selectboxes
    resale = st.radio('Resale?', options=['No', 'Yes'], index=0 if get_median('Resale', 0) < 0.5 else 1)
    # Maintenance staff: map to simple choices
    ms_map = {'No': 0, 'Yes': 1, 'Unknown': 2}
    ms_default = 'Unknown'
    try:
        md = int(get_median('MaintenanceStaff', 2))
        if md == 0:
            ms_default = 'No'
        elif md == 1:
            ms_default = 'Yes'
    except Exception:
        ms_default = 'Unknown'
    maintenance = st.selectbox('Maintenance staff', options=['Unknown', 'No', 'Yes'], index=['Unknown', 'No', 'Yes'].index(ms_default))

    # A few quick toggles for frequently used amenities
    quick_cols = st.columns(3)
    amenity_values = {}
    amenity_cols = amen_cols if (df is not None) else []
    def amen_default(col):
        try:
            return bool(df[col].median() >= 0.5)
        except Exception:
            return False

    # Common quick toggles
    with quick_cols[0]:
        car_parking = st.selectbox('Car parking', options=['Unknown', 'No', 'Yes'], index=0)
    with quick_cols[1]:
        ac = st.checkbox('AC available', value=amen_default('AC') if 'AC' in amenity_cols else False)
    with quick_cols[2]:
        wifi = st.checkbox('Wifi', value=amen_default('Wifi') if 'Wifi' in amenity_cols else False)

    # Advanced amenities grid (multi-column) to avoid long lists
    if amenity_cols:
        with st.expander('Advanced amenities (many)'):
            cols = st.columns(3)
            for i, a in enumerate(amenity_cols):
                col = cols[i % len(cols)]
                amenity_values[a] = col.checkbox(a, value=amen_default(a))

    # Buttons: Predict and Clear
    btn_cols = st.columns([1, 1])
    do_predict = btn_cols[0].button('Predict')
    do_clear = btn_cols[1].button('Clear inputs')

    if do_clear:
        # reset some session values (non-destructive)
        if 'pred_history' in st.session_state:
            st.session_state.pop('pred_history')
        st.experimental_rerun()

    if do_predict:
        # ensure model is (re)loaded
        if selected_model is not None and selected_model.exists():
            try:
                model = pickle.load(open(selected_model, 'rb'))
            except Exception as e:
                st.error(f'Failed loading model: {e}')

        if model is None:
            st.error('No model found. Please run training to produce a model_*.pkl file.')
        else:
            data = {
                'Area': int(area),
                'No. of Bedrooms': int(bedrooms),
                'City': (df['City'].mode()[0] if (df is not None and 'City' in df.columns and not df['City'].mode().empty) else 'Hyderabad')
            }
            # prefer using Location if provided
            if loc and loc != '(unknown)':
                data['Location'] = loc
            # map simple flags
            data['Resale'] = 1 if resale == 'Yes' else 0
            data['MaintenanceStaff'] = ms_map.get(maintenance, 2)
            # quick toggles
            data['CarParking'] = 1 if car_parking == 'Yes' else (0 if car_parking == 'No' else 2)
            data['AC'] = int(bool(ac))
            data['Wifi'] = int(bool(wifi))
            # advanced amenities
            for a, val in amenity_values.items():
                data[a] = int(bool(val))

            input_df = pd.DataFrame([data])
            # input validation using meta if available
            if meta is not None:
                fr = meta.get('feature_ranges', {})
                cats = meta.get('categories', {})
                warnings = []
                if 'Area' in fr:
                    if area < fr['Area']['min'] or area > fr['Area']['max']:
                        warnings.append(f"Area {area} outside training range [{fr['Area']['min']:.0f} - {fr['Area']['max']:.0f}]")
                if 'No. of Bedrooms' in fr:
                    if bedrooms < fr['No. of Bedrooms']['min'] or bedrooms > fr['No. of Bedrooms']['max']:
                        warnings.append(f"Bedrooms {bedrooms} outside training range [{fr['No. of Bedrooms']['min']:.0f} - {fr['No. of Bedrooms']['max']:.0f}]")
                if 'City' in cats and data.get('City') and data['City'] not in cats['City']:
                    warnings.append(f"City '{data['City']}' was not present in training data (will use unknown category)")
                if warnings:
                    for w in warnings:
                        st.warning(w)

            try:
                pred = model.predict(input_df)
                price = float(pred[0])
                # show result in a larger card-like layout
                r1, r2 = st.columns([3, 2])
                with r1:
                    st.markdown('<div style="background:#f8f9fa;padding:18px;border-radius:8px">', unsafe_allow_html=True)
                    st.markdown(f'<h2 style="margin:0">{format_inr(price)}</h2>', unsafe_allow_html=True)
                    ppsq = price / area if area else 0
                    st.markdown(f'<div style="color:#666;margin-top:6px">Price per sqft: <strong>{format_inr(ppsq)}</strong></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with r2:
                    # relative to city median
                    city_median = None
                    if df is not None and 'City' in df.columns and data.get('City') in df['City'].unique():
                        try:
                            city_median = float(df.loc[df['City'] == data.get('City'), 'Price'].median())
                        except Exception:
                            city_median = None
                    if city_median:
                        pct = (price - city_median) / city_median * 100 if city_median else 0
                        st.metric('vs. city median', f'{pct:+.1f}%')
                    else:
                        st.write('No city median available')

                    # Link to open the selected coordinates in Google Maps
                    if picked_latlon:
                        lat, lon = picked_latlon
                        gmaps = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                        st.markdown(f"[Open in Google Maps]({gmaps})")

                # save history in session_state
                if 'pred_history' not in st.session_state:
                    st.session_state['pred_history'] = []
                hist_entry = {'price': price, 'area': area, 'bedrooms': bedrooms, 'city': data.get('City')}
                if picked_latlon:
                    hist_entry['lat'] = float(picked_latlon[0])
                    hist_entry['lon'] = float(picked_latlon[1])
                st.session_state['pred_history'].insert(0, hist_entry)
            except Exception as e:
                st.error(f'Prediction failed: {e}')

with right:
    st.header('Model & history')
    st.markdown(f"**Data source:** {data_source}")
    if df is not None:
        st.markdown(f"Rows in loaded DF: **{len(df):,}**")
    st.markdown('---')
    st.write('Prediction history (recent)')
    hist = st.session_state.get('pred_history', [])
    if hist:
        prices = [h['price'] for h in hist[:50]]
        st.line_chart(pd.DataFrame({'price': prices}))
        for h in hist[:10]:
            st.markdown(f"- {format_inr(h['price'])} — {h['area']} sqft — {h['bedrooms']} BR — {h['city']}")
        # allow export
        csv = pd.DataFrame(hist).to_csv(index=False).encode('utf-8')
        st.download_button('Download history CSV', data=csv, file_name='predictions_history.csv', mime='text/csv')
    else:
        st.write('_No predictions yet_')



