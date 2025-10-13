"""
Enhanced House Price Predictor - Streamlit App
A user-friendly interface for predicting house prices in India with advanced features
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Optional mapping/geocoding libs
try:
    import folium
    from streamlit_folium import st_folium
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    GEOCODE_AVAILABLE = True
except Exception:
    GEOCODE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title='House Price Predictor (INR)', 
    layout='wide',
    page_icon='🏡',
    initial_sidebar_state='expanded',
    menu_items={
        'Get Help': 'https://github.com/Saikumarlingaraju/House-Price-Predictor',
        'Report a bug': 'https://github.com/Saikumarlingaraju/House-Price-Predictor/issues',
        'About': 'House Price Predictor v2.0 - Powered by Machine Learning'
    }
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 2.5rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    
    /* Prediction result card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 2rem 0;
    }
    
    .prediction-price {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        color: #667eea;
        cursor: help;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pred_history' not in st.session_state:
    st.session_state['pred_history'] = []
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = []
if 'comparison_list' not in st.session_state:
    st.session_state['comparison_list'] = []

# Global variables
ROOT = Path(__file__).parent
model = None
df = None
meta = None
data_source = 'Unknown'
amen_cols = []

# Discover available models
models = sorted([p.name for p in ROOT.glob('model_*.pkl')])
if not models and (ROOT / 'model_india.pkl').exists():
    models = ['model_india.pkl']
if not models and (ROOT / 'model.pkl').exists():
    models = ['model.pkl']

if 'chosen_model' not in st.session_state and models:
    st.session_state['chosen_model'] = models[0]

# Utility Functions
def get_median(col, default=0):
    """Get median value from dataframe column"""
    if df is None or col not in df.columns:
        return default
    try:
        return float(df[col].median())
    except Exception:
        return default

def format_inr(x):
    """Format number as Indian Rupees"""
    try:
        val = int(x)
        if val >= 10000000:  # 1 crore
            return f'₹ {val/10000000:.2f} Cr'
        elif val >= 100000:  # 1 lakh
            return f'₹ {val/100000:.2f} L'
        else:
            return f'₹ {val:,}'
    except Exception:
        return str(x)

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers"""
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))

def is_streamlit_runtime():
    """Check if running in Streamlit runtime"""
    try:
        fn = getattr(st, '_is_running_with_streamlit', None)
        if callable(fn):
            return bool(fn())
    except Exception:
        pass
    try:
        import os
        if os.environ.get('STREAMLIT_SERVER_PORT') or os.environ.get('STREAMLIT_RUN_MAIN'):
            return True
    except Exception:
        pass
    return False

def load_model_and_data(model_name):
    """Load model, dataframe, and metadata"""
    global model, df, meta, data_source, amen_cols
    
    selected_model = ROOT / model_name
    if not selected_model.exists():
        return False
    
    try:
        # Load model
        with open(selected_model, 'rb') as f:
            model = pickle.load(f)
        
        # Load dataframe
        df_candidate = ROOT / f"df_{model_name.replace('model_', '').replace('.pkl','')}.pkl"
        if df_candidate.exists():
            with open(df_candidate, 'rb') as f:
                df = pickle.load(f)
            data_source = df_candidate.name
        
        # Load metadata
        meta_candidate = ROOT / f"{selected_model.stem}_meta.json"
        if meta_candidate.exists():
            with open(meta_candidate, 'r', encoding='utf8') as f:
                meta = json.load(f)
        
        # Detect amenity columns
        amen_cols = []
        if df is not None:
            for c in df.columns:
                if c in ['Price', 'Area', 'No. of Bedrooms', 'City', 'Location', 'Latitude', 'Longitude']:
                    continue
                try:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        vals = set(df[c].dropna().unique())
                        if vals.issubset({0, 1}):
                            amen_cols.append(c)
                except Exception:
                    continue
        
        return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Model Selection")
    
    if models:
        chosen = st.selectbox(
            'Choose prediction model',
            options=models,
            index=models.index(st.session_state.get('chosen_model')) if st.session_state.get('chosen_model') in models else 0,
            help="Select which trained model to use for predictions"
        )
        
        if chosen != st.session_state.get('chosen_model'):
            st.session_state['chosen_model'] = chosen
            load_model_and_data(chosen)
        elif model is None:
            load_model_and_data(chosen)
    else:
        st.warning('⚠️ No model files found. Please run training first.')
        st.info('Run: `python train_india.py --csv merged_files.csv --state YourState`')
    
    st.markdown('---')
    
    # Model Info
    if df is not None:
        st.markdown("### 📊 Data Overview")
        st.metric("Total Properties", f"{len(df):,}")
        
        if 'City' in df.columns:
            st.metric("Cities", df['City'].nunique())
        
        if 'Location' in df.columns:
            st.metric("Locations", df['Location'].nunique())
        
        # Price statistics
        if 'Price' in df.columns:
            st.markdown("### 💰 Price Range")
            col1, col2 = st.columns(2)
            col1.metric("Min", format_inr(df['Price'].min()))
            col2.metric("Max", format_inr(df['Price'].max()))
            st.metric("Median", format_inr(df['Price'].median()))
    
    # Model Metrics
    if meta is not None and 'metrics' in meta:
        st.markdown('---')
        st.markdown("### 🎯 Model Performance")
        m = meta['metrics']
        
        col1, col2 = st.columns(2)
        col1.metric('MAE', format_inr(m['mae']))
        col2.metric('R² Score', f"{m['r2']:.3f}")
        st.metric('RMSE', format_inr(m['rmse']))
        
        # Performance indicator
        r2_val = m['r2']
        if r2_val > 0.8:
            st.success("✅ Excellent accuracy")
        elif r2_val > 0.6:
            st.info("ℹ️ Good accuracy")
        else:
            st.warning("⚠️ Fair accuracy")
    
    st.markdown('---')
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button('🎲 Try Sample Prediction', use_container_width=True):
        if model and df is not None:
            sample = {
                'Area': df['Area'].median() if 'Area' in df.columns else 1000,
                'No. of Bedrooms': int(df['No. of Bedrooms'].median()) if 'No. of Bedrooms' in df.columns else 2,
                'City': df['City'].mode()[0] if 'City' in df.columns and not df['City'].mode().empty else 'Hyderabad'
            }
            for a in amen_cols:
                sample[a] = int(bool(df[a].median() >= 0.5)) if a in df.columns else 0
            
            sample_df = pd.DataFrame([sample])
            try:
                pred = model.predict(sample_df)
                st.success(f'Sample prediction: {format_inr(pred[0])}')
            except Exception as e:
                st.error(f'Prediction failed: {e}')
    
    if st.button('🗑️ Clear History', use_container_width=True):
        st.session_state['pred_history'] = []
        st.success('History cleared!')
        st.rerun()
    
    # Export history
    if st.session_state['pred_history']:
        hist_df = pd.DataFrame(st.session_state['pred_history'])
        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            '📥 Download History',
            data=csv,
            file_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            use_container_width=True
        )

# Main Header
st.markdown("""
<div class='main-header'>
    <h1 style='margin:0; font-size:2.5rem'>🏡 House Price Predictor</h1>
    <p style='font-size:1.2rem; margin-top:1rem; opacity:0.95'>
        Get instant, AI-powered property valuations across India
    </p>
</div>
""", unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Predict Price", "📊 Analytics Dashboard", "📈 Price Comparison", "ℹ️ Help & Guide"])

# ============================================================================
# TAB 1: PREDICT PRICE
# ============================================================================
with tab1:
    if model is None:
        st.error("⚠️ No model loaded. Please select a model from the sidebar.")
    else:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("### 📝 Property Details")
            st.markdown("Fill in the property information below to get an instant price prediction.")
            
            # Basic inputs
            col1, col2 = st.columns(2)
            
            with col1:
                area = st.number_input(
                    '🏢 Area (sqft)',
                    value=int(get_median('Area', 1000)),
                    min_value=100,
                    max_value=50000,
                    step=50,
                    help="Enter the total built-up area in square feet"
                )
            
            with col2:
                bedrooms = st.number_input(
                    '🛏️ Bedrooms',
                    value=int(get_median('No. of Bedrooms', 2)),
                    min_value=0,
                    max_value=10,
                    step=1,
                    help="Number of bedrooms in the property"
                )
            
            # Area slider for quick adjustment
            if meta and 'feature_ranges' in meta and 'Area' in meta['feature_ranges']:
                ranges = meta['feature_ranges']['Area']
                area = st.slider(
                    '🎚️ Fine-tune Area',
                    min_value=int(ranges['min']),
                    max_value=int(ranges['max']),
                    value=int(area),
                    step=50
                )
            
            # Location/City selection
            st.markdown("### 📍 Location")
            
            if df is not None and 'Location' in df.columns:
                locations = sorted(df['Location'].dropna().unique().tolist())
                location = st.selectbox(
                    'Area/Locality',
                    options=[''] + locations,
                    help="Select the specific locality or area"
                )
            else:
                location = st.text_input('Location (optional)', '')
            
            if df is not None and 'City' in df.columns:
                cities = sorted(df['City'].dropna().unique().tolist())
                city = st.selectbox(
                    'City',
                    options=cities,
                    index=0 if cities else None,
                    help="Select the city"
                )
            else:
                city = st.text_input('City', value='Hyderabad')
            
            # Property features
            st.markdown("### 🏠 Property Features")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                resale = st.selectbox('🔄 Property Type', options=['New', 'Resale'], index=0)
            
            with col2:
                maintenance = st.selectbox('👷 Maintenance Staff', options=['Unknown', 'No', 'Yes'], index=0)
            
            with col3:
                parking = st.selectbox('🚗 Car Parking', options=['Unknown', 'No', 'Yes'], index=0)
            
            # Common amenities (prominent display)
            st.markdown("### ⭐ Key Amenities")
            
            amenity_dict = {}
            
            # Create grid for common amenities
            cols = st.columns(4)
            common_amenities = ['Gym', 'SwimmingPool', 'Club', 'Park', 'AC', 'Wifi', 'Intercom', 'GasConnection']
            
            for idx, amenity in enumerate(common_amenities):
                if amenity in amen_cols:
                    with cols[idx % 4]:
                        amenity_dict[amenity] = st.checkbox(
                            amenity.replace('SwimmingPool', 'Pool').replace('GasConnection', 'Gas'),
                            value=bool(df[amenity].median() >= 0.5) if amenity in df.columns else False
                        )
            
            # Advanced amenities in expander
            if amen_cols:
                other_amenities = [a for a in amen_cols if a not in common_amenities]
                if other_amenities:
                    with st.expander(f"🔧 Additional Amenities ({len(other_amenities)})"):
                        cols = st.columns(3)
                        for idx, amenity in enumerate(other_amenities):
                            with cols[idx % 3]:
                                amenity_dict[amenity] = st.checkbox(
                                    amenity,
                                    value=bool(df[amenity].median() >= 0.5) if amenity in df.columns else False,
                                    key=f"adv_{amenity}"
                                )
            
            # Prediction buttons
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                predict_btn = st.button('🔮 Predict Price', use_container_width=True, type="primary")
            
            with col2:
                compare_btn = st.button('📊 Compare', use_container_width=True)
            
            with col3:
                save_btn = st.button('💾 Save', use_container_width=True)
        
        with col_right:
            st.markdown("### 📊 Input Summary")
            
            summary_data = {
                'Property': [
                    f"{area:,} sqft",
                    f"{bedrooms} BHK",
                    city,
                    location if location else "Not specified",
                    resale
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Show amenities count
            selected_amenities = sum(amenity_dict.values())
            st.metric("✨ Selected Amenities", f"{selected_amenities}/{len(amen_cols)}")
            
            # Price range indicator
            if df is not None and 'Price' in df.columns:
                st.markdown("### 💰 Price Range in Dataset")
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=df['Price'],
                    name='Price Distribution',
                    marker_color='#667eea'
                ))
                fig.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Handle prediction
        if predict_btn:
            with st.spinner('🔮 Calculating property value...'):
                # Prepare input data
                input_data = {
                    'Area': int(area),
                    'No. of Bedrooms': int(bedrooms),
                    'City': city
                }
                
                if location:
                    input_data['Location'] = location
                
                # Map categorical features
                input_data['Resale'] = 1 if resale == 'Resale' else 0
                
                ms_map = {'Unknown': 2, 'No': 0, 'Yes': 1}
                input_data['MaintenanceStaff'] = ms_map.get(maintenance, 2)
                
                park_map = {'Unknown': 2, 'No': 0, 'Yes': 1}
                input_data['CarParking'] = park_map.get(parking, 2)
                
                # Add all amenities
                for amenity in amen_cols:
                    input_data[amenity] = int(amenity_dict.get(amenity, False))
                
                input_df = pd.DataFrame([input_data])
                
                try:
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    
                    # Display result
                    st.markdown(f"""
                    <div class='prediction-card'>
                        <h3 style='margin:0'>Predicted Price</h3>
                        <div class='prediction-price'>{format_inr(prediction)}</div>
                        <p style='opacity:0.9'>Based on current market trends and property features</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional metrics
                    col1, col2, col3 = st.columns(3)
                    
                    price_per_sqft = prediction / area
                    col1.metric("💵 Price per Sqft", format_inr(price_per_sqft))
                    
                    if df is not None and 'Price' in df.columns:
                        median_price = df['Price'].median()
                        diff_pct = ((prediction - median_price) / median_price) * 100
                        col2.metric("📊 vs Market Median", f"{diff_pct:+.1f}%")
                        
                        # Percentile
                        percentile = (df['Price'] < prediction).sum() / len(df) * 100
                        col3.metric("📈 Market Percentile", f"{percentile:.0f}th")
                    
                    # Save to history
                    hist_entry = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'price': float(prediction),
                        'area': int(area),
                        'bedrooms': int(bedrooms),
                        'city': city,
                        'location': location,
                        'price_per_sqft': float(price_per_sqft)
                    }
                    st.session_state['pred_history'].insert(0, hist_entry)
                    
                    # Show similar properties
                    if df is not None and len(df) > 0:
                        st.markdown("### 🏘️ Similar Properties")
                        
                        # Filter similar properties
                        similar = df[
                            (df['Area'] >= area * 0.8) & 
                            (df['Area'] <= area * 1.2) &
                            (df['No. of Bedrooms'] == bedrooms)
                        ]
                        
                        if len(similar) > 0:
                            st.write(f"Found {len(similar)} similar properties")
                            
                            # Show sample
                            sample_similar = similar.sample(min(5, len(similar)))
                            for idx, row in sample_similar.iterrows():
                                with st.expander(f"📍 {row.get('Location', 'Unknown')} - {format_inr(row['Price'])}"):
                                    col1, col2, col3 = st.columns(3)
                                    col1.write(f"**Area:** {row['Area']:.0f} sqft")
                                    col2.write(f"**Bedrooms:** {int(row['No. of Bedrooms'])}")
                                    col3.write(f"**Price/sqft:** {format_inr(row['Price']/row['Area'])}")
                
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("Please check your input values and try again.")

# ============================================================================
# TAB 2: ANALYTICS DASHBOARD  
# ============================================================================
with tab2:
    if df is None:
        st.warning("📊 No data available. Please load a model from the sidebar.")
    else:
        st.markdown("### 📊 Market Analytics Dashboard")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = df['Price'].mean()
            st.metric("💰 Average Price", format_inr(avg_price))
        
        with col2:
            avg_area = df['Area'].mean()
            st.metric("🏢 Average Area", f"{avg_area:.0f} sqft")
        
        with col3:
            avg_price_sqft = (df['Price'] / df['Area']).mean()
            st.metric("💵 Avg Price/Sqft", format_inr(avg_price_sqft))
        
        with col4:
            avg_bedrooms = df['No. of Bedrooms'].mean()
            st.metric("🛏️ Avg Bedrooms", f"{avg_bedrooms:.1f}")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Price Distribution")
            fig = px.histogram(
                df, 
                x='Price', 
                nbins=50,
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🏢 Area Distribution")
            fig = px.histogram(
                df,
                x='Area',
                nbins=50,
                color_discrete_sequence=['#764ba2']
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # City-wise analysis
        if 'City' in df.columns:
            st.markdown("#### 🌆 City-wise Analysis")
            
            city_stats = df.groupby('City').agg({
                'Price': ['mean', 'median', 'count'],
                'Area': 'mean'
            }).round(0)
            
            city_stats.columns = ['Avg Price', 'Median Price', 'Count', 'Avg Area']
            city_stats = city_stats.sort_values('Median Price', ascending=False)
            
            # Bar chart
            fig = px.bar(
                city_stats.reset_index(),
                x='City',
                y='Median Price',
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.dataframe(city_stats, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("#### 🔥 Feature Correlation")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c in ['Price', 'Area', 'No. of Bedrooms']]
        
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3: PRICE COMPARISON
# ============================================================================
with tab3:
    st.markdown("### 📈 Price Comparison Tool")
    st.write("Compare predictions side-by-side to make informed decisions")
    
    if not st.session_state['pred_history']:
        st.info("📝 Make some predictions first to use the comparison tool!")
    else:
        # Show history
        st.markdown("#### 🕐 Recent Predictions")
        
        hist_df = pd.DataFrame(st.session_state['pred_history'][:10])
        
        # Format for display
        if not hist_df.empty:
            display_df = hist_df.copy()
            display_df['price'] = display_df['price'].apply(format_inr)
            display_df['price_per_sqft'] = display_df['price_per_sqft'].apply(format_inr)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Price trend chart
            if len(hist_df) > 1:
                st.markdown("#### 📊 Prediction Trend")
                
                fig = px.line(
                    hist_df,
                    y='price',
                    markers=True,
                    labels={'index': 'Prediction #', 'price': 'Price (₹)'}
                )
                fig.update_traces(line_color='#667eea', line_width=3)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Comparison selector
            st.markdown("#### ⚖️ Compare Predictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                idx1 = st.selectbox('Select first prediction', range(len(hist_df)), format_func=lambda x: f"Prediction {x+1}")
            
            with col2:
                idx2 = st.selectbox('Select second prediction', range(len(hist_df)), index=min(1, len(hist_df)-1), format_func=lambda x: f"Prediction {x+1}")
            
            if idx1 != idx2:
                pred1 = hist_df.iloc[idx1]
                pred2 = hist_df.iloc[idx2]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>Prediction {idx1+1}</h4>
                        <h2>{format_inr(pred1['price'])}</h2>
                        <p>{pred1['area']} sqft • {pred1['bedrooms']} BHK</p>
                        <p>{pred1['city']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    price_diff = pred2['price'] - pred1['price']
                    pct_diff = (price_diff / pred1['price']) * 100
                    
                    st.markdown(f"""
                    <div class='metric-card' style='text-align:center'>
                        <h4>Difference</h4>
                        <h2 style='color: {"#4caf50" if price_diff > 0 else "#f44336"}'>{format_inr(abs(price_diff))}</h2>
                        <p style='font-size:1.2rem'>{pct_diff:+.1f}%</p>
                        <p>{"Higher" if price_diff > 0 else "Lower"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>Prediction {idx2+1}</h4>
                        <h2>{format_inr(pred2['price'])}</h2>
                        <p>{pred2['area']} sqft • {pred2['bedrooms']} BHK</p>
                        <p>{pred2['city']}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ============================================================================
# TAB 4: HELP & GUIDE
# ============================================================================
with tab4:
    st.markdown("### ℹ️ How to Use This App")
    
    st.markdown("""
    <div class='info-box'>
        <h4>🚀 Getting Started</h4>
        <ol>
            <li><strong>Select a Model:</strong> Choose a trained model from the sidebar</li>
            <li><strong>Enter Property Details:</strong> Fill in area, bedrooms, location, and amenities</li>
            <li><strong>Get Prediction:</strong> Click "Predict Price" to get instant valuation</li>
            <li><strong>Analyze Results:</strong> View similar properties and market insights</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📊 Understanding Results
        
        - **Predicted Price**: AI-generated property valuation
        - **Price per Sqft**: Cost efficiency metric
        - **Market Percentile**: Where your property stands
        - **Similar Properties**: Comparable listings
        """)
        
        st.markdown("""
        #### 🎯 Tips for Accuracy
        
        - Provide accurate area measurements
        - Select correct location/city
        - Include all available amenities
        - Use recent model for best results
        """)
    
    with col2:
        st.markdown("""
        #### 🔧 Features
        
        - **Real-time Predictions**: Instant price estimates
        - **Interactive Maps**: Visual location selection
        - **History Tracking**: Save and compare predictions
        - **Analytics Dashboard**: Market insights and trends
        - **Export Data**: Download prediction history
        """)
        
        st.markdown("""
        #### ❓ FAQs
        
        **Q: How accurate are predictions?**  
        A: Check R² score in sidebar (>0.8 = excellent)
        
        **Q: Can I save predictions?**  
        A: Yes! Use the "Save" button and export history
        
        **Q: Which cities are supported?**  
        A: Check the sidebar for available locations
        """)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='success-box'>
        <h4>💡 Need Help?</h4>
        <p>Visit our <a href='https://github.com/Saikumarlingaraju/House-Price-Predictor'>GitHub Repository</a> for:</p>
        <ul>
            <li>Detailed documentation</li>
            <li>Training new models</li>
            <li>Reporting issues</li>
            <li>Contributing to the project</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Model info
    if meta:
        st.markdown("### 🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Algorithm:**", "Random Forest Regressor")
            if 'n_estimators' in meta:
                st.write("**Trees:**", meta['n_estimators'])
            if 'state' in meta:
                st.write("**Trained on:**", meta['state'])
        
        with col2:
            if 'features' in meta:
                st.write("**Features used:**", len(meta['features']))
            if 'metrics' in meta:
                st.write("**Training R²:**", f"{meta['metrics']['r2']:.4f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; padding:2rem 0'>
    <p>🏡 House Price Predictor v2.0 | Powered by Machine Learning</p>
    <p>Made with ❤️ for smarter real estate decisions</p>
</div>
""", unsafe_allow_html=True)
