import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Load the DataFrame if needed
df = pickle.load(open('df.pkl', 'rb'))

# Set the background color and title
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        padding: 20px;
    }
    .title {
        text-align: center;
        font-size: 40px;
        color: #007BFF;
        margin-bottom: 20px;
    }
    .subheader {
        text-align: center;
        font-size: 24px;
        color: #6c757d;
        margin-bottom: 40px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #495057;
        margin-top: 20px;
    }
    .disclaimer {
        text-align: center;
        font-size: 16px;
        color: #dc3545;  /* Bootstrap danger color for emphasis */
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Streamlit app title
st.markdown('<h1 class="title">🏡 House Price Prediction App</h1>', unsafe_allow_html=True)

# Add a disclaimer message
st.markdown("""
    <div style='background-color: #f7f7f7; padding: 10px; border-radius: 5px;'>
        <h5 style='color: #333;'>Disclaimer:</h5>
        <p style='color: #555;'>The predictions provided by this model are based on a limited dataset. 
        For more accurate and reliable results, a larger and more diverse dataset is necessary. 
        As the volume and variety of data increase, so does the potential for enhanced model accuracy. 
        Please consider this while interpreting the predictions.</p>
    </div>
""", unsafe_allow_html=True)


# Input fields with descriptions and sample values
st.markdown('<h2 class="subheader">Enter the details below:</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    average_area_income = st.number_input(
        "💰 Average Area Income ($)",
        value=50000,
        help="Enter the average income in the area in dollars."
    )
    average_age_of_house = st.number_input(
        "🏠 Average Age of House (years)",
        value=10,
        help="Enter the average age of houses in the area in years."
    )

with col2:
    number_of_rooms = st.number_input(
        "🛏️ Number of Rooms",
        value=7,
        help="Enter the average number of rooms in the houses."
    )
    number_of_bedrooms = st.number_input(
        "🛌 Number of Bedrooms",
        value=3,
        help="Enter the average number of bedrooms in the houses."
    )
    area_population = st.number_input(
        "👥 Area Population",
        value=100000,
        help="Enter the total population in the area."
    )

# Button to predict
if st.button("🔍 Predict"):
    # Prepare input data for prediction
    input_data = np.array([[average_area_income, average_age_of_house, 
                             number_of_rooms, number_of_bedrooms, 
                             area_population]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display prediction result
    st.subheader("✨ Prediction Result")
    st.write(f"The predicted price for the house is: **${prediction[0]:,.2f}**")
    
# Footer with developer name
st.markdown('<div class="footer">Developed by - Harshal Kumawat</div>', unsafe_allow_html=True)
