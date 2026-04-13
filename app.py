import streamlit as st
import pandas as pd
import joblib
import numpy as np


# -----------------------
# 1. Load Model & Scaler
# -----------------------
@st.cache_resource
def load_assets():
    # Loading the tuned XGBoost model and the scaler saved from Colab
    model = joblib.load("best_xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("Ensure 'best_xgb_model.pkl' and 'scaler.pkl' are in your repository.")

# -----------------------
# 2. Page Configuration
# -----------------------
st.set_page_config(page_title="Pro House Predictor", page_icon="🏡", layout="wide")

# -----------------------
# 3. Custom CSS for Branding
# -----------------------
st.markdown(
    """
    <style>
        .main { background-color: #0e1117; }
        .title-text { font-size: 38px; font-weight: bold; color: #4CAF50; }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            width: 100%;
            height: 3em;
            font-weight: bold;
        }
        .result-card {
            padding: 30px;
            background-color: #1e2130;
            border-radius: 15px;
            border-left: 5px solid #4CAF50;
            text-align: center;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# -----------------------
# 4. Sidebar Information
# -----------------------
with st.sidebar:
    st.title("📊 Model Insights")
    st.markdown(
        """
    **Algorithm:** Tuned XGBoost  
    **Objective:** Predict Market Value  
    **Data Source:** Portfolio Dataset  
    
    This app uses a Gradient Boosting approach to capture non-linear relationships in real estate data.
    """
    )
    st.divider()
    st.write("Built by [Murad Amin](https://github.com/Muradamen)")

# -----------------------
# 5. Header Section
# -----------------------
st.markdown(
    '<p class="title-text">🏡 House Price Prediction App</p>', unsafe_allow_html=True
)
st.write(
    "Adjust the property and neighborhood parameters below to generate an AI-powered price estimate."
)
st.divider()

# -----------------------
# 6. Input Layout (2 Columns)
# -----------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location & Neighborhood")
    dist_center = st.slider("Distance to City Center (km)", 0.5, 50.0, 10.0)
    crime_rate = st.slider("Crime Rate Index (0-100)", 0, 100, 25)
    schools = st.number_input("Nearby Schools", 0, 20, 5)
    noise = st.select_slider("Noise Level (0-10)", options=list(range(11)), value=4)
    income = st.number_input("Median Monthly Income ($)", 500, 20000, 4500)

with col2:
    st.subheader("🏠 Property Details")
    lot_size = st.number_input("Lot Size (m²)", 100, 15000, 1200)
    beds = st.slider("Bedrooms", 1, 10, 3)
    baths = st.slider("Bathrooms", 1, 8, 2)
    age = st.number_input("House Age (Years)", 0, 150, 20)
    energy = st.slider("Energy Efficiency Score", 0, 100, 65)
    garden = st.number_input("Garden Size (m²)", 0, 2000, 100)
    renovated = st.checkbox("Recently Renovated?")

# -----------------------
# 7. Prediction Logic
# -----------------------
st.divider()

if st.button("🔍 Run Valuation Analysis"):
    # Organize features in the exact order the model expects
    features = pd.DataFrame(
        [
            [
                lot_size,
                beds,
                baths,
                age,
                dist_center,
                crime_rate,
                schools,
                income,
                int(renovated),
                energy,
                garden,
                noise,
            ]
        ],
        columns=[
            "Lot_Size",
            "Bedrooms",
            "Bathrooms",
            "House_Age",
            "Distance_to_CityCenter",
            "Crime_Rate",
            "Nearby_Schools",
            "Monthly_Income",
            "Renovated",
            "Energy_Efficiency_Score",
            "Garden_Size",
            "Noise_Level",
        ],
    )

    # Pre-processing
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        # Displaying the Result
        st.markdown(
            f"""
            <div class="result-card">
                <h3 style="color: #aaaaaa;">Estimated Property Value</h3>
                <h1 style="color: #4CAF50; font-size: 55px;">${prediction:,.2f}k</h1>
                <p style="color: #777777;">Based on current neighborhood trends and property condition.</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.balloons()

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# -----------------------
# 8. Footer
# -----------------------
st.markdown(
    "<br><hr><center>Portfolio Project | MiraTech Solutions</center>",
    unsafe_allow_html=True,
)
