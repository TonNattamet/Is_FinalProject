import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import tensorflow as tf
import joblib
import numpy as np

# Load trained model
model = tf.keras.models.load_model("model/bangkok_train_nn_final.keras")

# Load encoders & scalers
label_encoder = joblib.load("model/station_label_encoder_final.pkl")
scaler = joblib.load("model/lat_lng_scaler_final.pkl")

# Initialize Geocoder
geolocator = Nominatim(user_agent="geoapi")

# UI Setup
st.set_page_config(page_title="🚆 Bangkok Train AI", page_icon="🚇", layout="centered")
st.title("🚆 Bangkok Train Line Predictor")
st.markdown("### Select your location on the map or enter an address to predict the nearest train line.")

# Default Location: Bangkok
default_location = [13.7563, 100.5018]

# Function to get Lat/Lng from Address
def get_lat_lng(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except:
        return None

# Address Input
address = st.text_input("📍 Enter an address (Optional):", "")

if address:
    coords = get_lat_lng(address)
    if coords:
        st.success(f"✅ Found Location: {coords}")
        default_location = [coords[0], coords[1]]  # Update map location
    else:
        st.error("❌ Address not found. Try another location.")

# Create Folium Map
m = folium.Map(location=default_location, zoom_start=12)

# Add **Draggable Marker**
marker = folium.Marker(default_location, draggable=True)
marker.add_to(m)

# Capture Map Interactions
map_data = st_folium(m, height=400, width=700)

# **FIXED: Check for User Clicks or Marker Moves**
if map_data is not None:
    if "last_clicked" in map_data and map_data["last_clicked"] is not None:
        lat, lng = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
    elif "last_object_clicked" in map_data and map_data["last_object_clicked"] is not None:
        lat, lng = map_data["last_object_clicked"]["lat"], map_data["last_object_clicked"]["lng"]
    else:
        lat, lng = default_location  # Use default values if no interaction

# **Latitude/Longitude Inputs (Now Updates from Map)**
lat = st.number_input("🌍 Latitude:", value=lat, format="%.6f")
lng = st.number_input("📍 Longitude:", value=lng, format="%.6f")

# Predict Button
if st.button("🔍 Predict Train Line"):
    # Normalize Input
    input_data = scaler.transform([[lat, lng]])

    # Predict Train Line
    prediction_probs = model.predict(input_data)
    predicted_index = np.argmax(prediction_probs)  # Get the highest probability class
    predicted_line = label_encoder.inverse_transform([predicted_index])[0]

    st.success(f"🚇 Nearest Train Line: **{predicted_line}**")
