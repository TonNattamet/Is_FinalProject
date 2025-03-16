import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib
import numpy as np
import tensorflow as tf
from geopy.geocoders import Nominatim

def show():
    # Load trained model
    model = tf.keras.models.load_model("model/bangkok_train_nn_final.keras")

    # Load encoders & scalers
    label_encoder = joblib.load("model/station_label_encoder_final.pkl")
    scaler = joblib.load("model/lat_lng_scaler_final.pkl")

    # Initialize Geocoder
    geolocator = Nominatim(user_agent="geoapi")

    # UI Setup
    st.title("🚆 Bangkok Train Line Predictor")
    st.markdown("### Select your location on the map or enter an address to predict the nearest train line.")
    st.markdown("### 🔹 Click on the **Blue Pin** after moving it to refresh latitude & longitude values.")

    # Default Location: Bangkok
    default_location = [13.7563, 100.5018]

    # Function to get Lat/Lng from Address
    def get_lat_lng(address):
        try:
            location = geolocator.geocode(address)
            if location:
                return location.latitude, location.longitude
        except Exception as e:
            st.error(f"⚠️ Error fetching location: {e}")
            return None

    # Address Input
    address = st.text_input("📍 Enter an address (Optional):", "", key="train_address_input")

    if address:
        coords = get_lat_lng(address)
        if coords:
            st.success(f"✅ Found Location: {coords}")
            default_location = [coords[0], coords[1]]  # Update map location
        else:
            st.error("❌ Address not found. Try another location.")

    # Initialize lat/lng from the default or fetched location
    lat, lng = default_location

    # Create Folium Map
    m = folium.Map(location=[lat, lng], zoom_start=12)

    # Add **Draggable Marker**
    marker = folium.Marker([lat, lng], draggable=True)
    marker.add_to(m)

    # Capture Map Interactions
    map_data = st_folium(m, height=400, width=700)

    # **Fix: Properly Capture User's New Location**
    if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
        lat, lng = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
    elif map_data and "last_object_clicked" in map_data and map_data["last_object_clicked"]:
        lat, lng = map_data["last_object_clicked"]["lat"], map_data["last_object_clicked"]["lng"]

    # **Latitude/Longitude Inputs (Now Updates from Map)**
    lat = st.number_input("🌍 Latitude:", value=float(lat), format="%.6f", key="lat_input")
    lng = st.number_input("📍 Longitude:", value=float(lng), format="%.6f", key="lng_input")

    # Predict Button
    if st.button("🔍 Predict Train Line"):
        # Normalize Input
        input_data = scaler.transform([[lat, lng]])

        # Predict Train Line
        prediction_probs = model.predict(input_data)
        predicted_index = np.argmax(prediction_probs)  # Get the highest probability class
        predicted_line = label_encoder.inverse_transform([predicted_index])[0]

        st.success(f"🚇 Nearest Train Line: **{predicted_line}**")

# Run the function to show UI
show()
