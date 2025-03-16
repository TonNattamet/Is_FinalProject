import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_ml_model():
    return joblib.load("model/baked_food_rf_final_v3.pkl")  # Load the final model

# Load trained model
model = load_ml_model()

# Define the MinMaxScaler (must match training range)
scaler = MinMaxScaler()
scaler.fit([[50, 0, 0, 0, 0], [600, 50, 100, 100, 30]])  # Adjust min/max range

def show():
    st.header("🍞 Baked Food Health Classifier")
    st.markdown("### Enter nutritional values to predict if the food is **Healthy** or **Unhealthy**")
    
    # Sidebar for additional info
    st.sidebar.header("⚙️ Adjust Model Settings")
    st.sidebar.markdown("Use the sliders below to explore different values.")
    
    # User Inputs with sliders
    calories = st.slider("Calories (kcal)", 50, 600, 250)
    protein = st.slider("Protein (g)", 0.0, 50.0, 15.0)
    carbs = st.slider("Carbohydrates (g)", 0.0, 100.0, 30.0)
    sugar = st.slider("Sugar (g)", 0.0, 100.0, 3.0)
    saturated_fat = st.slider("Saturated Fat (g)", 0.0, 30.0, 2.0)
    
    # Make Prediction
    if st.button("🔍 Predict Healthiness"):
        features = np.array([[calories, protein, carbs, sugar, saturated_fat]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        label = "✅ Healthy" if prediction[0] == 1 else "❌ Unhealthy"
        st.success(f"Predicted Classification: {label}")
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 Developed with Streamlit & Machine Learning")


show()