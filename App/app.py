import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image

def load_ml_model():
    return joblib.load("ml_model.pkl")

def load_nn_model():
    return tf.keras.models.load_model("nn_model.h5")

st.sidebar.title("Machine Learning & Neural Network Web App")
page = st.sidebar.radio("Go to", ["Home", "Machine Learning Model", "Neural Network Model", "About the Project"])

if page == "Home":
    st.title("88888888888888888888888888")
    st.write("This application demonstrates Machine Learning and Neural Network models.")
    st.write("Select a page from the sidebar to explore more.")

elif page == "Machine Learning Model":
    st.title("House Price Prediction Model")
    st.write("Enter the features of the house to predict its price.")
    
    model = load_ml_model()
    
    lot_area = st.number_input("Lot Area (sqft)", min_value=500, max_value=50000, value=10000)
    overall_qual = st.slider("Overall Quality", 1, 10, 5)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000)
    total_rooms = st.number_input("Total Rooms", min_value=1, max_value=20, value=5)
    garage_cars = st.slider("Garage Cars", 0, 5, 1)
    
    if st.button("Predict Price"):
        features = np.array([[lot_area, overall_qual, year_built, total_rooms, garage_cars]])
        prediction = model.predict(features)
        st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

elif page == "Neural Network Model":
    st.title("Handwritten Digit Recognition")
    st.write("Upload an image of a handwritten digit (0-9) to classify it.")
    
    model = load_nn_model()
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
        prediction = np.argmax(model.predict(img_array))
        
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.success(f"Predicted Digit: {prediction}")

elif page == "About the Project":
    st.title("Project Overview")
    st.write("### Data Preparation")
    st.write("Explains data preprocessing for both datasets.")
    st.write("### Machine Learning Model")
    st.write("Details about the regression model used for house price prediction.")
    st.write("### Neural Network Model")
    st.write("Details about CNN used for digit recognition.")
