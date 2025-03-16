import streamlit as st

def show():
    st.header("📊 Introduction & Data Preparation")
    
    st.markdown("""
    ## Overview
    This application demonstrates two machine learning models: 
    1. **Baked Food Health Classifier** - Determines if a food is **Healthy or Unhealthy** based on nutritional values.
    2. **Thai Dish Predictor** - Predicts a Thai dish based on provided ingredients using a Neural Network.
    
    ## Data Collection & Preprocessing
    - The data was collected from food databases and preprocessed to ensure consistency.
    - Key features such as **calories, protein, carbohydrates, sugar, and fat** were selected for the classifier.
    - Ingredient-based predictions utilize **Word2Vec embeddings** to transform text into numerical features.
    
    ## Feature Engineering
    - The nutritional data is **normalized** using MinMaxScaler.
    - Thai dish ingredients are converted into **vector embeddings** using Word2Vec.
    """)
