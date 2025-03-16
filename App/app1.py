import streamlit as st
import joblib
import numpy as np
from gensim.models import Word2Vec

# Load trained models
model = joblib.load("model/thai_dish_model_final.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
word2vec = Word2Vec.load("model/word2vec_model.pkl")

# UI Enhancements
st.set_page_config(page_title="Thai Dish Predictor", page_icon="🍛", layout="centered")
st.title("🍛 Thai Dish Predictor")
st.markdown("### Enter ingredients to predict the Thai dish name!")

# Sidebar for additional info
st.sidebar.header("ℹ️ How to Use")
st.sidebar.markdown("Enter a list of ingredients separated by `+` (e.g., `beef+lime juice+fish sauce`).")
st.sidebar.markdown("Click the **Predict Dish** button to get the predicted Thai dish name.")
st.sidebar.markdown("Enjoy discovering Thai cuisine! 🇹🇭")

# Function to Convert Ingredients to Word2Vec Features
def vectorize_ingredients(ingredients):
    words = ingredients.split("+")
    vectors = [word2vec.wv[word] for word in words if word in word2vec.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)  # Ensure 100-dimensional output

# User Input
st.markdown("#### 🍽️ Input Ingredients")
ingredients_input = st.text_input("Enter Ingredients (e.g., beef+lime juice+fish sauce)", "")

# Make Prediction
if st.button("🔍 Predict Dish", use_container_width=True):
    if ingredients_input:
        # Convert user input into Word2Vec format
        input_vector = vectorize_ingredients(ingredients_input).reshape(1, -1)
        
        # Predict dish
        prediction = model.predict(input_vector)
        
        # Convert numeric prediction back to dish name
        predicted_dish = label_encoder.inverse_transform(prediction)
        
        st.success(f"🍽️ Predicted Thai Dish: **{predicted_dish[0]}**")
    else:
        st.warning("⚠️ Please enter some ingredients before predicting.")

# Footer
st.markdown("---")
st.markdown("🚀 Developed with Streamlit & Machine Learning | Thai Cuisine AI 🍜")
