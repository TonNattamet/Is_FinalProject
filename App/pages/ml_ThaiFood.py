import streamlit as st
import joblib
import numpy as np
from gensim.models import Word2Vec

# Load trained models
model = joblib.load("model/thai_dish_model_final.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
word2vec = Word2Vec.load("model/word2vec_model.pkl")

def show():
    st.header("🍛 Thai Dish Predictor")
    st.markdown("### Enter ingredients to predict the Thai dish name!")
    
    # Sidebar for additional info
    st.sidebar.header("ℹ️ How to Use")
    st.sidebar.markdown("Enter ingredients separated by `+` (e.g., `beef+lime juice+fish sauce`).")
    st.sidebar.markdown("Click the **Predict Dish** button to get the predicted Thai dish name.")
    st.sidebar.markdown("Enjoy discovering Thai cuisine! 🇹🇭")
    
    # Ingredient List
    available_ingredients = [
        "Beef", "Pork", "Chicken", "Shrimp", "Rice noodles", "Fish sauce", "Garlic", "Chili", "Soy sauce", "Lime juice",
        "Coconut milk", "Eggs", "Vegetables", "Herbs", "Rice", "Peanuts", "Curry paste", "Tamarind paste", "Oyster sauce",
        "Tofu", "Mushrooms", "Palm sugar", "Turmeric", "Ginger", "Holy basil", "Green beans", "Sweet and sour sauce",
        "Pandan leaves", "Cassava", "Pickled mustard greens", "Jackfruit", "Pork belly", "Duck", "Blood tofu", "Morning glory"
    ]
    
    # Display available ingredients
    st.markdown("#### 🍽️ Available Ingredients")
    st.write(", ".join(available_ingredients))
    
    # User Input
    ingredients_input = st.text_input("Enter Ingredients (e.g., beef+lime juice+fish sauce)", "")
    
    # Function to Convert Ingredients to Word2Vec Features
    def vectorize_ingredients(ingredients):
        words = ingredients.split("+")
        vectors = [word2vec.wv[word] for word in words if word in word2vec.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)  # Ensure 100-dimensional output
    
    # Make Prediction
    if st.button("🔍 Predict Dish"):
        if ingredients_input:
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
show()