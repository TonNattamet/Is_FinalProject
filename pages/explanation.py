import streamlit as st

def show():
    st.header("🤖 Machine Learning & Neural Network Development")

    st.markdown("""
    ## 🚀 การพัฒนาโมเดล Machine Learning และ Neural Network
    
    ### **1. Ml Bakery Health Classifier**
    - **Algorithm Used:** Random Forest Classifier  
    - **Steps:**  
      1. Data Cleaning & Handling Missing Values  
      2. Feature Selection (Calories, Protein, Carbs, Sugar, Saturated Fat)  
      3. Model Training (Random Forest)  
      4. Hyperparameter Tuning & Accuracy Improvement  

    ### **2. Ml Thai Dish Predictor**
    - **Algorithm Used:** Word2Vec + XGBoost  
    - **Steps:**  
      1. Convert ingredients into vectors using Word2Vec  
      2. Train XGBoost model for classification  
      3. Fine-tune model for better predictions  

    ### **3. NN LineTrain_Predict**
    - **Algorithm Used:** Deep Neural Network (TensorFlow)  
    - **Steps:**  
      1. Preprocess Latitude & Longitude  
      2. Train a neural network with multiple layers  
      3. Improve accuracy with better architecture & data balancing  

    ### **Key Improvements**
    - Used **data normalization** for better ML & NN performance  
    - Added **oversampling** to balance training data  
    - Used **Batch Normalization & Dropout** in NN to prevent overfitting  
    """)

show()
