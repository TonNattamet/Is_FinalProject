import streamlit as st

def show():
    st.header("📊 Introduction & Data Preparation")
    
    st.markdown("""
    ## เนื้อหาภายในเว็บไซต์
    - Introduction & Data Preparation.
    - 1. Ml Bakery Health Classifier.
    - 2. Ml Thai Dish Predictor.
    - 3. NN LineTrain_Predict.
    
    ## ระบุที่มาของ Dataset
    - **1. Ml Bakery Health Classifier:**  
      [Kaggle Dataset](https://www.kaggle.com/datasets/sandeep1080/baked-food-nutritions-check-are-you-healthy)
      
    - **2. Ml Thai Dish Predictor:**  
      [Kaggle Dataset](https://www.kaggle.com/datasets/ponthakornsodchun/foods-in-thailand)
      
    - **3. NN LineTrain_Predict:**  
      [Kaggle Dataset](https://www.kaggle.com/datasets/gusbell/thailand-public-train-data-bangkok-area)

    ## อธิบาย Feature ของแต่ละ Dataset
    ### **1. Ml Bakery Health Classifier**
    - ใช้ข้อมูล **โภชนาการของขนมอบ** เพื่อทำนายว่าสุขภาพดีหรือไม่ 🥐✅❌

    ### **2. Ml Thai Dish Predictor**
    - ใช้วัตถุดิบ **เพื่อทำนายชื่ออาหารไทย** 🍜🇹🇭

    ### **3. NN LineTrain_Predict**
    - ใช้พิกัด **เพื่อทำนายสายรถไฟที่ใกล้ที่สุด** 🚆📍
    """)

show()
