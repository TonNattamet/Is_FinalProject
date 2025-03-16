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
    
    ---

    ## อธิบาย Feature ของแต่ละ Dataset

    ### **1. Ml Bakery Health Classifier**
    - **Feature:**
      - `Calories-kcal` : จำนวนแคลอรี่ของขนมอบ
      - `Protein-g` : ปริมาณโปรตีน (กรัม)
      - `Carbohydrates-g` : ปริมาณคาร์โบไฮเดรต (กรัม)
      - `Sugar-g` : ปริมาณน้ำตาล (กรัม)
      - `SaturatedFat-g` : ปริมาณไขมันอิ่มตัว (กรัม)
    - **AI ใช้ข้อมูลนี้เพื่อทำนาย:** ขนมอบนั้น **สุขภาพดีหรือไม่** 🥐✅❌
    
    ---

    ### **2. Ml Thai Dish Predictor**
    - **Feature:**
      - `ingredients` : ส่วนผสมอาหาร (เช่น "Beef+lime juice+fish sauce")
      - `th_name` : ชื่ออาหารในภาษาไทย
    - **AI ใช้ข้อมูลนี้เพื่อทำนาย:** ชื่อเมนูอาหารไทยจากวัตถุดิบที่ป้อนเข้า 🍜🇹🇭

    ---

    ### **3. NN LineTrain_Predict**
    - **Feature:**
      - `stationId` : รหัสสถานี
      - `nameEng` : ชื่อสถานี (อังกฤษ)
      - `geoLat` / `geoLng` : ตำแหน่งพิกัดของสถานี (ละติจูด / ลองจิจูด)
      - `lineNameEng` : ชื่อสายรถไฟ (เช่น **BTS Sukhumvit Line**)
      - `lineColorHex` : สีของสายรถไฟ
      - `lineServiceName` : ประเภทของบริการ (เช่น **BTS, MRT, SRT, ARL**)
    - **AI ใช้ข้อมูลนี้เพื่อทำนาย:** เส้นทางรถไฟที่ใกล้ที่สุดจากตำแหน่งที่เลือกบนแผนที่ 🚆📍
    
    ---
    """)