# import streamlit as st

# # Set page config at the start
# st.set_page_config(page_title="Wel", page_icon="🍽️", layout="wide")

# # Sidebar Navigation
# st.sidebar.title("📌 Navigation")
# page = st.sidebar.radio("Select Page", [
#     "Introduction & Data Preparation",
#     "1.Ml Bakery Health Classifier",
#     "2.Ml Thai Dish Predictor",
#     "3.NN Tic Tac Toe",
# ])

# # Import respective pages
# if page == "Introduction & Data Preparation":
#     import pages.introduction as intro
#     intro.show()

# elif page == "3.NN LineTrain_Predict":
#     import pages.Nn_LineTrain_Predict as Nn_LineTrain_Predict
#     Nn_LineTrain_Predict.show()

# elif page == "1.Ml Bakery Health Classifier":
#     import pages.ml_Bakery_Nutrition as ml_Bakerynutrition
#     ml_Bakerynutrition.show()

# elif page == "2.Ml Thai Dish Predictor":
#     import pages.ml_ThaiFood as ml_ThaiFood
#     ml_ThaiFood.show()




# import streamlit as st
# from pages.introduction import show as introduction
# from pages.explanation import show as explanation
# from pages.ml_ThaiFood import show as ml_ThaiFood
# from pages.ml_Bakery_Nutrition import show as ml_Bakery_Nutrition
# from pages.nn_LineTrain_Predict import show as nn_LineTrain_Predict

# # Sidebar Information
# st.sidebar.header("About Us")
# st.sidebar.markdown("**จัดทำโดย:** นายณัฐเมศร์ วงศ์ธนิตเลิศ (ห้อง DE-RA ชั้นปีที่ 4)")
# st.sidebar.markdown("**อาจารย์ผู้สอน:** ดร.ณัฐกิตติ์ จิตรเอื้อตระกูล (NJR)")
# st.sidebar.markdown("**มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ**")
# st.sidebar.markdown("**ภาควิชาวิทยาการคอมพิวเตอร์และสารสนเทศ**")

# # Page Navigation
# PAGES = {
#     "📊 Introduction": introduction,
#     "📖 ML & NN Explanation": explanation,
#     "🍜 ML Thai Dish Predictor": ml_ThaiFood,
#     "🥐 ML Bakery Health Classifier": ml_Bakery_Nutrition,
#     "🚆 NN Train Line Predictor": nn_LineTrain_Predict
# }

# st.sidebar.title("📌 Navigation")
# choice = st.sidebar.radio("🔍 Select a Page:", list(PAGES.keys()))
# PAGES[choice]()  # Call the selected page function





import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

# Load trained models
ml_bakery_model = joblib.load("model/baked_food_rf_final_v3.pkl")
ml_thai_food_model = joblib.load("model/thai_dish_model_final.pkl")
thai_food_label_encoder = joblib.load("model/thai_food_label_encoder.pkl")
word2vec = Word2Vec.load("model/word2vec_model.pkl")
nn_train_model = tf.keras.models.load_model("model/bangkok_train_nn_final.keras")
station_label_encoder = joblib.load("model/station_label_encoder_final.pkl")
scaler = joblib.load("model/lat_lng_scaler_final.pkl")

# Initialize Geocoder
geolocator = Nominatim(user_agent="geoapi")

# Set up Streamlit UI
st.sidebar.title("Navigation")
PAGES = ["📊 Introduction", "📖 Explanation", "🍲 ML Thai Dish Predictor", "🥐 ML Bakery Nutrition", "🚆 NN Train Line Predictor"]
choice = st.sidebar.radio("Go to", PAGES)

# --------------------- 📊 INTRODUCTION PAGE ---------------------
if choice == "📊 Introduction":
    st.header("📊 Introduction & Data Preparation")
    st.markdown("""
    ## รายการเนื้อหาในเว็บไซต์
    - Introduction & Data Preparation
    - 1. ML Bakery Health Classifier
    - 2. ML Thai Dish Predictor
    - 3. NN Train Line Predictor
    
    ### 📌 Dataset Sources:
    - **Bakery Health Classifier**: [Kaggle Dataset](https://www.kaggle.com/datasets/sandeep1080/baked-food-nutritions-check-are-you-healthy)
    - **Thai Dish Predictor**: [Kaggle Dataset](https://www.kaggle.com/datasets/ponthakornsodchun/foods-in-thailand)
    - **Train Line Predictor**: [Kaggle Dataset](https://www.kaggle.com/datasets/gusbell/thailand-public-train-data-bangkok-area)
    """)

# --------------------- 📖 EXPLANATION PAGE ---------------------
elif choice == "📖 Explanation":
    st.header("📖 Explanation: ML & NN Development")
    st.markdown("""
    ## ✨ Model Development
    - **ML Models**: Thai Dish Predictor, Bakery Nutrition Classifier
    - **Neural Network Model**: Train Line Predictor
    
    ### 🔹 Steps in Model Development:
    1. **Data Preprocessing**
    2. **Feature Engineering**
    3. **Training ML & NN Models**
    4. **Model Evaluation & Tuning**
    5. **Deploying Models on Streamlit**
    """)

# --------------------- 🍲 ML THAI DISH PREDICTOR ---------------------
elif choice == "🍲 ML Thai Dish Predictor":
    st.header("🍛 Thai Dish Predictor")
    st.markdown("### Enter ingredients to predict the Thai dish name!")

    available_ingredients = ["Beef", "Pork", "Chicken", "Shrimp", "Rice noodles", "Fish sauce", "Garlic", "Chili",
                             "Soy sauce", "Lime juice", "Coconut milk", "Eggs", "Vegetables", "Herbs", "Rice", 
                             "Peanuts", "Curry paste", "Tamarind paste", "Oyster sauce"]

    st.markdown("#### 🍽️ Available Ingredients")
    st.write(", ".join(available_ingredients))

    ingredients_input = st.text_input("Enter Ingredients (e.g., beef+lime juice+fish sauce)", "")

    if st.button("🔍 Predict Dish"):
        if ingredients_input:
            def vectorize_ingredients(ingredients):
                words = ingredients.split("+")
                vectors = [word2vec.wv[word] for word in words if word in word2vec.wv]
                return np.mean(vectors, axis=0) if vectors else np.zeros(100)

            input_vector = vectorize_ingredients(ingredients_input).reshape(1, -1)
            prediction = ml_thai_food_model.predict(input_vector)
            predicted_dish = thai_food_label_encoder.inverse_transform(prediction)
            st.success(f"🍽️ Predicted Thai Dish: **{predicted_dish[0]}**")
        else:
            st.warning("⚠️ Please enter some ingredients before predicting.")

# --------------------- 🥐 ML BAKERY NUTRITION CLASSIFIER ---------------------
elif choice == "🥐 ML Bakery Nutrition":
    st.header("🥐 Baked Food Nutrition Classifier")
    st.markdown("### Enter nutritional values to classify the baked food.")

    # User Inputs
    calories = st.number_input("Calories (kcal)", min_value=50, max_value=600, value=300)
    protein = st.number_input("Protein (g)", min_value=0.0, max_value=50.0, value=5.0)
    carbs = st.number_input("Carbohydrates (g)", min_value=0.0, max_value=100.0, value=50.0)
    sugar = st.number_input("Sugar (g)", min_value=0.0, max_value=100.0, value=10.0)
    saturated_fat = st.number_input("Saturated Fat (g)", min_value=0.0, max_value=30.0, value=5.0)

    if st.button("🔍 Predict Healthiness"):
        features = np.array([[calories, protein, carbs, sugar, saturated_fat]])
        prediction = ml_bakery_model.predict(features)
        label = "Healthy" if prediction[0] == 1 else "Unhealthy"
        st.success(f"Predicted Classification: {label}")

# --------------------- 🚆 NEURAL NETWORK TRAIN LINE PREDICTOR ---------------------
elif choice == "🚆 NN Train Line Predictor":
    st.header("🚆 Bangkok Train Line Predictor")
    st.markdown("### Select your location on the map or enter an address to predict the nearest train line.")

    # Default Location: Bangkok
    default_location = [13.7563, 100.5018]

    def get_lat_lng(address):
        try:
            location = geolocator.geocode(address)
            if location:
                return location.latitude, location.longitude
        except:
            return None

    address = st.text_input("📍 Enter an address (Optional):", "")
    if address:
        coords = get_lat_lng(address)
        if coords:
            st.success(f"✅ Found Location: {coords}")
            default_location = [coords[0], coords[1]]

    # Folium Map
    m = folium.Map(location=default_location, zoom_start=12)
    marker = folium.Marker(default_location, draggable=True).add_to(m)
    map_data = st_folium(m, height=400, width=700)

    if map_data is not None:
        if "last_clicked" in map_data and map_data["last_clicked"] is not None:
            lat, lng = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        else:
            lat, lng = default_location

    lat = st.number_input("🌍 Latitude:", value=lat, format="%.6f")
    lng = st.number_input("📍 Longitude:", value=lng, format="%.6f")

    if st.button("🔍 Predict Train Line"):
        input_data = scaler.transform([[lat, lng]])
        prediction_probs = nn_train_model.predict(input_data)
        predicted_index = np.argmax(prediction_probs)
        predicted_line = station_label_encoder.inverse_transform([predicted_index])[0]
        st.success(f"🚇 Nearest Train Line: **{predicted_line}**")










