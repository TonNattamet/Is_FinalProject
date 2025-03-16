import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import folium
from streamlit_folium import st_folium
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from geopy.geocoders import Nominatim

# Set the page configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="AI Food & Train Predictor",
    page_icon="🍜",
    layout="wide"
)

# Sidebar Information
st.sidebar.header("About Us")
st.sidebar.markdown("**จัดทำโดย:** นายณัฐเมศร์ วงศ์ธนิตเลิศ (ห้อง DE-RA ชั้นปีที่ 4)")
st.sidebar.markdown("**อาจารย์ผู้สอน:** ดร.ณัฐกิตติ์ จิตรเอื้อตระกูล (NJR)")
st.sidebar.markdown("**มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ**")
st.sidebar.markdown("**ภาควิชาวิทยาการคอมพิวเตอร์และสารสนเทศ**")

# Navigation
PAGES = ["📊 Introduction", "🍜 ML Thai Dish Predictor", "🥐 ML Bakery Health Classifier", "🚆 NN Train Line Predictor"]
st.sidebar.title("📌 Navigation")
choice = st.sidebar.radio("🔍 Select a Page:", PAGES)


# ===========================
# 📊 Introduction Page
# ===========================
if choice == "📊 Introduction":
    st.title("📊 Welcome to AI Food & Train Predictor")
    st.markdown("""
        - 🍜 **ML Thai Dish Predictor**: Predict a Thai dish based on ingredients.
        - 🥐 **ML Bakery Health Classifier**: Determine if baked food is healthy or unhealthy.
        - 🚆 **NN Train Line Predictor**: Find the nearest train line in Bangkok based on location.
    """)
    # st.image("ai_food_train_banner.jpg", use_column_width=True)
    st.success("Select a page from the sidebar to get started!")


# ===========================
# 🍜 Thai Dish Predictor
# ===========================
if choice == "🍜 ML Thai Dish Predictor":
    st.header("🍛 Thai Dish Predictor")
    st.markdown("### Enter ingredients to predict the Thai dish name!")

    # Load Models
    @st.cache_resource
    def load_thai_food_models():
        return joblib.load("thai_dish_model_final.pkl"), joblib.load("label_encoder.pkl"), Word2Vec.load("word2vec_model.pkl")

    model, label_encoder, word2vec = load_thai_food_models()

    available_ingredients = ["Beef", "Pork", "Chicken", "Shrimp", "Rice noodles", "Fish sauce", "Garlic", "Chili", "Soy sauce", "Lime juice",
                            "Coconut milk", "Eggs", "Vegetables", "Herbs", "Rice", "Peanuts", "Curry paste", "Tamarind paste"]

    st.markdown("#### 🍽️ Available Ingredients")
    st.write(", ".join(available_ingredients))

    ingredients_input = st.text_input("Enter Ingredients (e.g., beef+lime juice+fish sauce)", "")

    def vectorize_ingredients(ingredients):
        words = ingredients.split("+")
        vectors = [word2vec.wv[word] for word in words if word in word2vec.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)

    if st.button("🔍 Predict Dish"):
        if ingredients_input:
            input_vector = vectorize_ingredients(ingredients_input).reshape(1, -1)
            prediction = model.predict(input_vector)
            predicted_dish = label_encoder.inverse_transform(prediction)
            st.success(f"🍽️ Predicted Thai Dish: **{predicted_dish[0]}**")
        else:
            st.warning("⚠️ Please enter some ingredients before predicting.")


# ===========================
# 🥐 Baked Food Health Classifier
# ===========================
if choice == "🥐 ML Bakery Health Classifier":
    st.header("🍞 Baked Food Health Classifier")
    st.markdown("### Enter nutritional values to predict if the food is **Healthy** or **Unhealthy**")

    # Load Model
    @st.cache_resource
    def load_bakery_model():
        return joblib.load("baked_food_rf_final_v3.pkl")

    model = load_bakery_model()

    # Define the MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit([[50, 0, 0, 0, 0], [600, 50, 100, 100, 30]])

    # User Inputs
    calories = st.slider("Calories (kcal)", 50, 600, 250)
    protein = st.slider("Protein (g)", 0.0, 50.0, 15.0)
    carbs = st.slider("Carbohydrates (g)", 0.0, 100.0, 30.0)
    sugar = st.slider("Sugar (g)", 0.0, 100.0, 3.0)
    saturated_fat = st.slider("Saturated Fat (g)", 0.0, 30.0, 2.0)

    if st.button("🔍 Predict Healthiness"):
        features = np.array([[calories, protein, carbs, sugar, saturated_fat]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        label = "✅ Healthy" if prediction[0] == 1 else "❌ Unhealthy"
        st.success(f"Predicted Classification: {label}")


# ===========================
# 🚆 Bangkok Train Line Predictor
# ===========================
if choice == "🚆 NN Train Line Predictor":
    st.header("🚆 Bangkok Train Line Predictor")

    # Load Models
    @st.cache_resource
    def load_train_models():
        return tf.keras.models.load_model("bangkok_train_nn_final.keras"), joblib.load("station_label_encoder_final.pkl"), joblib.load("lat_lng_scaler_final.pkl")

    model, label_encoder, scaler = load_train_models()
    geolocator = Nominatim(user_agent="geoapi")

    address = st.text_input("📍 Enter an address (Optional):", "")
    default_location = [13.7563, 100.5018]

    if address:
        location = geolocator.geocode(address)
        if location:
            default_location = [location.latitude, location.longitude]

    m = folium.Map(location=default_location, zoom_start=12)
    marker = folium.Marker(default_location, draggable=True).add_to(m)
    map_data = st_folium(m, height=400, width=700)

    if map_data and "last_clicked" in map_data:
        lat, lng = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
    else:
        lat, lng = default_location

    lat = st.number_input("🌍 Latitude:", value=lat, format="%.6f")
    lng = st.number_input("📍 Longitude:", value=lng, format="%.6f")

    if st.button("🔍 Predict Train Line"):
        input_data = scaler.transform([[lat, lng]])
        prediction_probs = model.predict(input_data)
        predicted_index = np.argmax(prediction_probs)
        predicted_line = label_encoder.inverse_transform([predicted_index])[0]
        st.success(f"🚇 Nearest Train Line: **{predicted_line}**")
