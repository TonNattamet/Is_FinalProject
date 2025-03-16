import streamlit as st

# Set page config at the start
st.set_page_config(page_title="Wel", page_icon="🍽️", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select Page", [
    "Introduction & Data Preparation",
    "Algorithm & Model Development",
    "Baked Food Health Classifier",
    "Thai Dish Predictor"
])

# Import respective pages
if page == "Introduction & Data Preparation":
    import pages.introduction as intro
    intro.show()

elif page == "Algorithm & Model Development":
    import pages.nn_test as nn_test
    nn_test.show()

elif page == "Baked Food Health Classifier":
    import pages.ml_Bakery_Nutrition as ml_Bakerynutrition
    ml_Bakerynutrition.show()

elif page == "Thai Dish Predictor":
    import pages.ml_ThaiFood as ml_ThaiFood
    ml_ThaiFood.show()