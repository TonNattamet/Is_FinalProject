import streamlit as st

# Set page config at the start
st.set_page_config(page_title="Wel", page_icon="🍽️", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select Page", [
    "Introduction & Data Preparation",
    "1.Ml Bakery Health Classifier",
    "2.Ml Thai Dish Predictor",
    "3.NN Tic Tac Toe",
])

# Import respective pages
if page == "Introduction & Data Preparation":
    import pages.introduction as intro
    intro.show()

elif page == "3.NN LineTrain_Predict":
    import pages.Nn_LineTrain_Predict as Nn_LineTrain_Predict
    Nn_LineTrain_Predict.show()

elif page == "1.Ml Bakery Health Classifier":
    import pages.ml_Bakery_Nutrition as ml_Bakerynutrition
    ml_Bakerynutrition.show()

elif page == "2.Ml Thai Dish Predictor":
    import pages.ml_ThaiFood as ml_ThaiFood
    ml_ThaiFood.show()