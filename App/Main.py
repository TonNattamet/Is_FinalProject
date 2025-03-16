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




import streamlit as st
from pages.introduction import show as introduction
from pages.explanation import show as explanation
from pages.ml_ThaiFood import show as ml_ThaiFood
from pages.ml_Bakery_Nutrition import show as ml_Bakery_Nutrition
from pages.nn_LineTrain_Predict import show as nn_LineTrain_Predict




st.sidebar.header("About Us")
st.sidebar.markdown("จัดทำโดย:นายณัฐเมศร์ วงศ์ธนิตเลิศ ห้อง DE-RA ชั้นปีที่ 4).")
st.sidebar.markdown("อาจารย์ผู้สอน:ดร.ณัฐกิตติ์ จิตรเอื้อตระกูล (NJR)")
st.sidebar.markdown("มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ")
st.sidebar.markdown("ภาควิชาวิทยาการคอมพิวเตอร์และสารสนเทศ")

PAGES = {
    "📊 Introduction": introduction,
    "😊 Explanation": explanation,
    "🍲 ML & NN Development": ml_ThaiFood,
    "🥙 ML Model Demo": ml_Bakery_Nutrition,
    "🚆 NN Model Demo": nn_LineTrain_Predict
}

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[choice]()
