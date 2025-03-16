
import streamlit as st

# Set Page Configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Welcome",
    page_icon="😍",
    layout="wide"
)

# Import Pages
from pages.introduction import show as introduction
from pages.explanation import show as explanation
from pages.ml_ThaiFood import show as ml_ThaiFood
from pages.ml_Bakery_Nutrition import show as ml_Bakery_Nutrition
from pages.nn_LineTrain_Predict import show as nn_LineTrain_Predict

# Sidebar Information
st.sidebar.header("About Us")
st.sidebar.markdown("**จัดทำโดย:** นายณัฐเมศร์ วงศ์ธนิตเลิศ (ห้อง DE-RA ชั้นปีที่ 4)")
st.sidebar.markdown("**อาจารย์ผู้สอน:** ดร.ณัฐกิตติ์ จิตรเอื้อตระกูล (NJR)")
st.sidebar.markdown("**มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ**")
st.sidebar.markdown("**ภาควิชาวิทยาการคอมพิวเตอร์และสารสนเทศ**")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
choice = st.sidebar.radio("🔍 Select a Page:", [
    "📊 Introduction",
    "📖 Explanation",
    "🍜 ML Thai Dish Predictor",
    "🥐 ML Bakery Health Classifier",
    "🚆 NN Train Line Predictor"
])

# Page Routing Using If-Else
if choice == "📊 Introduction":
    introduction()

elif choice == "📖 Explanation":
    explanation()

elif choice == "🍜 ML Thai Dish Predictor":
    ml_ThaiFood()

elif choice == "🥐 ML Bakery Health Classifier":
    ml_Bakery_Nutrition()

elif choice == "🚆 NN Train Line Predictor":
    nn_LineTrain_Predict()















