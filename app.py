import streamlit as st
from ultralytics import YOLO
from PIL import Image
import base64
from io import BytesIO
import json
import os
import base64

def get_base64_bg(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_base64 = get_base64_bg("assets/bg.jpg")

# 1. CẤU HÌNH TRANG
st.set_page_config(page_title="Food Recognition", layout="wide")

# 2. CSS ĐỂ ẨN GIAO DIỆN STREAMLIT VÀ TẠO STYLE APPLE
st.markdown("""
<style>
    /* Ẩn header, footer, menu của Streamlit */
    [data-testid="stHeader"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    footer {display: none;}
    .block-container {padding: 0rem !important; max-width: 100% !important;}
    
    /* Ảnh nền Hero (Full màn hình) */
    st.markdown(f"""
    <style>
    .stApp {{
        background:
            linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)),
            url("data:image/jpeg;base64,{bg_base64}") no-repeat center/cover;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)


    /* Biến cái nút upload mặc định thành Card kính mờ */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.3);
        max-width: 500px;
        margin: 10vh auto; /* Căn giữa */
        text-align: center;
    }
    
    /* Chỉnh chữ tiêu đề */
    .hero-title {
        text-align: center;
        color: white;
        padding-top: 8vh;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .hero-title h1 { font-size: 56px; font-weight: 600; margin-bottom: 0px; text-shadow: 0 2px 10px rgba(0,0,0,0.3); }
    .hero-title p { font-size: 24px; opacity: 0.9; font-weight: 300; }

    /* Hộp kết quả */
    .result-box {
        background: white;
        border-radius: 24px;
        padding: 40px;
        max-width: 900px;
        margin: 0 auto 50px auto;
        box-shadow: 0 20px 60px rgba(0,0,0,0.2);
        font-family: -apple-system, sans-serif;
        color: #1d1d1f;
    }
    
    /* Pills dinh dưỡng */
    .pill { display: inline-block; padding: 10px 20px; border-radius: 30px; font-size: 15px; font-weight: 500; margin: 5px; }
    .pill-blue { background: #e1f5fe; color: #01579b; }
    .pill-green { background: #e8f5e9; color: #2e7d32; }
    .pill-purple { background: #F3E5F5; color: #7B1FA2; }
    .pill-yellow { background: #FFFDE7; color: #FBC02D; }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

try:
    model = load_model()
    with open("data/nutrition.json", "r", encoding="utf-8") as f:
        nutrition_data = json.load(f)
except Exception as e:
    st.error(f"Lỗi tải file: {e}")
    st.stop()

# 4. GIAO DIỆN CHÍNH
st.markdown("""
    <div class="hero-title">
        <h1>Food Recognition.</h1>
        <p>Vietnamese Cuisine Simplified.</p>
    </div>
""", unsafe_allow_html=True)

# Khu vực Upload (Streamlit tự xử lý, nhưng đã bị CSS làm đẹp)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Xử lý ảnh
    img = Image.open(uploaded_file)
    results = model.predict(img, conf=0.25)
    
    # Giá trị mặc định
    label_display = "Unknown Dish"
    confidence = ""
    desc = "AI could not identify this dish."
    pills_html = ""

    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        label = model.names[int(box.cls)]
        prob = float(box.conf)
        
        info = nutrition_data.get(label, {})
        label_display = info.get("display_name", label).replace("_", " ").title()
        confidence = f"CONFIDENCE {prob:.1%}"
        desc = info.get("description", "")
        
        # Tạo HTML cho các viên thuốc dinh dưỡng
        cal = info.get("calories", "N/A")
        fat = info.get("fat", "N/A")
        sugar = info.get("sugar", "N/A")
        salt = info.get("salt", "N/A")
        
        pills_html = f"""
            <span class="pill pill-blue">🔥 {cal} kcal</span>
            <span class="pill pill-green">💧 Fat: {fat}g</span>
            <span class="pill pill-purple">🍭 Sugar: {sugar}g</span>
            <span class="pill pill-yellow">🧂 Salt: {salt}g</span>
        """

    # Chuyển ảnh sang base64 để hiển thị đẹp trong HTML
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Hiển thị kết quả (Inject HTML)
    st.markdown(f"""
    <div class="result-box">
        <div style="display: flex; gap: 40px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 300px;">
                <img src="data:image/jpeg;base64,{img_str}" style="width: 100%; border-radius: 20px;">
            </div>
            <div style="flex: 1; min-width: 300px; display: flex; flex-direction: column; justify-content: center;">
                <p style="color: #86868b; font-weight: 700; font-size: 13px; letter-spacing: 1px;">{confidence}</p>
                <h1 style="font-size: 48px; margin: 5px 0 15px 0;">{label_display}</h1>
                <p style="font-size: 18px; line-height: 1.6; color: #424245; margin-bottom: 30px;">{desc}</p>
                <div style="height: 1px; background: #e5e5e7; margin-bottom: 20px;"></div>
                <div>{pills_html}</div>
            </div>
        </div>
    </div>

    """, unsafe_allow_html=True)

