import streamlit as st
from ultralytics import YOLO
from PIL import Image
import base64
from io import BytesIO
import json

# ===============================
# 1. CONFIG
# ===============================
st.set_page_config(page_title="Food Recognition", layout="wide")

# ===============================
# 2. BACKGROUND BASE64
# ===============================
def get_base64_bg(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_base64 = get_base64_bg("assets/bg.jpg")

# ===============================
# 3. CSS (KHÔNG ĐỤNG stFileUploader)
# ===============================
st.markdown(f"""
<style>
.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}

.hero {{
    height: 70vh;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    background:
        linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)),
        url("data:image/jpeg;base64,{bg_base64}")
        no-repeat center / cover;
    color: white;
}}

.hero h1 {{
    font-size: 56px;
    font-weight: 600;
    margin-bottom: 8px;
}}

.hero p {{
    font-size: 24px;
    opacity: 0.85;
}}

.container {{
    max-width: 980px;
    margin: -120px auto 100px;
    padding: 0 20px;
}}

.upload-card {{
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 50px;
    box-shadow: 0 30px 80px rgba(0,0,0,0.25);
}}

.upload-title {{
    text-align: center;
    font-size: 20px;
    margin-bottom: 20px;
    color: #1d1d1f;
}}

.result-box {{
    background: white;
    border-radius: 24px;
    padding: 40px;
    margin-top: 80px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.2);
}}

.pill {{
    display: inline-block;
    padding: 10px 20px;
    border-radius: 30px;
    font-size: 15px;
    font-weight: 500;
    margin: 5px;
}}

.pill-blue {{ background: #e1f5fe; color: #01579b; }}
.pill-green {{ background: #e8f5e9; color: #2e7d32; }}
.pill-purple {{ background: #F3E5F5; color: #7B1FA2; }}
.pill-yellow {{ background: #FFFDE7; color: #FBC02D; }}
</style>
""", unsafe_allow_html=True)

# ===============================
# 4. HERO
# ===============================
st.markdown("""
<div class="hero">
    <div>
        <h1>Food Recognition, simplified.</h1>
        <p>Identify Vietnamese dishes from a single image.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# 5. LOAD MODEL (AI – GIỮ NGUYÊN)
# ===============================
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

with open("data/nutrition.json", "r", encoding="utf-8") as f:
    nutrition_data = json.load(f)

# ===============================
# 6. UPLOAD CARD (TRIỆT ĐỂ)
# ===============================
st.markdown('<div class="container"><div class="upload-card">', unsafe_allow_html=True)
st.markdown('<div class="upload-title">📷 Drop an image of a dish</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

st.markdown('</div></div>', unsafe_allow_html=True)

# ===============================
# 7. RESULT + YOLO
# ===============================
if uploaded_file:
    img = Image.open(uploaded_file)
    results = model.predict(img, conf=0.25)

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

        pills_html = f"""
            <span class="pill pill-blue">🔥 {info.get("calories","N/A")} kcal</span>
            <span class="pill pill-green">💧 Fat: {info.get("fat","N/A")}g</span>
            <span class="pill pill-purple">🍭 Sugar: {info.get("sugar","N/A")}g</span>
            <span class="pill pill-yellow">🧂 Salt: {info.get("salt","N/A")}g</span>
        """

    buf = BytesIO()
    img.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    st.markdown(f"""
    <div class="container">
        <div class="result-box">
            <div style="display:flex; gap:40px; flex-wrap:wrap;">
                <div style="flex:1; min-width:300px;">
                    <img src="data:image/jpeg;base64,{img_b64}"
                         style="width:100%; border-radius:20px;">
                </div>
                <div style="flex:1; min-width:300px;">
                    <p style="color:#86868b; letter-spacing:1px; font-weight:700;">
                        {confidence}
                    </p>
                    <h1 style="font-size:48px;">{label_display}</h1>
                    <p style="color:#424245; font-size:18px;">
                        {desc}
                    </p>
                    <div style="height:1px; background:#e5e5e7; margin:20px 0;"></div>
                    <div>{pills_html}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
