import streamlit as st
import base64
import json
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="FoodDetector",
    page_icon="🍜",
    layout="wide"
)

# ===============================
# UTILS
# ===============================
def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ===============================
# LOAD ASSETS
# ===============================
BG_IMAGE = img_to_base64("assets/bg.jpg")

# ===============================
# GLOBAL CSS
# ===============================
st.markdown(f"""
<style>
:root {{
    --primary: #22c55e;
    --secondary: #f97316;
    --dark: #0f172a;
    --light: #f8fafc;
    --glass: rgba(255,255,255,0.75);
}}

body {{
    background: var(--light);
}}

.block-container {{
    padding: 0;
    max-width: 100%;
}}

/* HERO */
.hero {{
    position: relative;
    width: 100%;
    height: 60vh;
    overflow: hidden;
}}

.hero-bg {{
    position: absolute;
    inset: 0;
    background-image: url("data:image/jpeg;base64,{BG_IMAGE}");
    background-size: cover;
    background-position: center;
}}

.hero-overlay {{
    position: absolute;
    inset: 0;
    background: linear-gradient(
        rgba(15,23,42,0.55),
        rgba(15,23,42,0.65)
    );
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: white;
}}

.hero-title {{
    font-size: 64px;
    font-weight: 900;
    letter-spacing: 1px;
}}

.hero-sub {{
    font-size: 22px;
    opacity: 0.9;
}}

/* SECTIONS */
.section {{
    max-width: 1100px;
    margin: -120px auto 80px auto;
    padding: 0 24px;
}}

.card {{
    background: var(--glass);
    backdrop-filter: blur(18px);
    border-radius: 28px;
    padding: 48px;
    box-shadow: 0 30px 80px rgba(0,0,0,0.25);
}}

/* UPLOAD */
.upload-card {{
    text-align: center;
}}

.upload-icon {{
    font-size: 64px;
    margin-bottom: 16px;
}}

/* RESULT */
.result {{
    margin-top: 80px;
}}

.result h2 {{
    font-size: 48px;
    margin-bottom: 8px;
}}

.conf {{
    font-size: 14px;
    letter-spacing: 0.15em;
    font-weight: 700;
    color: #475569;
}}

.pill {{
    display: inline-block;
    padding: 10px 20px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 600;
    margin: 6px 6px 0 0;
}}

.blue {{ background: #e0f2fe; color: #0369a1; }}
.green {{ background: #dcfce7; color: #166534; }}
.yellow {{ background: #fef9c3; color: #854d0e; }}
.pink {{ background: #fce7f3; color: #9d174d; }}

/* FOOTER */
.footer {{
    text-align: center;
    padding: 40px;
    color: #64748b;
}}
</style>
""", unsafe_allow_html=True)

# ===============================
# HERO
# ===============================
st.markdown("""
<div class="hero">
    <div class="hero-bg"></div>
    <div class="hero-overlay">
        <div class="hero-title">FoodDetector 🕵️</div>
        <div class="hero-sub">Detect Vietnamese dishes from an image</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL + DATA
# ===============================
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

with open("data/nutrition.json", "r", encoding="utf-8") as f:
    nutrition_data = json.load(f)

# ===============================
# MAIN SECTION
# ===============================
st.markdown('<div class="section">', unsafe_allow_html=True)

# UPLOAD
st.markdown('<div class="card upload-card">', unsafe_allow_html=True)
st.markdown('<div class="upload-icon">📸</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an image of food",
    type=["jpg", "jpeg", "png"]
)

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# RESULT
# ===============================
if uploaded_file:
    img = Image.open(uploaded_file)
    results = model.predict(img, conf=0.25)

    label_display = "Unknown Dish"
    confidence = "—"
    desc = "No description available."
    pills_html = ""

    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        label = model.names[int(box.cls)]
        prob = float(box.conf)

        info = nutrition_data.get(label, {})
        label_display = info.get("display_name", label).replace("_", " ").title()
        confidence = f"{prob:.1%}"
        desc = info.get("description", desc)

        pills_html = f"""
            <span class="pill blue">🔥 {info.get("calories","N/A")} kcal</span>
            <span class="pill green">🥩 Fat {info.get("fat","N/A")}g</span>
            <span class="pill pink">🍭 Sugar {info.get("sugar","N/A")}g</span>
            <span class="pill yellow">🧂 Salt {info.get("salt","N/A")}g</span>
        """

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(img, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div class="card result">
            <div class="conf">CONFIDENCE {confidence}</div>
            <h2>{label_display}</h2>
            <p>{desc}</p>
            <div>{pills_html}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<div class="footer">
    Built with YOLO & Streamlit · FoodDetector Demo
</div>
""", unsafe_allow_html=True)
