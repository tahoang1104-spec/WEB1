import streamlit as st
import base64
import json
from ultralytics import YOLO
from PIL import Image

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="FoodDetector",
    page_icon="🕵️",
    layout="wide"
)

# ===============================
# UTILS
# ===============================
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ===============================
# ASSETS
# ===============================
HERO_BG = img_to_base64("assets/bg.jpg")

# ===============================
# GLOBAL CSS (WEBSITE STYLE)
# ===============================
st.markdown(f"""
<style>
:root {{
    --dark: #0f172a;
    --glass: rgba(255,255,255,0.78);
}}

.block-container {{
    padding: 0;
    max-width: 100%;
}}

body {{
    background: #f8fafc;
}}

/* HERO */
.hero {{
    position: relative;
    height: 65vh;
}}

.hero-bg {{
    position: absolute;
    inset: 0;
    background: url("data:image/jpeg;base64,{HERO_BG}") center/cover no-repeat;
}}

.hero-overlay {{
    position: absolute;
    inset: 0;
    background: linear-gradient(
        rgba(15,23,42,0.55),
        rgba(15,23,42,0.7)
    );
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: white;
}}

.hero h1 {{
    font-size: 64px;
    font-weight: 900;
}}

.hero p {{
    font-size: 22px;
    opacity: 0.9;
}}

/* SECTIONS */
.section {{
    max-width: 1100px;
    margin: -120px auto 80px;
    padding: 0 24px;
}}

.card {{
    background: var(--glass);
    backdrop-filter: blur(20px);
    border-radius: 28px;
    padding: 48px;
    box-shadow: 0 40px 100px rgba(0,0,0,0.25);
}}

/* UPLOAD */
.upload {{
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
.pink {{ background: #fce7f3; color: #9d174d; }}
.yellow {{ background: #fef9c3; color: #854d0e; }}

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
        <div>
            <h1>FoodDetector 🕵️</h1>
            <p>Detect Vietnamese dishes from a single image</p>
        </div>
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

# UPLOAD CARD
st.markdown('<div class="card upload">', unsafe_allow_html=True)
st.markdown('<div class="upload-icon">📸</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a food image",
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
    pills = ""

    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        label = model.names[int(box.cls)]
        prob = float(box.conf)

        info = nutrition_data.get(label, {})
        label_display = info.get("display_name", label).replace("_", " ").title()
        confidence = f"{prob:.1%}"
        desc = info.get("description", desc)

        pills = f"""
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
            <div>{pills}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<div class="footer">
    FoodDetector · YOLO · Streamlit
</div>
""", unsafe_allow_html=True)
