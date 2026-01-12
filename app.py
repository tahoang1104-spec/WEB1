import streamlit as st
import base64
import json
from ultralytics import YOLO
from PIL import Image

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FoodDetector",
    page_icon="🕵️",
    layout="wide"
)

# ================= UTILS =================
def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ================= LOAD ASSETS =================
bg_base64 = img_to_base64("assets/bg.jpg")

# ================= GLOBAL STYLE =================
st.markdown("""
<style>
.block-container {
    padding: 0;
    max-width: 100%;
}

body {
    background-color: #f5f5f5;
}

/* HEADER */
.header-container {
    position: relative;
    width: 100%;
    height: calc(160px + 15vw);
    overflow: hidden;
}

.header-image {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.header-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        rgba(0,0,0,0.55),
        rgba(0,0,0,0.75)
    );
    display: flex;
    said: center;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: white;
}

.header-title {
    font-size: calc(32px + 2vw);
    font-weight: 800;
}

.header-subtitle {
    font-size: calc(14px + 0.6vw);
    margin-top: 8px;
}

/* CONTENT */
.main-container {
    max-width: 1100px;
    margin: -80px auto 80px;
    padding: 0 24px;
}

.card {
    background: white;
    border-radius: 18px;
    padding: 36px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.15);
    margin-bottom: 40px;
}

/* UPLOADER */
[data-testid="stFileUploader"] {
    border-radius: 16px;
    padding: 24px;
    background: #f9fafb;
}

/* RESULT */
.result-title {
    font-size: 40px;
    font-weight: 700;
}

.confidence {
    letter-spacing: 0.15em;
    font-size: 13px;
    font-weight: 700;
    color: #6b7280;
}

.pill {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    margin: 6px 6px 0 0;
}

.blue { background:#e0f2fe; color:#0369a1; }
.green { background:#dcfce7; color:#166534; }
.pink { background:#fce7f3; color:#9d174d; }
.yellow { background:#fef9c3; color:#854d0e; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown(f"""
<div class="header-container">
    <img src="data:image/jpeg;base64,{bg_base64}" class="header-image">
    <div class="header-overlay">
        <div>
            <div class="header-title">FoodDetector 🕵️</div>
            <div class="header-subtitle">
                Detect Vietnamese dishes from a single image
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

with open("data/nutrition.json", "r", encoding="utf-8") as f:
    nutrition = json.load(f)

# ================= MAIN =================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# INTRO
st.markdown("""
<div class="card">
FoodDetector uses a fine-tuned <b>YOLO</b> model to detect Vietnamese dishes from an image
and estimate basic nutrition information.
</div>
""", unsafe_allow_html=True)

# CONFIDENCE
confidence = st.slider(
    "Adjust confidence threshold",
    min_value=10,
    max_value=90,
    value=50
) / 100

# UPLOAD
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload a food image",
    type=["jpg","jpeg","png"]
)
st.markdown('</div>', unsafe_allow_html=True)

# RESULT
if uploaded:
    img = Image.open(uploaded)
    results = model.predict(img, conf=confidence)

    label = "Unknown dish"
    conf = "—"
    desc = "No description available."
    pills = ""

    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        cls = model.names[int(box.cls)]
        prob = float(box.conf)
        info = nutrition.get(cls, {})
        label = info.get("display_name", cls).replace("_"," ").title()
        conf = f"{prob:.1%}"
        desc = info.get("description", desc)

        pills = f"""
        <span class="pill blue">🔥 {info.get("calories","N/A")} kcal</span>
        <span class="pill green">🥩 Fat {info.get("fat","N/A")}g</span>
        <span class="pill pink">🍭 Sugar {info.get("sugar","N/A")}g</span>
        <span class="pill yellow">🧂 Salt {info.get("salt","N/A")}g</span>
        """

    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.image(img, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="confidence">CONFIDENCE {conf}</div>
            <div class="result-title">{label}</div>
            <p>{desc}</p>
            <div>{pills}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
