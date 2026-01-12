import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import json
import base64

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Food AI",
    layout="wide",
    page_icon="🍔"
)

# ===============================
# GLOBAL STYLE (CLEAN + PLAYFUL)
# ===============================
st.markdown("""
<style>
body {
    background: linear-gradient(180deg, #f8fafc, #eef2ff);
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* HEADER */
.header {
    text-align: center;
    margin-bottom: 3rem;
}

.header h1 {
    font-size: 56px;
    font-weight: 800;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header p {
    font-size: 20px;
    color: #475569;
}

/* CARD */
.card {
    background: white;
    border-radius: 28px;
    padding: 42px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.08);
}

/* UPLOAD */
.upload {
    max-width: 520px;
    margin: 0 auto 60px auto;
    text-align: center;
}

.upload-icon {
    font-size: 56px;
    margin-bottom: 12px;
}

/* RESULT */
.result-card h2 {
    font-size: 42px;
    margin-bottom: 6px;
}

.confidence {
    font-size: 14px;
    letter-spacing: 0.15em;
    font-weight: 700;
    color: #64748b;
}

.pills {
    margin-top: 20px;
}

.pill {
    display: inline-block;
    padding: 10px 18px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 600;
    margin: 6px 6px 0 0;
}

.blue { background: #e0f2fe; color: #0369a1; }
.green { background: #dcfce7; color: #166534; }
.pink { background: #fce7f3; color: #9d174d; }
.yellow { background: #fef9c3; color: #854d0e; }

/* FOOTER NOTE */
.note {
    text-align: center;
    margin-top: 60px;
    color: #94a3b8;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown("""
<div class="header">
    <h1>🍜 Food AI</h1>
    <p>Upload a food photo. Let AI recognize your dish ✨</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# UPLOAD CARD
# ===============================
st.markdown('<div class="card upload">', unsafe_allow_html=True)
st.markdown('<div class="upload-icon">📸</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image of food",
    type=["jpg", "jpeg", "png"]
)

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# LOAD AI MODEL
# ===============================
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

try:
    model = load_model()
    with open("data/nutrition.json", "r", encoding="utf-8") as f:
        nutrition_data = json.load(f)
except Exception as e:
    st.error(f"Failed to load model or data: {e}")
    st.stop()

# ===============================
# RESULT
# ===============================
if uploaded_file:
    img = Image.open(uploaded_file)
    results = model.predict(img, conf=0.25)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(img, use_container_width=True)

    with col2:
        st.markdown('<div class="card result-card">', unsafe_allow_html=True)

        label_display = "Unknown Dish"
        confidence = "—"
        desc = "AI could not confidently identify this dish."
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
                <span class="pill green">🥩 Fat: {info.get("fat","N/A")}g</span>
                <span class="pill pink">🍭 Sugar: {info.get("sugar","N/A")}g</span>
                <span class="pill yellow">🧂 Salt: {info.get("salt","N/A")}g</span>
            """

        st.markdown(f"""
        <div class="confidence">CONFIDENCE {confidence}</div>
        <h2>{label_display}</h2>
        <p style="color:#475569; font-size:17px;">{desc}</p>
        <div class="pills">{pills_html}</div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<div class="note">
    Powered by YOLO · Built with Streamlit
</div>
""", unsafe_allow_html=True)
