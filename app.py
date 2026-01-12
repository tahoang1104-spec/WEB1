import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import json
import base64

# ===============================
# 1. CẤU HÌNH TRANG
# ===============================
st.set_page_config(
    page_title="Food AI Playground",
    layout="wide"
)

# ===============================
# 2. CSS – GIAO DIỆN VUI NHỘN, CÂN XỨNG
# ===============================
st.markdown("""
<style>
body {
    background: #f6f7fb;
}

.block-container {
    padding-top: 2rem;
}

/* HEADER */
.header {
    text-align: center;
    margin-bottom: 40px;
}

.header h1 {
    font-size: 52px;
    font-weight: 700;
}

.header p {
    font-size: 20px;
    color: #6b7280;
}

/* CARD */
.card {
    background: white;
    border-radius: 24px;
    padding: 40px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

/* UPLOAD */
.upload-card {
    max-width: 500px;
    margin: 0 auto 50px auto;
    text-align: center;
}

.upload-icon {
    font-size: 48px;
    margin-bottom: 10px;
}

/* RESULT */
.result-card {
    margin-top: 20px;
}

.pill {
    display: inline-block;
    padding: 10px 18px;
    border-radius: 20px;
    font-size: 14px;
    margin: 6px 6px 0 0;
}

.blue { background: #e0f2fe; color: #0369a1; }
.green { background: #dcfce7; color: #166534; }
.pink { background: #fce7f3; color: #9d174d; }
.yellow { background: #fef9c3; color: #854d0e; }
</style>
""", unsafe_allow_html=True)

# ===============================
# 3. HEADER
# ===============================
st.markdown("""
<div class="header">
    <h1>🍜 Food AI Playground</h1>
    <p>Upload a food photo and let AI guess your dish ✨</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# 4. UPLOAD CARD (KHÔNG LỖI)
# ===============================
st.markdown('<div class="card upload-card">', unsafe_allow_html=True)
st.markdown('<div class="upload-icon">📸</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image of food",
    type=["jpg", "jpeg", "png"]
)

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# 5. LOAD MODEL YOLO (PHẦN AI)
# ===============================
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

try:
    model = load_model()
    with open("data/nutrition.json", "r", encoding="utf-8") as f:
        nutrition_data = json.load(f)
except Exception as e:
    st.error(f"Lỗi tải model hoặc dữ liệu: {e}")
    st.stop()

# ===============================
# 6. XỬ LÝ + HIỂN THỊ KẾT QUẢ
# ===============================
if uploaded_file:
    img = Image.open(uploaded_file)
    results = model.predict(img, conf=0.25)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, use_container_width=True)

    with col2:
        st.markdown('<div class="card result-card">', unsafe_allow_html=True)

        label_display = "Unknown Dish"
        confidence = "0%"
        desc = "AI could not identify this dish."
        pills_html = ""

        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            label = model.names[int(box.cls)]
            prob = float(box.conf)

            info = nutrition_data.get(label, {})
            label_display = info.get("display_name", label).replace("_", " ").title()
            confidence = f"{prob:.1%} confidence"
            desc = info.get("description", "No description available.")

            pills_html = f"""
                <span class="pill blue">🔥 {info.get("calories","N/A")} kcal</span>
                <span class="pill green">🥩 Fat: {info.get("fat","N/A")}g</span>
                <span class="pill pink">🍭 Sugar: {info.get("sugar","N/A")}g</span>
                <span class="pill yellow">🧂 Salt: {info.get("salt","N/A")}g</span>
            """

        st.markdown(f"""
        <h2>{label_display}</h2>
        <p style="color:#6b7280">{confidence}</p>
        <p>{desc}</p>
        <div>{pills_html}</div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
