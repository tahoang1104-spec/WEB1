# ================= FIX TRIỆT ĐỂ – COPY ĐÈ TOÀN BỘ PHẦN CSS CŨ =================
st.markdown(f"""
<style>
/* RESET STREAMLIT */
html, body, [class*="css"] {{
    margin: 0;
    padding: 0;
}}

section.main > div {{
    padding: 0 !important;
    max-width: 100% !important;
}}

.block-container {{
    padding: 0 !important;
    max-width: 100vw !important;
}}

header, footer {{
    display: none;
}}

/* HERO FULL WIDTH */
.hero {{
    position: relative;
    width: 100vw;
    height: 70vh;
    left: 50%;
    right: 50%;
    margin-left: -50vw;
    margin-right: -50vw;
}}

.hero-bg {{
    position: absolute;
    inset: 0;
    background: url("data:image/jpeg;base64,{bg_base64}") center/cover no-repeat;
}}

.hero-overlay {{
    position: absolute;
    inset: 0;
    background: linear-gradient(
        rgba(0,0,0,0.55),
        rgba(0,0,0,0.75)
    );
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: white;
}}

.hero-title {{
    font-size: 64px;
    font-weight: 900;
}}

.hero-subtitle {{
    font-size: 20px;
    margin-top: 8px;
}}

/* CONTENT */
.main-container {{
    max-width: 1100px;
    margin: -100px auto 80px;
    padding: 0 24px;
}}

.card {{
    background: white;
    border-radius: 18px;
    padding: 36px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.15);
    margin-bottom: 40px;
}}

/* UPLOADER */
[data-testid="stFileUploader"] {{
    border-radius: 16px;
    padding: 24px;
    background: #f9fafb;
}}
</style>
""", unsafe_allow_html=True)

# ================= HERO (KHÔNG ĐỔI) =================
st.markdown(f"""
<div class="hero">
    <div class="hero-bg"></div>
    <div class="hero-overlay">
        <div>
            <div class="hero-title">FoodDetector 🕵️</div>
            <div class="hero-subtitle">
                Detect Vietnamese dishes from a single image
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
