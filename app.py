import json
import os
import re

import openai
import streamlit as st
from dotenv import load_dotenv

from src.rag_backend import CITIES, initialize_rag_store, query_market_price
from src.vision_backend import extract_from_receipt_image, extract_from_voice

load_dotenv(override=True)

st.set_page_config(
    page_title="NirkhNama AI — Smart Price Auditor",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS — Dark Glassmorphism Design System
# ─────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --emerald: #0D9373;
    --emerald-dark: #0A7B6B;
    --emerald-glow: rgba(13, 147, 115, 0.35);
    --gold: #F59E0B;
    --gold-soft: rgba(245, 158, 11, 0.15);
    --red: #EF4444;
    --red-soft: rgba(239, 68, 68, 0.12);
    --green: #10B981;
    --green-soft: rgba(16, 185, 129, 0.12);
    --dark-bg: #0E1117;
    --card-bg: rgba(26, 26, 46, 0.65);
    --card-border: rgba(255, 255, 255, 0.06);
    --glass-blur: blur(16px);
    --text-primary: #FAFAFA;
    --text-secondary: #94A3B8;
    --radius: 16px;
    --radius-sm: 10px;
}

/* ── Global Typography ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

h1, h2, h3, h4, h5, h6,
.hero-title, .hero-subtitle {
    font-family: 'Outfit', sans-serif !important;
}

/* ── Hero Banner ── */
.hero-container {
    background: linear-gradient(135deg, #0D9373 0%, #065f4a 50%, #1A1A2E 100%);
    border-radius: var(--radius);
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%; right: -30%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(255,255,255,0.04) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-container::after {
    content: '';
    position: absolute;
    bottom: -40%; left: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(13, 147, 115, 0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-icon {
    font-size: 2.8rem;
    margin-bottom: 0.3rem;
    display: inline-block;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #fff;
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
    line-height: 1.15;
}
.hero-title span {
    background: linear-gradient(90deg, #6EE7B7, #A7F3D0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.75);
    margin: 0;
    font-weight: 400;
    line-height: 1.5;
    max-width: 520px;
}

/* ── Sidebar Styling ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #121827 0%, #0E1117 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
section[data-testid="stSidebar"] .stSelectbox label {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Tab Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: var(--card-bg);
    backdrop-filter: var(--glass-blur);
    border-radius: var(--radius);
    padding: 6px;
    border: 1px solid var(--card-border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm);
    padding: 10px 20px;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500;
    font-size: 0.9rem;
    color: var(--text-secondary);
    transition: all 0.3s ease;
    border: none !important;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--emerald), var(--emerald-dark)) !important;
    color: #fff !important;
    font-weight: 600;
    box-shadow: 0 4px 15px var(--emerald-glow);
}
.stTabs [data-baseweb="tab"]:hover {
    color: #fff;
    background: rgba(255,255,255,0.05);
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ── Primary Button — Audit Glow ── */
.stButton > button[kind="primary"],
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--emerald), var(--emerald-dark)) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.5px;
    padding: 0.75rem 1.5rem !important;
    color: #fff !important;
    box-shadow: 0 4px 20px var(--emerald-glow);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    animation: pulseGlow 2.5s ease-in-out infinite;
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 4px 20px var(--emerald-glow); }
    50% { box-shadow: 0 4px 35px rgba(13, 147, 115, 0.55), 0 0 60px rgba(13, 147, 115, 0.2); }
}
.stButton > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 30px rgba(13, 147, 115, 0.5) !important;
}

/* ── Secondary Button ── */
.stButton > button[kind="secondary"],
div[data-testid="stButton"] > button[kind="secondary"] {
    background: var(--card-bg) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    backdrop-filter: var(--glass-blur);
    transition: all 0.3s ease !important;
}
.stButton > button[kind="secondary"]:hover,
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: var(--emerald) !important;
    transform: translateY(-1px) !important;
}

/* ── Result Card Styles ── */
.result-card {
    background: var(--card-bg);
    backdrop-filter: var(--glass-blur);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    border: 1px solid var(--card-border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.3s ease;
    animation: fadeSlideIn 0.4s ease-out;
}
.result-card:hover {
    border-color: rgba(255,255,255,0.12);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-card.overcharged {
    border-left: 3px solid var(--red);
    background: linear-gradient(135deg, var(--red-soft), var(--card-bg));
}
.result-card.fair {
    border-left: 3px solid var(--green);
    background: linear-gradient(135deg, var(--green-soft), var(--card-bg));
}
.result-item-name {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    color: var(--text-primary);
    margin: 0;
}
.result-price-row {
    display: flex;
    gap: 1.2rem;
    align-items: center;
    margin-top: 0.3rem;
}
.result-price {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: var(--text-secondary);
}
.result-price strong {
    color: var(--text-primary);
    font-weight: 600;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    border-radius: 100px;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    white-space: nowrap;
}
.badge-fair {
    background: var(--green-soft);
    color: var(--green);
    border: 1px solid rgba(16, 185, 129, 0.25);
}
.badge-overcharged {
    background: var(--red-soft);
    color: var(--red);
    border: 1px solid rgba(239, 68, 68, 0.25);
}

/* ── Summary Stats Bar ── */
.stats-bar {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 130px;
    background: var(--card-bg);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}
.stat-card:hover {
    border-color: rgba(255,255,255,0.12);
}
.stat-number {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    line-height: 1;
}
.stat-number.text-green { color: var(--green); }
.stat-number.text-red { color: var(--red); }
.stat-number.text-gold { color: var(--gold); }
.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.35rem;
}

/* ── Complaint Section ── */
.complaint-header {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), var(--card-bg));
    border: 1px solid rgba(245, 158, 11, 0.15);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    margin: 1.2rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.complaint-header-icon {
    font-size: 1.5rem;
}
.complaint-header-text h3 {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--gold);
    margin: 0;
}
.complaint-header-text p {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin: 0.15rem 0 0;
}

/* ── Footer ── */
.footer {
    margin-top: 3rem;
    padding: 1.5rem 0 1rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    text-align: center;
}
.footer-text {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: var(--text-secondary);
    letter-spacing: 0.3px;
}
.footer-brand {
    color: var(--emerald);
    font-weight: 600;
}

/* ── Info / Warning Cards ── */
.stAlert > div {
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── File Uploader ── */
section[data-testid="stFileUploader"] {
    border-radius: var(--radius-sm);
}

/* ── Subheader ── */
.results-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* ── Hide Streamlit Defaults ── */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
footer { visibility: hidden; }

/* ── Responsive ── */
@media (max-width: 640px) {
    .hero-title { font-size: 1.6rem; }
    .hero-subtitle { font-size: 0.9rem; }
    .stats-bar { flex-direction: column; }
    .result-card { flex-direction: column; align-items: flex-start; gap: 0.6rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────
# RAG Store
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    return initialize_rag_store()


try:
    price_table = load_rag()
except Exception as e:
    st.error(f"Failed to initialize RAG store: {e}")
    st.stop()

if "audit_done" not in st.session_state:
    st.session_state["audit_done"] = False
if "results_data" not in st.session_state:
    st.session_state["results_data"] = []


# ─────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-container">
    <div class="hero-icon">🛡️</div>
    <h1 class="hero-title">Nirkh<span>Nama</span> AI</h1>
    <p class="hero-subtitle">
        Upload your grocery receipt or speak your purchase — we'll check if you were
        overcharged against official government rates across 17 cities in Pakistan.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────
# SIDEBAR — City Selector
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2rem;">🛡️</div>
        <div style="font-family:'Outfit',sans-serif; font-weight:700; font-size:1.15rem;
                    color:#FAFAFA; letter-spacing:-0.3px; margin-top:0.2rem;">
            NirkhNama AI
        </div>
        <div style="font-family:'Inter',sans-serif; font-size:0.7rem; color:#64748B;
                    letter-spacing:0.8px; text-transform:uppercase; margin-top:0.15rem;">
            Smart Price Auditor
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    city = st.selectbox(
        "📍 SELECT CITY",
        CITIES,
        index=CITIES.index("Lahore") if "Lahore" in CITIES else 0,
        help="Prices are compared against the official average rate for the city you select.",
    )

    st.markdown("---")

    st.markdown(
        """
    <div style="padding: 0.8rem; background: rgba(13,147,115,0.08); border-radius: 10px;
                border: 1px solid rgba(13,147,115,0.15); margin-top: 0.5rem;">
        <div style="font-family:'Inter',sans-serif; font-size:0.75rem; color:#94A3B8; line-height:1.5;">
            <strong style="color:#0D9373;">How it works</strong><br>
            1. 📸 Snap or upload your receipt<br>
            2. 🤖 AI extracts items &amp; prices<br>
            3. ✅ Prices checked against Nirkh Nama<br>
            4. 📝 Get complaint letter if overcharged
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────
# INPUT TABS — Receipt & Voice
# ─────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📸  Snap Receipt", "🎤  Voice Input"])

with tab1:
    st.info(
        "Take a clear photo of your grocery receipt or a handwritten price slip, "
        "or capture it live with your camera."
    )
    input_mode = st.radio(
        "Receipt source",
        ["Upload photo", "Use camera"],
        horizontal=True,
        key="receipt_source",
    )
    image_file = None
    camera_file = None
    if input_mode == "Upload photo":
        image_file = st.file_uploader(
            "Upload receipt photo", type=["jpg", "jpeg", "png"], key="image_uploader"
        )
    else:
        camera_file = st.camera_input("Capture receipt", key="camera_input")

with tab2:
    st.info(
        "Tap the mic and say what you bought and how much you paid. Urdu and English both work."
    )
    audio_file = st.audio_input(
        "Record your purchase (e.g. 'Aloo 150 rupey kilo, tamatar 80 rupey')",
        key="audio_recorder",
    )

receipt_image = image_file if image_file is not None else camera_file

st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
audit_clicked = st.button(
    "🔍  Audit My Prices", type="primary", use_container_width=True
)

# ─────────────────────────────────────────────────────────
# PRICE AUDIT LOGIC
# ─────────────────────────────────────────────────────────
if audit_clicked:
    extracted_items = []

    if receipt_image is not None:
        with st.spinner("Reading your receipt with GPT-4o Vision..."):
            try:
                image_bytes = receipt_image.getvalue()
                extracted_items = extract_from_receipt_image(
                    image_bytes,
                    mime_type=(getattr(receipt_image, "type", None) or "image/jpeg"),
                )
            except Exception as e:
                st.error(f"Receipt extraction failed: {e}")
                st.stop()
    elif audio_file is not None:
        with st.spinner("Transcribing with Whisper, then parsing with GPT-4o..."):
            try:
                audio_bytes = audio_file.read()
                extracted_items = extract_from_voice(audio_bytes)
            except Exception as e:
                st.error(f"Voice extraction failed: {e}")
                st.stop()
    else:
        st.warning("Please upload a receipt photo OR record your voice first.")
        st.stop()

    if not extracted_items:
        st.error(
            "Could not extract any items. Try a clearer photo or speak more slowly and clearly."
        )
        st.stop()

    st.success(
        f"Found {len(extracted_items)} item(s). Checking against official prices..."
    )

    results_data = []
    for item in extracted_items:
        item_name = item.get("item_name", "Unknown")
        price_paid = item.get("price_paid", 0)
        try:
            price_paid_num = float(price_paid)
        except Exception:
            price_paid_num = 0.0

        with st.spinner(f"Checking official price for {item_name} in {city}..."):
            try:
                official_price_str = query_market_price(item_name, price_table, city)
            except Exception as e:
                st.error(f"Official price lookup failed for {item_name}: {e}")
                st.stop()

        cap_numbers = re.findall(r"\d+(?:\.\d+)?", official_price_str)
        cap_num = max((float(n) for n in cap_numbers), default=0.0)
        if cap_num > 0 and price_paid_num > cap_num:
            status = "Overcharged"
            diff = price_paid_num - cap_num
        else:
            status = "Fair Price"
            diff = 0.0

        results_data.append(
            {
                "Item": item_name,
                "Price Paid (PKR)": price_paid_num,
                "Official Cap": official_price_str,
                "Cap Num": cap_num,
                "Diff": diff,
                "Status": status,
            }
        )

    st.session_state["results_data"] = results_data
    st.session_state["audit_done"] = True


# ─────────────────────────────────────────────────────────
# RESULTS DISPLAY — Styled Cards
# ─────────────────────────────────────────────────────────
if st.session_state.get("audit_done") and st.session_state.get("results_data"):
    results_data = st.session_state["results_data"]

    overcharged = [r for r in results_data if r["Status"] == "Overcharged"]
    fair = [r for r in results_data if r["Status"] == "Fair Price"]
    total_overpaid = sum(r["Diff"] for r in overcharged)

    # ── Summary Stats Bar ──
    st.markdown(
        f"""
    <div class="results-heading">📊 Price Audit Results — {city}</div>
    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-number">{len(results_data)}</div>
            <div class="stat-label">Items Checked</div>
        </div>
        <div class="stat-card">
            <div class="stat-number text-green">{len(fair)}</div>
            <div class="stat-label">Fair Price</div>
        </div>
        <div class="stat-card">
            <div class="stat-number text-red">{len(overcharged)}</div>
            <div class="stat-label">Overcharged</div>
        </div>
        <div class="stat-card">
            <div class="stat-number text-gold">Rs.{total_overpaid:,.0f}</div>
            <div class="stat-label">Total Overpaid</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Individual Result Cards ──
    for r in results_data:
        card_class = "overcharged" if r["Status"] == "Overcharged" else "fair"
        badge_class = "badge-overcharged" if r["Status"] == "Overcharged" else "badge-fair"
        badge_icon = "⚠️" if r["Status"] == "Overcharged" else "✅"
        badge_text = r["Status"]

        diff_text = ""
        if r["Status"] == "Overcharged" and r["Diff"] > 0:
            diff_text = f'<span style="color:var(--red); font-size:0.8rem; font-weight:600; margin-left:0.5rem;">+Rs.{r["Diff"]:,.0f} over</span>'

        st.markdown(
            f"""
        <div class="result-card {card_class}">
            <div>
                <p class="result-item-name">{r["Item"]}</p>
                <div class="result-price-row">
                    <span class="result-price">Paid: <strong>Rs.{r["Price Paid (PKR)"]:,.0f}</strong></span>
                    <span class="result-price">Official: <strong>{r["Official Cap"]}</strong></span> {diff_text}
                </div>
            </div>
            <div class="badge {badge_class}">{badge_icon} {badge_text}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Overcharged Actions ──
    if overcharged:
        st.markdown(
            f"""
        <div class="complaint-header">
            <div class="complaint-header-icon">⚖️</div>
            <div class="complaint-header-text">
                <h3>Take Action — You Were Overcharged</h3>
                <p>Generate a formal complaint letter in English &amp; Urdu to file with the Deputy Commissioner or Pakistan Citizens Portal.</p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button(
            "📝  Generate Official Complaint Letter",
            type="secondary",
            use_container_width=True,
        ):
            with st.spinner("Drafting your complaint in English and Urdu..."):
                try:
                    from openai import OpenAI

                    client = OpenAI()
                    overcharged_summary = "\n".join(
                        [
                            f"- {r['Item']}: paid Rs.{r['Price Paid (PKR)']}, official cap is {r['Official Cap']}"
                            for r in overcharged
                        ]
                    )
                    complaint_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a consumer rights assistant in Pakistan. Write formal complaint letters.",
                            },
                            {
                                "role": "user",
                                "content": f"""Write a formal complaint letter addressed to 'The Deputy Commissioner / Pakistan Citizens Portal (PMDU)'.

The complainant is a resident of {city}. The following grocery items were sold above the official government Nirkh Nama average price for {city}:
{overcharged_summary}

The letter must:
- State the specific items and exact price discrepancy
- Reference that this violates the official Nirkh Nama (government price list)
- Request immediate action against the vendor
- Be professional but firm in tone
- Use [YOUR NAME], [YOUR CNIC], and [YOUR AREA/SECTOR] as placeholders

Write the ENGLISH version first with a clear heading, then write the URDU version below with a clear heading. Keep both versions complete.""",
                            },
                        ],
                    )
                    complaint_text = complaint_response.choices[0].message.content
                    st.text_area(
                        "Your Complaint Letter", value=complaint_text, height=450
                    )
                    st.info(
                        "Copy this and send it via the Pakistan Citizens Portal app, or WhatsApp it to your local DC office."
                    )
                except Exception as e:
                    st.error(f"Complaint generation failed: {e}")
    else:
        st.balloons()
        st.success(
            "🎉 Great news! All prices are within official limits. You paid fair prices today!"
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button("🔄  Check Another Receipt", use_container_width=True):
        st.session_state["audit_done"] = False
        st.session_state["results_data"] = []
        st.rerun()


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown(
    """
<div class="footer">
    <div class="footer-text">
        Built with ❤️ for Pakistani consumers &nbsp;·&nbsp;
        Powered by <span class="footer-brand">NirkhNama AI</span> &nbsp;·&nbsp;
        Prices sourced from the official Government Nirkh Nama
    </div>
</div>
""",
    unsafe_allow_html=True,
)
