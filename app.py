import json
import os
import re

import openai
import streamlit as st
from dotenv import load_dotenv

from src.rag_backend import CITIES, initialize_rag_store, query_market_price
from src.vision_backend import extract_from_receipt_image, extract_from_voice

load_dotenv()

st.set_page_config(page_title="NirkhNama AI", layout="centered")


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


st.title("NirkhNama AI")
st.caption(
    "Upload your grocery receipt or speak your purchase - we'll check if you were overcharged against official government rates."
)
st.divider()

city = st.selectbox(
    "Select your city",
    CITIES,
    index=CITIES.index("Lahore") if "Lahore" in CITIES else 0,
    help="Prices are compared against the official average rate for the city you select.",
)

tab1, tab2 = st.tabs(["Snap Receipt", "Voice Input"])

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

audit_clicked = st.button("Audit My Prices", type="primary", use_container_width=True)

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
        else:
            status = "Fair Price"

        results_data.append(
            {
                "Item": item_name,
                "Price Paid (PKR)": price_paid_num,
                "Official Cap": official_price_str,
                "Status": status,
            }
        )

    st.session_state["results_data"] = results_data
    st.session_state["audit_done"] = True

if st.session_state.get("audit_done") and st.session_state.get("results_data"):
    results_data = st.session_state["results_data"]

    st.subheader(f"Price Audit Results ({city})")
    st.dataframe(results_data, use_container_width=True)

    overcharged = [r for r in results_data if r["Status"] == "Overcharged"]

    if overcharged:
        st.error(f"You were overcharged on {len(overcharged)} item(s)!")

        if st.button(
            "Generate Official Complaint Letter",
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
                        model="gpt-4o",
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
            "Great news! All prices are within official limits. You paid fair prices today!"
        )

    st.divider()
    if st.button("Check Another Receipt", use_container_width=True):
        st.session_state["audit_done"] = False
        st.session_state["results_data"] = []
        st.rerun()
