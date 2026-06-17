import base64
import json
import os
import tempfile

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
os.getenv("OPENAI_API_KEY")

client = OpenAI()


def extract_from_receipt_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> list:
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "You are an OCR assistant for Pakistani grocery receipts. "
                                "Extract every item and its price from this image. "
                                "Always translate each item name into its common ENGLISH grocery name "
                                "(e.g. 'piyaz' or 'پیاز' becomes 'Onions', 'aloo' becomes 'Potatoes', "
                                "'tamatar' becomes 'Tomatoes'). "
                                "Return ONLY a valid JSON array with no markdown, no backticks, no explanation. "
                                "Each object must have exactly two keys: 'item_name' (string, in English) and 'price_paid' "
                                "(number in PKR). Example: [{\"item_name\": \"Tomatoes\", \"price_paid\": 120}]. "
                                "If no items found, return []."
                            ),
                        },
                    ],
                }
            ],
        )

        response_text = response.choices[0].message.content

        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except Exception:
            return []

    except Exception:
        return []


def extract_from_voice(audio_bytes: bytes) -> list:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="ur",
                ).text
        finally:
            os.unlink(tmp_path)

        if not transcript or not transcript.strip():
            return []

        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract grocery item names and prices from spoken text. "
                        "The text may be in Urdu, English, or a mix. Always translate each item name "
                        "into its common ENGLISH grocery name (e.g. 'piyaz'/'پیاز' becomes 'Onions', "
                        "'aloo'/'آلو' becomes 'Potatoes', 'tamatar'/'ٹماٹر' becomes 'Tomatoes'). "
                        "Always return a JSON object with one key 'items' containing an array. "
                        "Each item in the array must have 'item_name' (string, in English) and 'price_paid' "
                        "(number in PKR). If no items or prices are mentioned, return {\"items\": []}."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Extract all grocery items and their prices from this spoken text: "
                        f"\"{transcript}\""
                    ),
                },
            ],
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        items = data.get("items", [])
        return items if isinstance(items, list) else []

    except Exception:
        return []
