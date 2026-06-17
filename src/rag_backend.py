import os
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
os.getenv("OPENAI_API_KEY")


# Standard SPI 17-city column order used in the government price bulletin.
CITIES = [
    "Islamabad",
    "Rawalpindi",
    "Gujranwala",
    "Sialkot",
    "Lahore",
    "Faisalabad",
    "Sargodha",
    "Multan",
    "Bahawalpur",
    "Karachi",
    "Hyderabad",
    "Sukkur",
    "Larkana",
    "Peshawar",
    "Bannu",
    "Quetta",
    "Khuzdar",
]

# Number of city columns present (in order) on each of the first three pages
# of the food price table.
_CITIES_PER_PAGE = [7, 7, 3]


def initialize_rag_store():
    """Parse the government price PDF into a structured per-city price table.

    Returns a list of rows: {"description": str, "city_prices": {city: avg}}.
    """
    try:
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_path = os.path.join(ROOT_DIR, "daily_price_list.pdf")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                "daily_price_list.pdf not found. Please add this file to the nirkhnama_app/ root folder."
            )

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        items = {}
        for page_idx in range(min(3, len(documents))):
            cities_on_page = _CITIES_PER_PAGE[page_idx]
            for line in documents[page_idx].page_content.splitlines():
                line = line.strip()
                m = re.match(r"^(\d+)\s+(.*)$", line)
                if not m:
                    continue

                sl = int(m.group(1))
                rest = m.group(2)

                first_price = re.search(r"-?\d+\.\d{2}", rest)
                if not first_price:
                    continue

                description = rest[: first_price.start()].strip()
                price_tokens = re.findall(r"-?\d+\.\d{2}", rest)

                needed = price_tokens[: cities_on_page * 3]
                avgs = [
                    float(needed[i * 3 + 1])
                    for i in range(cities_on_page)
                    if i * 3 + 1 < len(needed)
                ]

                if sl not in items:
                    items[sl] = {"description": description, "avgs": []}
                if page_idx == 0 and description:
                    items[sl]["description"] = description
                items[sl]["avgs"].extend(avgs)

        table = []
        for sl in sorted(items):
            avgs = items[sl]["avgs"]
            if len(avgs) < len(CITIES):
                continue
            city_prices = {CITIES[i]: avgs[i] for i in range(len(CITIES))}
            table.append(
                {
                    "description": items[sl]["description"],
                    "city_prices": city_prices,
                }
            )

        if not table:
            raise RuntimeError("No price rows could be parsed from the PDF.")

        return table

    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to initialize RAG store: {e}") from e


def query_market_price(item_name: str, price_table, city: str) -> str:
    """Return the average price of item_name in the selected city."""
    try:
        if not price_table:
            return "Price not listed for this item."

        descriptions = [row["description"] for row in price_table]
        listing = "\n".join(f"{i}: {d}" for i, d in enumerate(descriptions))

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You match a grocery item to the single best entry in a numbered list. "
                        "The item may be in Urdu, roman Urdu, or English; translate it to English first "
                        "(e.g. 'piyaz'/'پیاز' = Onions, 'aloo'/'آلو' = Potatoes, 'tamatar'/'ٹماٹر' = Tomatoes). "
                        "Reply with ONLY the integer index of the best matching entry, "
                        "or -1 if there is no reasonable match."
                    )
                ),
                HumanMessage(content=f"Item: {item_name}\n\nList:\n{listing}"),
            ]
        )

        match = re.search(r"-?\d+", response.content)
        if not match:
            return "Price not listed for this item."

        idx = int(match.group())
        if idx < 0 or idx >= len(price_table):
            return "Price not listed for this item."

        price = price_table[idx]["city_prices"].get(city)
        if price is None or price <= 0:
            return "Price not listed for this item."

        return f"Rs. {price:.2f} (avg in {city})"

    except Exception:
        return "Price lookup failed."
