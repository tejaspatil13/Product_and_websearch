import os
import requests

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from utils.logger import get_logger


# ============================================================
# CONFIG
# ============================================================

load_dotenv(override=True)

logger = get_logger(__name__)

PRODUCT_API = os.getenv("PRODUCT_API")


# ============================================================
# LLM OUTPUT SCHEMA
# ============================================================

class ProductOutput(BaseModel):
    product_name: str = ""
    manufacturer: str = ""
    sku: str = ""
    part_number: str = ""
    category: str = ""
    subcategory: str = ""
    description: str = ""
    price: float = 0
    status: str = ""
    compatibility: str = ""
    product_image_url: str = ""


class ProductSearchOutput(BaseModel):
    message: str
    products: List[ProductOutput]


# ============================================================
# PRODUCT SEARCH TOOL
# ============================================================

@tool
def product_search(query: str) -> list:
    """
    Search the product API and return maximum 5 products.
    """

    logger.info("Product search started: %s", query)

    try:

        response = requests.post(
            f"{PRODUCT_API}/search",
            json={
                "query": query,
                "category_name": "",
                "subcategory_name": "",
                "manufacturer_name": "",
                "userId": 0
            },
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        logger.info("Product API request completed")

        products = []

        # ====================================================
        # EXTRACT PRODUCTS FROM API RESPONSE
        # ====================================================

        if isinstance(data, dict):

            product_data = data.get("Products", {})

            if isinstance(product_data, dict):

                for category_products in product_data.values():

                    if isinstance(category_products, list):
                        products.extend(category_products)

            elif isinstance(product_data, list):

                products = product_data

        elif isinstance(data, list):

            products = data

        # ====================================================
        # LIMIT TO 5 PRODUCTS
        # ====================================================

        products = products[:5]

        logger.info(
            "PRODUCTS EXTRACTED: %s",
            len(products)
        )

        if not products:
            return []

        # IMPORTANT:
        # Return Python list/dicts.
        # Don't json.dumps().
        # Don't create Pydantic objects here.

        return products

    except Exception:

        logger.exception(
            "Product search failed"
        )

        return []


# ============================================================
# PRODUCT LLM
# ============================================================

product_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ============================================================
# STRUCTURED OUTPUT MODEL
# ============================================================

structured_product_model = product_model.with_structured_output(
    ProductSearchOutput
)


# ============================================================
# PRODUCT FORMATTER
# ============================================================
product_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a Product Response Formatter.

You will receive products returned directly from the Product API.

Use ONLY the information provided.

Rules:
- Never invent product information.
- Never invent specifications.
- Never invent prices.
- Never invent compatibility.
- If a field is missing, return an empty value.
- Maximum 5 products will be provided.
- Do not use web search.

For each product extract:

- Product Name
- Manufacturer
- SKU
- Part Number
- Category
- Subcategory
- Description
- Price
- Status
- Compatibility
- Product Image URL

For description, combine the available short and detailed
description fields when appropriate.

Return the result using the provided structured schema.
"""
    ),
    (
        "human",
        "Products returned by the Product API:\n\n{products}"
    )
])

# ============================================================
# PRODUCT CHAIN
# ============================================================

product_chain = product_prompt | structured_product_model


# ============================================================
# PRODUCT AGENT
# ============================================================

product_agent = create_agent(
    model=product_model,
    tools=[product_search],
    system_prompt="""
You are a Product Search Agent.

For every product-related query:

1. ALWAYS call product_search.
2. Do not answer product questions from your own knowledge.
3. Use only information returned by product_search.
4. Never invent product information.
5. Do not use web search.

The product_search tool returns a maximum of 5 products.

After receiving the products, use the product information to
produce the final product response.

If no products are returned, say:

"I couldn't find any products matching your request."
"""
)