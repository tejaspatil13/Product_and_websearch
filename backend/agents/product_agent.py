import os
import requests
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.tools import tool
from utils.logger import get_logger


# ============================================================
# CONFIG
# ============================================================

load_dotenv(override=True)

logger = get_logger(__name__)
PRODUCT_API = os.getenv("PRODUCT_API")


# ============================================================
# PRODUCT SCHEMAS
# ============================================================

class Product(BaseModel):
    id: str
    name: str
    sku: Optional[str] = None
    manufacturer: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None


class ProductResponse(BaseModel):
    answer: str
    products: list[Product]


# ============================================================
# PRODUCT EXTRACTION
# ============================================================

def extract_products(data: dict, limit: int = 20) -> list[dict]:
    """
    Extract up to `limit` valid products from the Product API response.
    """

    products = []

    for category_group in data.get("Products", []):

        if not isinstance(category_group, dict):
            continue

        for category_products in category_group.values():

            if not isinstance(category_products, list):
                continue

            for product in category_products:

                if not isinstance(product, dict):
                    continue

                if not product.get("ProductID"):
                    continue

                if not product.get("ProductName"):
                    continue

                products.append(product)

                # Stop once we have 20 products
                if len(products) >= limit:
                    return products

    return products


# ============================================================
# PRODUCT NORMALIZATION
# ============================================================

def normalize_product(product: dict) -> dict:
    """
    Convert API product format into our application's Product format.
    """

    compact_product = {
        "id": str(product["ProductID"]),
        "name": str(product["ProductName"]),
    }

    # SKU
    if product.get("SKU"):
        compact_product["sku"] = str(product["SKU"])

    # MANUFACTURER
    if product.get("ManufacturerName"):
        compact_product["manufacturer"] = str(
            product["ManufacturerName"]
        )

    # DESCRIPTION
    if product.get("ShortDescription"):
        compact_product["description"] = str(
            product["ShortDescription"]
        )

    # IMAGE
    if product.get("ProductImage"):
        # Take only the first image if multiple URLs exist
        compact_product["image_url"] = (
            str(product["ProductImage"])
            .split("|")[0]
            .strip()
        )

    return compact_product


# ============================================================
# PRODUCT SEARCH TOOL
# ============================================================

@tool
def product_search(query: str) -> list[dict]:
    """
    Search the Product API and return up to 20 products.

    Use this tool for:
    - Product searches
    - Manufacturers
    - Categories
    - CCTV products
    - Security cameras
    - Fire alarm products
    - Access control products
    """

    logger.info(
        "[PRODUCT_TOOL] Search started | query=%s",
        query
    )

    # --------------------------------------------------------
    # API CONFIGURATION
    # --------------------------------------------------------

    if not PRODUCT_API:

        logger.error(
            "[PRODUCT_TOOL] PRODUCT_API is not configured"
        )

        return []

    try:

        # ----------------------------------------------------
        # API REQUEST
        # ----------------------------------------------------

        logger.info("[PRODUCT_API] Request started")

        response = requests.post(
            f"{PRODUCT_API}/search",
            json={
                "query": query,
                "category_name": "",
                "subcategory_name": "",
                "manufacturer_name": "",
                "userId": 0,
            },
            timeout=30,
        )

        response.raise_for_status()

        logger.info(
            "[PRODUCT_API] Response received | status=%s",
            response.status_code
        )

        # ----------------------------------------------------
        # PARSE JSON
        # ----------------------------------------------------

        data = response.json()

        # ----------------------------------------------------
        # EXTRACT PRODUCTS
        # ----------------------------------------------------

        products = extract_products(
            data,
            limit=20
        )

        logger.info(
            "[PRODUCT_TOOL] Products extracted | count=%s",
            len(products)
        )

        # ----------------------------------------------------
        # NORMALIZE PRODUCTS
        # ----------------------------------------------------

        result = []

        for product in products:

            compact_product = normalize_product(product)

            result.append(compact_product)

        logger.info(
            "[PRODUCT_TOOL] Returning products | count=%s",
            len(result)
        )

        return result

    # --------------------------------------------------------
    # REQUEST ERROR
    # --------------------------------------------------------

    except requests.RequestException as e:

        logger.exception(
            "[PRODUCT_API] Request failed | error=%s",
            e
        )

        return []

    # --------------------------------------------------------
    # GENERAL ERROR
    # --------------------------------------------------------

    except Exception as e:

        logger.exception(
            "[PRODUCT_TOOL] Search failed | error=%s",
            e
        )

        return []


# ============================================================
# PRODUCT AGENT RUNNER
# ============================================================

def run_product_agent(query: str) -> dict:

    logger.info(
        "[PRODUCT_AGENT] Started | query=%s",
        query
    )

    try:

        # ----------------------------------------------------
        # CLEAN QUERY
        # ----------------------------------------------------

        query = query.strip()

        if not query:

            logger.warning(
                "[PRODUCT_AGENT] Empty query received"
            )

            return {
                "answer": "Please enter a product search.",
                "products": []
            }

        # ----------------------------------------------------
        # PRODUCT TOOL EXECUTION
        # ----------------------------------------------------

        logger.info(
            "[PRODUCT_AGENT] Calling product_search tool"
        )

        products = product_search.invoke(query)

        logger.info(
            "[PRODUCT_AGENT] Product tool completed | products=%s",
            len(products)
        )

        # ----------------------------------------------------
        # NO PRODUCTS
        # ----------------------------------------------------

        if not products:

            logger.info(
                "[PRODUCT_AGENT] No products found"
            )

            return ProductResponse(
                answer=f"I couldn't find any products matching '{query}'.",
                products=[]
            ).model_dump()

        # ----------------------------------------------------
        # PYDANTIC VALIDATION
        # ----------------------------------------------------

        validated_products = []

        for product in products:

            try:

                validated_product = Product(
                    **product
                )

                validated_products.append(
                    validated_product
                )

            except Exception as e:

                logger.warning(
                    "[PRODUCT_AGENT] Invalid product skipped | error=%s",
                    e
                )
 
        # ----------------------------------------------------
        # FINAL ANSWER
        # ----------------------------------------------------
        validated_products = validated_products[:5]

        count = len(validated_products)

        answer = (
            f"Here are {count} products I found "
            f"for '{query}'."
        )

        logger.info(
            "[PRODUCT_AGENT] Response generated | products=%s",
            count
        )

        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        response = ProductResponse(
            answer=answer,
            products=validated_products
        )

        logger.info(
            "[PRODUCT_AGENT] Completed successfully"
        )

        return response.model_dump()

    # --------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------

    except Exception as e:

        logger.exception(
            "[PRODUCT_AGENT] Execution failed | error=%s",
            e
        )

        return {
            "answer": (
                "Sorry, I was unable to search for products "
                "at the moment."
            ),
            "products": []
        }