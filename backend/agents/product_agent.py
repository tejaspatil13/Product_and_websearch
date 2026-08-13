import os
import requests

from typing import Any, Optional

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

def extract_products(obj: Any) -> list[dict]:
    """
    Recursively find product objects inside the Product API
    response.
    """

    products = []

    # --------------------------------------------------------
    # DICTIONARY
    # --------------------------------------------------------

    if isinstance(obj, dict):

        if (
            "ProductID" in obj
            and "ProductName" in obj
        ):
            products.append(obj)

        for value in obj.values():

            products.extend(
                extract_products(value)
            )

    # --------------------------------------------------------
    # LIST
    # --------------------------------------------------------

    elif isinstance(obj, list):

        for item in obj:

            products.extend(
                extract_products(item)
            )

    return products


# ============================================================
# PRODUCT SEARCH TOOL
# ============================================================

@tool
def product_search(query: str) -> list[dict]:
    """
    Search the Product API and return the top 5 products.

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

    # ========================================================
    # API CONFIGURATION
    # ========================================================

    if not PRODUCT_API:

        logger.error(
            "[PRODUCT_TOOL] PRODUCT_API is not configured"
        )

        return []

    try:

        # ====================================================
        # API REQUEST
        # ====================================================

        logger.info(
            "[PRODUCT_API] Request started"
        )

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

        logger.info(
            "[PRODUCT_API] Response received | status=%s",
            response.status_code
        )

        # ====================================================
        # PARSE JSON
        # ====================================================

        data = response.json()

        # ====================================================
        # EXTRACT PRODUCTS
        # ====================================================

        products = extract_products(data)

        logger.info(
            "[PRODUCT_TOOL] Raw products extracted | count=%s",
            len(products)
        )

        # ====================================================
        # TOP 5
        # ====================================================

        products = products[:5]

        logger.info(
            "[PRODUCT_TOOL] Top-K applied | k=5 | count=%s",
            len(products)
        )

        # ====================================================
        # COMPACT PRODUCT DATA
        # ====================================================

        result = []

        for product in products:

            product_id = product.get("ProductID")
            product_name = product.get("ProductName")

            # ------------------------------------------------
            # REQUIRED FIELDS
            # ------------------------------------------------

            if not product_id or not product_name:

                logger.warning(
                    "[PRODUCT_TOOL] Invalid product skipped"
                )

                continue

            # ------------------------------------------------
            # BASIC DATA
            # ------------------------------------------------

            compact_product = {
                "id": str(product_id),
                "name": str(product_name)
            }

            # ------------------------------------------------
            # SKU
            # ------------------------------------------------

            sku = product.get("SKU")

            if sku:
                compact_product["sku"] = str(sku)

            # ------------------------------------------------
            # MANUFACTURER
            # ------------------------------------------------

            manufacturer = product.get(
                "ManufacturerName"
            )

            if manufacturer:

                compact_product["manufacturer"] = str(
                    manufacturer
                )

            # ------------------------------------------------
            # DESCRIPTION
            # ------------------------------------------------

            description = product.get(
                "ShortDescription"
            )

            if description:

                compact_product["description"] = str(
                    description
                )

            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            image_url = product.get(
                "ProductImage"
            )

            if image_url:

                compact_product["image_url"] = str(
                    image_url
                )

            # ------------------------------------------------
            # ADD PRODUCT
            # ------------------------------------------------

            result.append(
                compact_product
            )

        logger.info(
            "[PRODUCT_TOOL] Returning products | count=%s",
            len(result)
        )

        return result

    # ========================================================
    # REQUEST ERROR
    # ========================================================

    except requests.RequestException as e:

        logger.exception(
            "[PRODUCT_API] Request failed | error=%s",
            e
        )

        return []

    # ========================================================
    # GENERAL ERROR
    # ========================================================

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

        # ====================================================
        # CLEAN QUERY
        # ====================================================

        query = query.strip()

        if not query:

            logger.warning(
                "[PRODUCT_AGENT] Empty query received"
            )

            return {
                "answer": "Please enter a product search.",
                "products": []
            }

        # ====================================================
        # PRODUCT TOOL EXECUTION
        # ====================================================

        logger.info(
            "[PRODUCT_AGENT] Calling product_search tool"
        )

        products = product_search.invoke(
            query
        )

        logger.info(
            "[PRODUCT_AGENT] Product tool completed | products=%s",
            len(products)
        )

        # ====================================================
        # NO PRODUCTS
        # ====================================================

        if not products:

            logger.info(
                "[PRODUCT_AGENT] No products found"
            )

            return ProductResponse(
                answer=(
                    f"I couldn't find any products matching "
                    f"'{query}'."
                ),
                products=[]
            ).model_dump()

        # ====================================================
        # BUILD PRODUCT SCHEMAS
        # ========================================================

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

        # ====================================================
        # FINAL ANSWER
        # ====================================================

        count = len(validated_products)

        answer = (
            f"Here are {count} products I found "
            f"for '{query}'."
        )

        logger.info(
            "[PRODUCT_AGENT] Response generated | products=%s",
            count
        )

        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        response = ProductResponse(
            answer=answer,
            products=validated_products
        )

        logger.info(
            "[PRODUCT_AGENT] Completed successfully"
        )

        return response.model_dump()

    # ========================================================
    # ERROR HANDLING
    # ========================================================

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