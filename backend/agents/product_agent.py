import os
import requests
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda

from utils.logger import get_logger


# ============================================================
# CONFIG
# ============================================================

load_dotenv(override=True)

logger = get_logger(__name__)

PRODUCT_API = os.getenv("PRODUCT_API")


# ============================================================
# PYDANTIC OUTPUT SCHEMA
# ============================================================

class ProductOutput(BaseModel):

    product_name: Optional[str] = None
    manufacturer: Optional[str] = None
    sku: Optional[str] = None
    part_number: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    description: Optional[str] = None
    price: Optional[Any] = None
    status: Optional[str] = None
    compatibility: Optional[str] = None
    product_image_url: Optional[str] = None

    summary: str = ""


class ProductSearchOutput(BaseModel):

    message: str = ""

    products: List[ProductOutput] = Field(
        default_factory=list
    )


# ============================================================
# PYDANTIC OUTPUT PARSER
# ============================================================

parser = PydanticOutputParser(
    pydantic_object=ProductSearchOutput
)


# ============================================================
# PRODUCT SEARCH TOOL
# ============================================================

@tool
def product_search(query: str) -> list:
    """
    Search the Product API and return maximum 5 products.
    """

    logger.info(
        "Product search started: %s",
        query
    )

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

        logger.info(
            "Product API request completed"
        )

        products = []

        # ====================================================
        # EXTRACT PRODUCTS
        # ====================================================

        if isinstance(data, dict):

            product_data = data.get(
                "Products",
                {}
            )

            if isinstance(product_data, dict):

                for category_products in product_data.values():

                    if isinstance(category_products, list):

                        products.extend(
                            category_products
                        )

            elif isinstance(product_data, list):

                products = product_data

        elif isinstance(data, list):

            products = data

        # ====================================================
        # MAXIMUM 5 PRODUCTS
        # ====================================================

        products = products[:5]

        logger.info(
            "Products extracted: %s",
            len(products)
        )

        return products

    except Exception:

        logger.exception(
            "Product search failed"
        )

        return []


# ============================================================
# LLM
# ============================================================

product_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ============================================================
# PROMPT TEMPLATE
# ============================================================

product_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Product Response Formatter.

You will receive products directly from a Product API.

Your job is to format those products into the required
structured output.

IMPORTANT RULES:

1. Use ONLY information present in the API response.

2. NEVER invent product information.

3. NEVER invent specifications.

4. NEVER invent prices.

5. NEVER invent compatibility.

6. NEVER add information from your own knowledge.

7. Maximum 5 products can be returned.

8. For each product, include ONLY fields for which
   information is actually available in the API response.

9. Do NOT create empty fields for information that is
   not available.

10. Create one "summary" for each product.

11. The summary must be based ONLY on the available
    information for that product.

12. Do not use web search.

13. If no products are provided, return an empty products list.

14. Return the result according to the Pydantic schema.

{format_instructions}
"""
        ),

        (
            "human",
            """
User query:

{query}

Products returned by the Product API:

{products}
"""
        )
    ]
)


# ============================================================
# CHAIN
# ============================================================

product_chain = (
    product_prompt
    | product_model
    | parser
)


# ============================================================
# PRODUCT RUNNER
# ============================================================

def run_product_search(query: str) -> dict:
    """
    Receives the user's product query.

    Flow:

    query
       ↓
    product_search tool
       ↓
    Product API
       ↓
    maximum 5 products
       ↓
    PromptTemplate
       ↓
    LLM
       ↓
    PydanticOutputParser
       ↓
    dictionary
    """

    logger.info(
        "Running product search for query: %s",
        query
    )

    # ========================================================
    # CALL PRODUCT TOOL
    # ========================================================

    products = product_search.invoke(
        query
    )

    # Safety limit
    products = products[:5]

    logger.info(
        "Sending %s products to LLM",
        len(products)
    )

    # ========================================================
    # NO PRODUCTS
    # ========================================================

    if not products:

        return {
            "message": "I couldn't find any products matching your request.",
            "products": []
        }

    # ========================================================
    # RUN CHAIN
    # ========================================================

    result = product_chain.invoke(
        {
            "query": query,
            "products": products,
            "format_instructions": parser.get_format_instructions()
        }
    )

    # ========================================================
    # PYDANTIC → DICTIONARY
    # ========================================================

    return result.model_dump(
        exclude_none=True
    )


# ============================================================
# PRODUCT AGENT
# ============================================================

product_agent = RunnableLambda(
    run_product_search
)