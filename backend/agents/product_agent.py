import os
import json
import requests

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

from utils.logger import get_logger


load_dotenv()

logger = get_logger(__name__)

PRODUCT_API = os.getenv("PRODUCT_API")


@tool
def product_search(query: str) -> str:
    """
    Search the provided product API for products.
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

        logger.info("Product search completed")

        # -----------------------------------------
        # DEBUG: See actual API response structure
        # -----------------------------------------

        logger.info(
            "Product API response type: %s",
            type(data).__name__
        )

        # -----------------------------------------
        # Limit response size
        # -----------------------------------------

        if isinstance(data, list):

            limited_data = data[:5]

        elif isinstance(data, dict):

            limited_data = {}

            # Keep only first 5 items for any list
            # found inside the response.

            for key, value in data.items():

                if isinstance(value, list):

                    limited_data[key] = value[:5]

                else:

                    limited_data[key] = value

        else:

            limited_data = data

        # -----------------------------------------
        # Convert to JSON
        # -----------------------------------------

        result = json.dumps(
            limited_data,
            ensure_ascii=False,
            default=str
        )

        # -----------------------------------------
        # Final character safety limit
        # -----------------------------------------

        MAX_CHARS = 8000

        if len(result) > MAX_CHARS:

            logger.warning(
                "Product response is large: %s characters. "
                "Truncating to %s characters.",
                len(result),
                MAX_CHARS
            )

            result = result[:MAX_CHARS]

        logger.info(
            "Product result sent to LLM: %s characters",
            len(result)
        )

        return result

    except requests.exceptions.RequestException as e:

        logger.exception(
            "Product API request failed: %s",
            e
        )

        return "Product search is currently unavailable."

    except Exception as e:

        logger.exception(
            "Unexpected product search error: %s",
            e
        )

        return "An error occurred while searching for products."


# -----------------------------------------
# Product LLM
# -----------------------------------------

product_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# -----------------------------------------
# Product Agent
# -----------------------------------------

product_agent = create_agent(

    model=product_model,

    tools=[
        product_search
    ],

    system_prompt="""
You are the Product Search Agent.

Your job is to search the provided product database.

You MUST use product_search for product-related
questions.

You must ONLY use information returned by the
product_search tool.

Never invent:

- Product names
- Prices
- Manufacturers
- SKUs
- Specifications
- URLs

If products are found:

Give the user a clean and concise response.

Show useful information such as:

- Product name
- Manufacturer
- Category
- Price
- SKU
- Description
- Product URL

Only show fields that are actually present
in the search results.

If no products are found:

Say:

"I couldn't find any products matching your request.
Try searching with a different product name,
manufacturer, category, or keyword."

Do not expose raw API responses.

Do not expose technical errors.

Do not call any web search.

The product_search tool is the only source
for product information.
"""
)