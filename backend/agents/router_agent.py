from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from utils.logger import get_logger
from dotenv import load_dotenv

# LOGGER

load_dotenv(override=True)
logger = get_logger(__name__)


# ROUTER SCHEMA
class RouteDecision(BaseModel):
    route: Literal["product", "web", "general"]
    reason: str


# ROUTER MODEL
router_model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


# STRUCTURED ROUTER
router_llm = router_model.with_structured_output(
    RouteDecision
)


# ROUTER PROMPT
ROUTER_PROMPT = """
You are the Router Agent for an AI Search Assistant.

Your ONLY responsibility is to identify which specialized
agent should handle the user's request.

Available routes:

1. PRODUCT

Use "product" when the user is asking about products
available through the Product API.

Examples:

- Show me Pelco cameras
- Find I-PRO cameras
- Show me Potter fire alarm products
- Find Avigilon products
- Show me CCTV cameras
- Find access control products
- Show me security products
- Find cameras
- Show me fire alarm products
- Find access control devices

2. WEB

Use "web" when the user needs general information,
current information, news, technology information,
or information that requires internet search.

Examples:

- What is machine learning?
- What is LangChain?
- What are the latest AI developments?
- What are the latest cybersecurity trends?
- Who is the CEO of OpenAI?
- What happened in AI today?

3. GENERAL

Use "general" for simple conversation.

Examples:

- Hello
- Hi
- Thank you
- Goodbye
- Good morning

IMPORTANT RULES:

- Do NOT answer the user's question.
- Do NOT search for products.
- Do NOT search the web.
- Do NOT use tools.
- Only determine the correct route.
- Return exactly one route:
  product, web, or general.
"""


# ROUTER FUNCTION
def route_query(query: str) -> str:

    logger.info(
        "[ROUTER] Query received | query=%s",
        query
    )

    decision = router_llm.invoke(
        [
            {
                "role": "system",
                "content": ROUTER_PROMPT
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )

    logger.info(
        "[ROUTER] Decision | route=%s | reason=%s",
        decision.route,
        decision.reason
    )

    return decision.route