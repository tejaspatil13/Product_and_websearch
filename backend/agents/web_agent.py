import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.tools import tool

from utils.logger import get_logger


load_dotenv()

logger = get_logger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


@tool("web_search")
def web_search(query: str) -> str:
    """
    Search the internet for current and general information.
    Use this tool for questions that require information from the web.
    """

    logger.info("Web search started: %s", query)

    try:

        response = tavily_client.search(
            query=query,
            max_results=5
        )

        logger.info("Web search completed")

        return str(response)

    except Exception:

        logger.exception("Tavily search failed")

        return "Web search is currently unavailable."


tools = [
    web_search
]


web_agent = create_agent(
    model="groq:llama-3.3-70b-versatile",
    tools=tools
)