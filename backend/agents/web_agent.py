import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from utils.logger import get_logger


load_dotenv(override=True)

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




web_model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


web_agent = create_agent(
    model=web_model,
    tools=[web_search],
    system_prompt="""
You are a Web Search Agent.

Your job is to answer general and current questions
using information from the internet.

IMPORTANT RULES:

1. Use web_search when the question requires internet information.

2. Use ONLY information from the search results.

3. Do not invent information.

4. Give a clear and concise answer.

5. Do not answer product-search questions.
"""
)