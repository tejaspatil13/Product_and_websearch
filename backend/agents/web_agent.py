import os

from dotenv import load_dotenv
from tavily import TavilyClient

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from utils.logger import get_logger





# CONFIG
load_dotenv(override=True)
logger = get_logger(__name__)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")





# TAVILY CLIENT

if not TAVILY_API_KEY:

    logger.error("[WEB] TAVILY_API_KEY is not configured")
    tavily_client = None


else:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)




# WEB SEARCH TOOL
@tool("web_search")
def web_search(query: str) -> str:
    """
    Search the internet for current and general information.
    Use this tool for questions that require information
    from the web.
    """

    logger.info(
        "[WEB_TOOL] Search started | query=%s", query)

    if not tavily_client:
        logger.error("[WEB_TOOL] Tavily client is unavailable")
        return "Web search is currently unavailable."

    try:

        
        # TAVILY SEARCH
        response = tavily_client.search(
            query=query,
            max_results=5
        )

        logger.info("[WEB_TOOL] Search completed")
        results = response.get("results",[])
        logger.info("[WEB_TOOL] Results received | count=%s",len(results))

        

        if not results:

            return "No relevant web results were found."

        
        # FORMAT SEARCH RESULTS
        

        formatted_results = []

        for result in results:

            title = result.get(
                "title",
                ""
            )

            content = result.get(
                "content",
                ""
            )

            url = result.get(
                "url",
                ""
            )

            formatted_results.append(
                f"Title: {title}\n"
                f"Content: {content}\n"
                f"URL: {url}"
            )

            

        return "\n\n".join(
            formatted_results
        )

    except Exception as e:
        logger.exception(
            "[WEB_TOOL] Tavily search failed | error=%s",
            e
        )
        return "Web search is currently unavailable."



# WEB MODEL
web_model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


# WEB PROMPT
WEB_PROMPT = """
You are a Web Search Agent.
Your job is to answer general and current questions
using information from the internet.

You have one tool: web_search

IMPORTANT RULES:
1. Use web_search when the question requires
   internet information.
2. Use ONLY information returned by web_search.
3. Do not invent information.
4. Do not invent search results.
5. For current, latest, recent, or time-sensitive
   information, always use web_search.
6. Give a clear and concise answer.
7. Do not answer product-search questions.
8. Do not use the Product API.
9. If web_search returns no useful results,
   clearly tell the user.
10. Do not expose internal tool reasoning.
"""



# WEB REACT AGENT
web_agent = create_agent(
    model=web_model,
    tools=[web_search],
    system_prompt=WEB_PROMPT
)


# ============================================================
# WEB AGENT RUNNER
# ============================================================

def run_web_agent(query: str) -> dict:

    logger.info(
        "[WEB_AGENT] Started | query=%s",
        query
    )

    try:

       
        # RUN WEB AGENT
        response = web_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }
        )

        logger.info(
            "[WEB_AGENT] Agent execution completed"
        )

       
        # GET MESSAGES
        

        messages = response.get(
            "messages",
            []
        )

        if not messages:

            logger.warning(
                "[WEB_AGENT] No messages returned"
            )

            return {
                "answer": "I could not find an answer.",
                "products": []
            }

        
        # GET FINAL MESSAGE
       
        final_message = messages[-1]

        answer = final_message.content

        logger.info(
            "[WEB_AGENT] Final answer generated"
        )

        # ====================================================
        # RESPONSE FORMAT
        # ====================================================

        return {
            "answer": answer,
            "products": []
        }

    except Exception as e:

        logger.exception(
            "[WEB_AGENT] Agent execution failed | error=%s",
            e
        )

        return {
            "answer": "Sorry, I was unable to perform the web search.",
            "products": []
        }