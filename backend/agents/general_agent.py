from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from utils.logger import get_logger


# ============================================================
# CONFIG
# ============================================================

load_dotenv(override=True)

logger = get_logger(__name__)


# ============================================================
# GENERAL MODEL
# ============================================================

general_model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7
)


# ============================================================
# GENERAL AGENT PROMPT
# ============================================================

GENERAL_PROMPT = """
You are the General Conversation Agent for an AI Search Assistant.

Your responsibility is to handle simple conversational queries
that do not require product search or web search.

Examples:

- Hello
- Hi
- Hey
- How are you?
- Good morning
- Thank you
- Thanks
- Goodbye
- What can you do?

RULES:

1. Respond naturally and helpfully.

2. Keep simple conversations concise.

3. Do not search for products.

4. Do not perform web searches.

5. Do not invent product information.

6. If the user asks a product-related question, that query
   should normally be handled by the Product Agent.

7. If the user asks for current information or information
   requiring internet search, that query should normally be
   handled by the Web Agent.

8. Do not mention internal agents, routing, tools, or APIs
   to the user.

9. Answer only the user's conversational request.
"""


# ============================================================
# GENERAL AGENT
# ============================================================

general_agent = create_agent(
    model=general_model,
    tools=[],
    system_prompt=GENERAL_PROMPT
)


# ============================================================
# GENERAL AGENT FUNCTION
# ============================================================

def run_general_agent(query: str) -> str:

    logger.info(
        "[GENERAL_AGENT] Started | query=%s",
        query
    )

    try:

        response = general_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }
        )

        answer = response["messages"][-1].content

        logger.info(
            "[GENERAL_AGENT] Response generated"
        )

        return answer

    except Exception as e:

        logger.exception(
            "[GENERAL_AGENT] Failed | error=%s",
            e
        )

        return "Sorry, I couldn't process your request."