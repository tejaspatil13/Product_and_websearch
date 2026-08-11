from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

from agents.product_agent import product_agent
from agents.web_agent import web_agent
from utils.logger import get_logger


logger = get_logger(__name__)


@tool
def run_product_agent(query: str) -> str:
    """
    Use this tool when the user is asking about products,
    manufacturers, categories, or product information.
    """

    logger.info("Router selected PRODUCT agent")

    response = product_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
    )

    return response["messages"][-1].content


@tool
def run_web_agent(query: str) -> str:
    """
    Use this tool when the user is asking a general question
    or needs information from the internet.
    """

    logger.info("Router selected WEB agent")

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

    return response["messages"][-1].content


router_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


router_agent = create_agent(
    model=router_model,
    tools=[
        run_product_agent,
        run_web_agent
    ],
    system_prompt="""
You are the main routing AI for an AI Search Assistant.

You have two available tools:

1. run_product_agent
2. run_web_agent

Your job is to decide which agent should handle
the user's question.

PRODUCT queries:
- Finding products
- Product manufacturers
- Product categories
- Product specifications
- Product searches

Examples:

"Show me Pelco cameras"
"Find I-PRO cameras"
"Show me Potter fire alarm products"
"Find Avigilon products"

For these, use run_product_agent.

GENERAL WEB queries:
- General knowledge
- Current information
- News
- Latest developments
- Technology information
- Questions requiring internet information

Examples:

"What is machine learning?"
"What are the latest AI developments?"
"What is LangChain?"
"What are the latest cybersecurity trends?"

For these, use run_web_agent.

IMPORTANT:

- Select only ONE agent for a normal user query.
- Do not call both agents unnecessarily.
- Do not answer product questions yourself.
- Do not answer web-search questions yourself when web information
  is required.
- Let the selected agent produce the answer.
- Return the selected agent's answer to the user.

For simple conversation such as:
"Hello"
"Hi"
"Thank you"

you may respond naturally without using either search agent.
"""
)