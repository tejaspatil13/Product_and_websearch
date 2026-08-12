from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.router_agent import router_agent
from utils.logger import get_logger


app = FastAPI()

logger = get_logger(__name__)


class ChatRequest(BaseModel):
    question: str


@app.get("/Health")
def home():

    logger.info("Home endpoint called")

    return {
        "message": "AI Search Assistant is running"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    logger.info(
        "Chat request received: %s",
        request.question
    )

    try:

        response = router_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": request.question
                    }
                ]
            }
        )

        logger.info(
            "Router agent completed successfully"
        )

        messages = response.get(
            "messages",
            []
        )

        # ====================================================
        # FIND PRODUCT TOOL RESULT
        # ====================================================

        for message in messages:

            if getattr(message, "name", None) == "run_product_agent":

                product_result = message.content

                logger.info(
                    "Product result found"
                )

                return {
                    "question": request.question,
                    "result": product_result
                }

        # ====================================================
        # FALLBACK FOR WEB AGENT
        # ====================================================

        if messages:

            final_message = messages[-1].content

            return {
                "question": request.question,
                "answer": final_message
            }

        return {
            "question": request.question,
            "answer": "No response generated."
        }

    except Exception as e:

        logger.exception(
            "Error while processing chat request"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )