from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.router_agent import router_agent
from utils.logger import get_logger


app = FastAPI(
    title="AI Search Assistant"
)

logger = get_logger(__name__)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
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

        logger.info("Router agent completed successfully")

        final_message = response["messages"][-1].content

        return {
            "question": request.question,
            "answer": final_message
        }

    except Exception as e:

        logger.exception(
            "Error while processing chat request"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process your request."
        )