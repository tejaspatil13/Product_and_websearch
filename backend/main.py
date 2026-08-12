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
        response = router_agent.invoke({
                "messages": [{
                        "role": "user",
                        "content": request.question
                    }]
                })

        logger.info("Router agent completed successfully")
        messages = response.get("messages", [])

        
        final_message = ""
        if messages:

            final_message = messages[-1].content

        

        logger.info(
            "Tool results found: %s",
        )

        return {
            "question": request.question,
            "answer": final_message
            }

    except Exception:

        logger.exception(
        "Error while processing chat request"
    )

    