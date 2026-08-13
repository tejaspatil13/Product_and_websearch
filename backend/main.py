from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from agents.router_agent import route_query
from agents.product_agent import run_product_agent
from agents.web_agent import run_web_agent
from agents.general_agent import run_general_agent

from utils.logger import get_logger


# ============================================================
# CONFIG
# ============================================================

logger = get_logger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
    ],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REQUEST SCHEMA
class ChatRequest(BaseModel):
    query: str


# CHAT ENDPOINT
@app.post("/chat")
def chat(request: ChatRequest):

    query = request.query.strip()

    logger.info(
        "[MAIN] Query received | query=%s",
        query
    )


    # ROUTER
    route = route_query(query)

    logger.info(
        "[MAIN] Router selected | route=%s",
        route
    )

    
    # PRODUCT AGENT
    if route == "product":             

        logger.info(
            "[MAIN] Calling Product Agent"
        )

        return run_product_agent(query)




    # WEB AGENT
    if route == "web":

        logger.info(
            "[MAIN] Calling Web Agent"
        )
        return run_web_agent(query)





    if route == "general":

        logger.info(
        "[MAIN] Calling General Agent"
        )
        return {
        "answer": run_general_agent(query),
        "products": []
    }
    