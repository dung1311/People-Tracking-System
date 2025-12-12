import logging

from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn

from database.session import init_db
from core.log_setup import configure_logging
from api.v1.api import api_router
import models  # Ensure models are registered

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    logger.info("Application started!")
    
    yield
    
    logger.info("Application is shutting down...")
    

app = FastAPI(
    title="People Tracking System",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "root"}

if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
