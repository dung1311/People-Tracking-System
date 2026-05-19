"""FastAPI application entrypoint."""

import logging
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.v1.api import api_router
from core.config import settings
from database.session import engine, init_db
from database.init_data import seed_default_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting up...")

    # 1. Init database tables
    init_db()

    # 2. Seed default data (admin user, default configs)
    seed_default_data(engine)

    # 3. Try to initialize MinIO (non-fatal if unavailable)
    try:
        from core.minio_client import get_minio
        get_minio()
        logger.info("MinIO connected")
    except Exception as e:
        logger.warning("MinIO not available: %s (file uploads will fail)", e)

    # 4. Preload ML models (lazy — only if CUDA/CPU is available)
    try:
        from core.model_loader import get_model_loader
        get_model_loader()
    except Exception as e:
        logger.warning("Model preload skipped: %s", e)

    yield

    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="People Tracking System API",
    description="Multi-Camera People Tracking with Re-ID, RBAC, and MinIO storage",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for local data
os.makedirs("data/videos", exist_ok=True)
app.mount("/static", StaticFiles(directory="data"), name="static")

# API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "people-tracking-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
