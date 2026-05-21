import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.v1.api import api_router
from database.session import init_db
from core.model_loader import get_model_loader

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    init_db()
    # Initialize MinIO
    from core.minio_client import get_minio_client
    get_minio_client()
    # Preload models
    get_model_loader()
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="MCT API",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan
)

# CORS
origins = [
    "http://localhost:5173", # Vite default
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
# Mount data folder to serve videos and images (if we decide to serve cropped images later)
# Assuming 'mct_demo.mp4' is in the root or data folder.
# The user's file list showed 'mct_demo.mp4' in root.
# We should probably serve the root or a specific media folder.
# Let's create a 'media' folder and symlink or move the video, or just serve root (risky but okay for demo).
# Safer: Serve 'data' and move video there.
# The user said `mct_demo.mp4` is in root.
# Let's mount the current directory as static for demo purposes, or better, mount `data` and ensure the video is accessible.
# Actually, the user asked to "show video".
# I'll create a symlink in `data/videos` for `mct_demo.mp4` if it's not there.
os.makedirs("data/videos", exist_ok=True)
if os.path.exists("mct_demo.mp4") and not os.path.exists("data/videos/mct_demo.mp4"):
    # Just copy or symlink. 
    # Python symlink might fail on some filesystems, but this is Linux.
    try:
        os.symlink(os.path.abspath("mct_demo.mp4"), "data/videos/mct_demo.mp4")
    except OSError:
        pass # Ignore if exists

app.mount("/static", StaticFiles(directory="data"), name="static")

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
