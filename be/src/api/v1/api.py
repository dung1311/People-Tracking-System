from fastapi import APIRouter
from .endpoints import cameras, tracks, search, frames, stream

api_router = APIRouter()
api_router.include_router(cameras.router, prefix="/cameras", tags=["cameras"])
api_router.include_router(tracks.router, prefix="/tracks", tags=["tracks"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(frames.router, prefix="/frames", tags=["frames"])
api_router.include_router(stream.router, prefix="/stream", tags=["stream"])

