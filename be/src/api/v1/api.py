from fastapi import APIRouter
from .endpoints import cameras, tracks

api_router = APIRouter()
api_router.include_router(cameras.router, prefix="/cameras", tags=["cameras"])
api_router.include_router(tracks.router, prefix="/tracks", tags=["tracks"])

