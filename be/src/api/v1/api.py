from fastapi import APIRouter
from .endpoints import cameras, tracks, search, frames, stream, auth, users, sessions, configs, ws

api_router = APIRouter()

# New authentications & administration routers
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])

# Enhanced resources routers
api_router.include_router(cameras.router, prefix="/cameras", tags=["cameras"])
api_router.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
api_router.include_router(configs.router, prefix="/configs", tags=["configs"])
api_router.include_router(ws.router, prefix="/ws", tags=["ws"])

# Original routers
api_router.include_router(tracks.router, prefix="/tracks", tags=["tracks"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(frames.router, prefix="/frames", tags=["frames"])
api_router.include_router(stream.router, prefix="/stream", tags=["stream"])
