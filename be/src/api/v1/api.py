"""API v1 router — aggregates all endpoint modules."""

from fastapi import APIRouter

from .endpoints import auth, cameras, configs, frames, search, sessions, stream, tracks, users, ws

api_router = APIRouter()

# Auth (no prefix — login/register)
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])

# User management
api_router.include_router(users.router, prefix="/users", tags=["users"])

# Camera management
api_router.include_router(cameras.router, prefix="/cameras", tags=["cameras"])

# Tracking configs
api_router.include_router(configs.router, prefix="/configs", tags=["configs"])

# Tracking sessions (MCT lifecycle)
api_router.include_router(sessions.router, prefix="/sessions", tags=["sessions"])

# Track queries
api_router.include_router(tracks.router, prefix="/tracks", tags=["tracks"])

# Person search
api_router.include_router(search.router, prefix="/search", tags=["search"])

# Frame extraction
api_router.include_router(frames.router, prefix="/frames", tags=["frames"])

# Single camera stream
api_router.include_router(stream.router, prefix="/stream", tags=["stream"])

# WebSocket
api_router.include_router(ws.router, prefix="/ws", tags=["websocket"])
