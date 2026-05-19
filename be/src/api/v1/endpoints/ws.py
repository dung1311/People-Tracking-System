"""WebSocket endpoints for real-time streaming."""

import asyncio
import base64
import json
import logging
from typing import Dict, Set

import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlmodel import Session

from core.security import decode_access_token
from core.session_manager import get_session_manager
from database.session import get_session, engine

router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections per session."""

    def __init__(self):
        self._connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, session_id: int, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self._connections:
            self._connections[session_id] = set()
        self._connections[session_id].add(websocket)

    def disconnect(self, session_id: int, websocket: WebSocket):
        if session_id in self._connections:
            self._connections[session_id].discard(websocket)
            if not self._connections[session_id]:
                del self._connections[session_id]

    async def broadcast(self, session_id: int, message: dict):
        if session_id not in self._connections:
            return
        dead = []
        for ws in self._connections[session_id]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections[session_id].discard(ws)


ws_manager = ConnectionManager()


@router.websocket("/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: int):
    """Stream tracking session status updates over WebSocket.

    Client should send initial auth message:
        {"type": "auth", "token": "..."}
    """
    await ws_manager.connect(session_id, websocket)

    try:
        # Wait for auth message
        auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        if auth_msg.get("type") != "auth" or not auth_msg.get("token"):
            await websocket.send_json({"type": "error", "data": "Auth required"})
            await websocket.close()
            return

        payload = decode_access_token(auth_msg["token"])
        if payload is None:
            await websocket.send_json({"type": "error", "data": "Invalid token"})
            await websocket.close()
            return

        await websocket.send_json({"type": "auth_ok", "data": {"user_id": payload.get("sub")}})

        mgr = get_session_manager()

        # Polling loop — send status updates
        while True:
            status = mgr.get_status(session_id)
            if status:
                await websocket.send_json({
                    "type": "stats",
                    "data": status,
                })
            else:
                # Session not running — check DB
                from sqlmodel import Session as DBSession
                with DBSession(engine) as db:
                    from models.tracking_session import TrackingSession
                    ts = db.get(TrackingSession, session_id)
                    if ts:
                        await websocket.send_json({
                            "type": "session_status",
                            "data": {"status": ts.status, "session_id": session_id},
                        })

            # Check for incoming messages (e.g., "stop" command)
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=2.0)
                if msg.get("type") == "stop":
                    mgr.stop_session(session_id)
                    await websocket.send_json({"type": "session_status", "data": {"status": "stopping"}})
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %d", session_id)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        ws_manager.disconnect(session_id, websocket)
