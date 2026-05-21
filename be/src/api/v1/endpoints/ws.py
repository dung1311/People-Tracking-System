import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.session_manager import get_session_manager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/network/{network_id}")
async def websocket_session_stream(websocket: WebSocket, network_id: int):
    await websocket.accept()
    logger.info(f"WebSocket client connected to stream Session {network_id}")
    
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=100) # Buffer up to 100 frames to prevent memory leaks

    # Define the thread-safe callback
    def on_session_event(event_data: dict):
        # Schedule putting the event into the async queue from the pipeline thread
        loop.call_soon_threadsafe(queue.put_nowait, event_data)

    # Register the callback with the SessionManager
    mgr = get_session_manager()
    mgr.register_ws_callback(network_id, on_session_event)

    try:
        # Loop to consume events from the queue and send them down the WebSocket
        while True:
            try:
                # Read from queue with a timeout so we can keep the websocket alive/heartbeat
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_json(event)
                queue.task_done()
            except asyncio.TimeoutError:
                # Send a tiny ping to keep connection alive if no frames are arriving
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from Session {network_id}")
    except Exception as e:
        logger.error(f"WebSocket error in Session {network_id}: {e}")
    finally:
        # Deregister callback to prevent memory leaks
        mgr.deregister_ws_callback(network_id, on_session_event)
        try:
            await websocket.close()
        except Exception:
            pass
