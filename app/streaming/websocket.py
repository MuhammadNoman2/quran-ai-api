"""The WebSocket endpoint.

    WS /api/v1/sessions/{session_id}/stream

Create the session over REST first (`POST /api/v1/sessions`), then connect. That
split is deliberate: capacity can be refused with a normal HTTP status and a
readable error, rather than by accepting a socket and immediately closing it.

Client -> server:
    {"type": "start", "surah": 1, "ayah": 2}   once, before any audio
    <binary frames>                             pcm_s16le at the session rate
    {"type": "flush"}                           analyse now
    {"type": "stop"}                            finish and score

Server -> client: the events in app/streaming/events.py.

No single bad frame ends a session (brief 40). Malformed JSON, odd-length PCM, a
failed recognition pass - each produces an `error` event and the session
continues. Only a protocol-level impossibility closes the socket, and it says so
with `fatal: true`.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from app.api.deps import get_analyzer, get_auth, get_session_store, get_settings
from app.core.config import Settings
from app.core.errors import APIError
from app.core.security import AuthProvider
from app.streaming.session import SessionStore
from app.streaming import events
from app.streaming.events import ErrorCode
from app.streaming.manager import StreamProcessor

logger = logging.getLogger(__name__)
router = APIRouter(tags=["sessions"])


@router.websocket("/sessions/{session_id}/stream")
async def stream(
    websocket: WebSocket,
    session_id: str,
    api_key: str | None = Query(
        default=None,
        description="Browsers cannot set headers on a WebSocket, so the key may go here",
    ),
    config: Settings = Depends(get_settings),
    store: SessionStore = Depends(get_session_store),
    auth: AuthProvider = Depends(get_auth),
) -> None:
    """Dependencies are injected rather than fetched.

    Calling get_auth() directly here would work in production but sits outside
    FastAPI's dependency graph, so the authentication path could not be
    exercised by tests via dependency_overrides - and a security check nothing
    can test is a liability regardless of whether it currently works.
    """
    # Authenticate before accepting, so an unauthorised client never gets a socket.
    try:
        auth.authenticate(api_key or websocket.headers.get("x-api-key"))
    except APIError:
        await websocket.close(code=4401, reason="unauthorized")
        return

    session = store.get(session_id)
    if session is None:
        await websocket.close(code=4404, reason="unknown or expired session")
        return

    await websocket.accept()
    processor = StreamProcessor(
        session=session,
        analyzer=get_analyzer(session.tier),
        config=config,
        send=websocket.send_json,
    )
    logger.info(
        "stream opened",
        extra={"session_id": session_id, "surah": session.surah, "ayah": session.ayah},
    )

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if (payload := message.get("bytes")) is not None:
                await processor.on_audio(payload)
                continue

            text = message.get("text")
            if text is None:
                continue

            try:
                control = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_json(
                    events.error(ErrorCode.INVALID_MESSAGE, "message is not valid JSON")
                )
                continue

            if not isinstance(control, dict):
                await websocket.send_json(
                    events.error(ErrorCode.INVALID_MESSAGE, "expected a JSON object")
                )
                continue

            kind = control.get("type")
            if kind == "start":
                await processor.start(control)
            elif kind == "flush":
                await processor.flush()
            elif kind == "stop":
                await processor.stop()
                break
            else:
                await websocket.send_json(
                    events.error(
                        ErrorCode.INVALID_MESSAGE,
                        f"unknown message type {kind!r}; expected start, flush or stop",
                    )
                )
    except WebSocketDisconnect:
        logger.info("stream disconnected", extra={"session_id": session_id})
    except Exception:
        logger.exception("stream failed", extra={"session_id": session_id})
        try:
            await websocket.send_json(
                events.error(ErrorCode.INTERNAL_ERROR, "stream failed", fatal=True)
            )
        except Exception:  # socket already gone
            pass
    finally:
        store.close(session_id)
        logger.info(
            "stream closed",
            extra={
                "session_id": session_id,
                "segments": processor.state.segments_analysed,
                "words_confirmed": len(processor.state.confirmed_words),
                "mistakes": len(processor.state.reported_mistakes),
            },
        )
