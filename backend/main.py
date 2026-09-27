"""
backend/main.py

FastAPI application entry point for JanSahayak.

Routes:
  GET  /         → serve frontend/index.html (via StaticFiles mount)
  GET  /healthz  → liveness probe — never calls Sarvam API (spec §111)
  GET  /readyz   → readiness probe — checks key presence + KB loaded (spec §112)
  WS   /ws       → per-session WebSocket handler

Concurrency model:
  One Uvicorn worker, pure asyncio.
  Each WebSocket connection gets its own WebSocketHandler instance with
  independent session state — no shared mutable state between sessions.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.websocket.handler import WebSocketHandler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """
    Startup:
      - Verify SARVAM_API_KEY is present (fail fast).
      - Log configuration summary.
      - Phase 3+: open shared Sarvam SDK client here.
      - Phase 7+: load knowledge base into memory here.

    Shutdown:
      - Phase 3+: close SDK connections.
    """
    # Fail fast — every other component depends on this key.
    if not settings.sarvam_api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    logger.info("JanSahayak starting up.")
    logger.info("  env        = %s", settings.app_env)
    logger.info("  STT model  = %s", settings.sarvam_stt_model)
    logger.info("  LLM model  = %s", settings.sarvam_llm_model)
    logger.info("  TTS model  = %s", settings.sarvam_tts_model)
    logger.info("  default lang = %s", settings.default_response_language)

    # [PHASE-7-KB] knowledge.retriever.load() goes here

    yield

    logger.info("JanSahayak shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="JanSahayak",
    description="Multilingual government citizen-service voice agent",
    version="0.1.0",
    lifespan=lifespan,
    # Disable auto-generated docs in production to reduce attack surface.
    docs_url="/docs" if os.getenv("APP_ENV", "development") == "development" else None,
    redoc_url=None,
)


# ---------------------------------------------------------------------------
# Static file serving — frontend at /
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
    logger.info("Serving frontend from %s", _FRONTEND_DIR)
else:
    logger.warning(
        "frontend/ directory not found at %s — static files will not be served.",
        _FRONTEND_DIR,
    )


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """Serve the frontend index page."""
    from fastapi.responses import FileResponse
    index = _FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "JanSahayak backend running. Frontend not found."})


@app.get("/healthz", include_in_schema=False)
async def healthz() -> JSONResponse:
    """
    Liveness probe.

    IMPORTANT: must not call any Sarvam API (spec §111).
    Only checks that the process is alive and config is loaded.
    """
    return JSONResponse({"status": "ok"})


@app.get("/readyz", include_in_schema=False)
async def readyz() -> JSONResponse:
    """
    Readiness probe (spec §112).

    Checks:
      - SARVAM_API_KEY is present in environment.
      - [Phase 7+] knowledge base is loaded.
      - [Phase 12+] callback DB is initialised.
    """
    checks: dict[str, bool] = {
        "api_key_present": bool(settings.sarvam_api_key),
        # [PHASE-7-KB] "knowledge_base_loaded": retriever.is_loaded(),
        # [PHASE-12-CB] "callback_db_ready": callback_store.is_ready(),
    }
    all_ok = all(checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse({"status": "ready" if all_ok else "not_ready", "checks": checks}, status_code=status_code)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Accept a WebSocket connection and hand it to a dedicated handler.

    Each call creates a fresh WebSocketHandler — sessions are fully isolated.
    No shared mutable state exists between concurrent connections.
    """
    await websocket.accept()

    handler = WebSocketHandler(websocket)
    logger.info("WebSocket connection accepted — session_id=%s", handler.session_id)

    await handler.run()
