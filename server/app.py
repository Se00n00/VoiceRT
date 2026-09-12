"""FastAPI application factory for the voice pipeline.

Entry file: inserts the repo root on sys.path so `server.*` / `engine.*`
absolute imports work without an install, builds the app, includes the
HTTP router + talk WebSocket, and serves on port 8003 under __main__.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_app():
    from fastapi import FastAPI

    from server.middleware import setup_middleware
    from server.routes import router
    from server.websocket import router as ws_router

    app = FastAPI(title="voice-pipeline")
    setup_middleware(app)
    app.include_router(router)
    app.include_router(ws_router)

    @app.on_event("startup")
    def _warm():
        # Warm the engine if weights are present; never fail startup
        # when running import-only checks / tests.
        try:
            from server.routes import get_engine

            get_engine()
            print("voice-pipeline ready", flush=True)
        except Exception as exc:
            print(f"voice-pipeline engine not warmed: {exc}", flush=True)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
