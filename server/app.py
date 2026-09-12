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
        # when running import-only checks / tests. On success, probe VRAM
        # and derive the serving plan (sessions from generation length).
        try:
            from server.routes import configure_capacity, get_engine

            eng = get_engine()
            print("voice-pipeline ready", flush=True)
            try:
                import os

                import yaml

                from runtime.capacity import plan_capacity, probe_vram
                from runtime.device import max_allocated_mb

                root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                with open(os.path.join(root, "configs", "pipeline.yaml")) as f:
                    pcfg = yaml.safe_load(f) or {}
                cap = pcfg.get("capacity", {}) or {}
                streaming = pcfg.get("streaming", {}) or {}
                info = probe_vram()
                plan = plan_capacity(
                    info["total_mb"], max_allocated_mb(),
                    max_new_tokens=int(streaming.get("max_tokens", 48)),
                    headroom_frac=float(cap.get("headroom_frac", 0.10)),
                    headroom_min_mb=float(cap.get("headroom_min_mb", 400)),
                    sessions_cap=int(cap.get("sessions_cap", 1000)),
                    max_inflight=int(cap.get("max_inflight", 4)),
                    queue_timeout_s=float(cap.get("queue_timeout_s", 10)),
                    per_turn_mb=float(cap.get("per_turn_mb", 150)),
                    safety=float(cap.get("safety_factor", 1.25)),
                )
                plan["gpu"] = info.get("name", "cpu")
                configure_capacity(plan)
                print("capacity: %(max_sessions)d sessions @ genlen "
                      "%(generation_length)d, %(max_inflight)d inflight "
                      "(%(gpu)s %(vram_total_mb).0fMB, baseline "
                      "%(baseline_mb).0fMB)" % plan, flush=True)
            except Exception as exc:
                print(f"capacity planning skipped ({exc})", flush=True)
        except Exception as exc:
            print(f"voice-pipeline engine not warmed: {exc}", flush=True)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
