
"""FastAPI app. Run from the backend folder: uvicorn app.main:app --reload"""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, service
from .model import get_bundle

log = logging.getLogger("intellimaintain")


async def _ticker():
    while True:
        await asyncio.sleep(config.SIM_TICK_SECONDS)
        try:
            await asyncio.to_thread(service.tick)
        except Exception:
            log.exception("simulation tick failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_bundle()
    service.init()
    task = (
        asyncio.create_task(_ticker())
        if config.SIM_TICK_SECONDS > 0
        else None
    )
    yield
    if task:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="IntelliMaintain AI",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://intellimaintain-frontend.onrender.com",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReadingIn(BaseModel):
    temperature_c: Optional[float] = None
    vibration_mm_s: Optional[float] = None
    rpm: Optional[float] = None
    operating_hours: Optional[float] = None


class DecisionIn(BaseModel):
    decision: Literal["approve", "reject"]
    note: Optional[str] = Field(default=None, max_length=500)


@app.get("/api/health")
def health():
    return {"ok": True, "data_source": config.DATA_LABEL}


@app.get("/api/fleet")
def get_fleet():
    return service.fleet()


@app.get("/api/machines/{machine_id}")
def get_machine(
    machine_id: str,
    n: int = Query(120, ge=20, le=500),
):
    try:
        return service.machine_detail(machine_id, n)
    except KeyError:
        raise HTTPException(404, f"Unknown machine {machine_id}")


@app.post("/api/machines/{machine_id}/readings")
def post_reading(machine_id: str, body: ReadingIn):
    """Submit one reading yourself. Omitted fields are stored as missing."""
    try:
        return service.ingest(machine_id, body.model_dump())
    except KeyError:
        raise HTTPException(404, f"Unknown machine {machine_id}")


@app.get("/api/alerts")
def get_alerts(
    status: Optional[
        Literal["pending", "approved", "rejected", "superseded"]
    ] = None,
    limit: int = Query(100, ge=1, le=500),
):
    return service.list_alerts(status, limit)


@app.post("/api/alerts/{alert_id}/decision")
def post_decision(alert_id: int, body: DecisionIn):
    try:
        return service.decide(alert_id, body.decision, body.note)
    except KeyError:
        raise HTTPException(404, f"Unknown alert {alert_id}")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/api/model")
def get_model():
    return get_bundle().meta


@app.post("/api/simulate/tick")
def simulate_tick():
    service.tick()
    return {"ok": True}


@app.post("/api/simulate/reset")
def simulate_reset():
    service.reset()
    return {"ok": True}


# Serve the built React app too, if present.
_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
