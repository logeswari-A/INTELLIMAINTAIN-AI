"""Persistence and orchestration: SQLite storage, simulation ticks, alerts, approvals.

No FastAPI imports here, so this layer is easy to test. The recommendation workflow only
records the operator's decision. Nothing in this project controls real equipment.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from . import agent, config, simulator
from .model import get_bundle

SAMPLE_SECONDS = 5
HYSTERESIS = 0.05      # a risk level only changes if the score clears the threshold by this margin
CONFIRM_SAMPLES = 3   # a state change must hold this many evaluations in a row before it raises or clears an alert

SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, idx INTEGER NOT NULL,
    last_signature TEXT NOT NULL DEFAULT 'healthy',
    cand_signature TEXT NOT NULL DEFAULT '',
    cand_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, machine_id TEXT NOT NULL, ts TEXT NOT NULL,
    temperature_c REAL, vibration_mm_s REAL, rpm REAL, operating_hours REAL,
    source TEXT NOT NULL DEFAULT 'simulated'
);
CREATE INDEX IF NOT EXISTS idx_readings_machine ON readings (machine_id, id);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, machine_id TEXT NOT NULL, created_at TEXT NOT NULL,
    status_kind TEXT NOT NULL, risk_level TEXT, risk_score REAL, dq_status TEXT NOT NULL,
    headline TEXT NOT NULL, explanation TEXT NOT NULL, recommended_action TEXT NOT NULL,
    signature TEXT NOT NULL, decision TEXT NOT NULL DEFAULT 'pending',
    decided_at TEXT, operator_note TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_machine ON alerts (machine_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------- setup

def init(seed_if_empty: bool = True) -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        empty = conn.execute("SELECT COUNT(*) FROM machines").fetchone()[0] == 0
        if empty and seed_if_empty:
            _seed(conn)


def _seed(conn) -> None:
    for idx, m in enumerate(simulator.FLEET):
        conn.execute("INSERT INTO machines (id, name, type, idx) VALUES (?,?,?,?)",
                     (m["id"], m["name"], m["type"], idx))
    start = datetime.now(timezone.utc) - timedelta(seconds=SAMPLE_SECONDS * config.SEED_HISTORY)
    for idx, m in enumerate(simulator.FLEET):
        for step in range(config.SEED_HISTORY):
            ts = (start + timedelta(seconds=SAMPLE_SECONDS * step)).isoformat(timespec="seconds")
            _insert_reading(conn, m["id"], ts, simulator.generate_reading(idx, m, step), "simulated")
    _evaluate(conn, confirm_after=1)


def reset() -> None:
    """Wipe all stored data and re-seed the simulated fleet."""
    with connect() as conn:
        conn.executescript("DROP TABLE IF EXISTS alerts; DROP TABLE IF EXISTS readings; DROP TABLE IF EXISTS machines;")
    init()


def _insert_reading(conn, machine_id, ts, values, source):
    conn.execute(
        "INSERT INTO readings (machine_id, ts, temperature_c, vibration_mm_s, rpm, operating_hours, source) "
        "VALUES (?,?,?,?,?,?,?)",
        (machine_id, ts, values.get("temperature_c"), values.get("vibration_mm_s"),
         values.get("rpm"), values.get("operating_hours"), source),
    )


# ---------------------------------------------------------------- reading rows

def _recent_rows(conn, machine_id: str, n: int) -> list:
    cur = conn.execute(
        "SELECT ts, temperature_c, vibration_mm_s, rpm, operating_hours, source FROM readings "
        "WHERE machine_id=? ORDER BY id DESC LIMIT ?", (machine_id, n))
    return [dict(r) for r in reversed(cur.fetchall())]


def _machine_rows(conn) -> list:
    return [dict(r) for r in conn.execute("SELECT * FROM machines ORDER BY idx")]


# ---------------------------------------------------------------- assessment + alerts

def _signature(a: dict) -> str:
    if a["status"] == "healthy":
        return "healthy"
    return f"{a['status']}|{a['data_quality']['status']}|{a['risk_level']}"


def _stable_signature(a: dict, last_sig: str) -> str:
    """Keep the previous signature while the score only wobbles near a risk threshold."""
    sig = _signature(a)
    if sig == last_sig or a["risk_score"] is None:
        return sig
    last = ("healthy", "reliable", "Low") if last_sig == "healthy" else tuple(last_sig.split("|"))
    new = ("healthy", "reliable", "Low") if sig == "healthy" else tuple(sig.split("|"))
    if last[1] == new[1] and last[2] != new[2] and last[2] != "None" and new[2] != "None":
        p = a["risk_score"]
        if min(abs(p - config.RISK_MEDIUM), abs(p - config.RISK_HIGH)) < HYSTERESIS:
            return last_sig
    return sig


def _assess_all(conn) -> list:
    bundle = get_bundle()
    out = []
    for m in _machine_rows(conn):
        rows = _recent_rows(conn, m["id"], config.ASSESSMENT_WINDOW)
        a = agent.assess(m, rows, bundle)
        out.append((m, rows, a))
    return out


def _evaluate(conn, assessed: Optional[list] = None, confirm_after: int = CONFIRM_SAMPLES) -> None:
    """Create or retire alerts when a machine's state changes.

    Debounced: the new state must be seen `confirm_after` evaluations in a row, so a flickering
    sensor or a risk score hovering at a threshold does not flood the operator with alerts.
    """
    for m, _rows, a in (assessed if assessed is not None else _assess_all(conn)):
        sig = _stable_signature(a, m["last_signature"])
        if sig == m["last_signature"]:
            if m["cand_count"]:
                conn.execute("UPDATE machines SET cand_signature='', cand_count=0 WHERE id=?", (m["id"],))
            continue
        count = m["cand_count"] + 1 if sig == m["cand_signature"] else 1
        if count < confirm_after:
            conn.execute("UPDATE machines SET cand_signature=?, cand_count=? WHERE id=?", (sig, count, m["id"]))
            continue

        # Confirmed state change.
        note = "Condition cleared automatically" if sig == "healthy" else "Replaced by a newer assessment"
        conn.execute("UPDATE alerts SET decision='superseded', decided_at=?, operator_note=? "
                     "WHERE machine_id=? AND decision='pending'", (_now(), note, m["id"]))
        if sig != "healthy":
            conn.execute(
                "INSERT INTO alerts (machine_id, created_at, status_kind, risk_level, risk_score, dq_status, "
                "headline, explanation, recommended_action, signature) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (m["id"], _now(), a["status"], a["risk_level"], a["risk_score"], a["data_quality"]["status"],
                 a["headline"], json.dumps(a["explanation"]), a["recommended_action"], sig))
        conn.execute("UPDATE machines SET last_signature=?, cand_signature='', cand_count=0 WHERE id=?",
                     (sig, m["id"]))


def tick() -> None:
    """Advance the simulation by one reading per machine, then re-evaluate."""
    with connect() as conn:
        for m in _machine_rows(conn):
            step = conn.execute(
                "SELECT COUNT(*) FROM readings WHERE machine_id=? AND source='simulated'", (m["id"],)
            ).fetchone()[0]
            _insert_reading(conn, m["id"], _now(), simulator.generate_reading(m["idx"], simulator.FLEET[m["idx"]], step), "simulated")
        _evaluate(conn)


def ingest(machine_id: str, values: dict) -> dict:
    """Accept one reading from outside (manual / API). Missing fields are stored as missing."""
    with connect() as conn:
        if not conn.execute("SELECT 1 FROM machines WHERE id=?", (machine_id,)).fetchone():
            raise KeyError(machine_id)
        _insert_reading(conn, machine_id, _now(), values, "manual")
        _evaluate(conn, confirm_after=1)
    return {"machine_id": machine_id, "stored": True, "source": "manual"}


# ---------------------------------------------------------------- queries for the API

def _latest(rows: list) -> Optional[dict]:
    return rows[-1] if rows else None


def _spark(rows: list, field: str = "vibration_mm_s", n: int = 40) -> list:
    return [r[field] for r in rows[-n:]]


def fleet() -> dict:
    with connect() as conn:
        assessed = _assess_all(conn)
        pending = {r["machine_id"]: r["c"] for r in conn.execute(
            "SELECT machine_id, COUNT(*) c FROM alerts WHERE decision='pending' GROUP BY machine_id")}
    machines = []
    for m, rows, a in assessed:
        machines.append({
            "id": m["id"], "name": m["name"], "type": m["type"],
            "latest": _latest(rows), "vibration_trend": _spark(rows),
            "pending_alerts": pending.get(m["id"], 0),
            **a,
        })
    machines.sort(key=lambda x: (x["priority"], -(x["risk_score"] or 0)))
    counts = {"healthy": 0, "warning": 0, "high_risk": 0, "data_check": 0}
    for x in machines:
        counts[x["status"]] += 1
    return {"generated_at": _now(), "data_source": config.DATA_LABEL,
            "total": len(machines), "counts": counts, "machines": machines}


def machine_detail(machine_id: str, n: int = 120) -> dict:
    with connect() as conn:
        m = conn.execute("SELECT * FROM machines WHERE id=?", (machine_id,)).fetchone()
        if not m:
            raise KeyError(machine_id)
        m = dict(m)
        rows = _recent_rows(conn, machine_id, max(n, config.ASSESSMENT_WINDOW))
        a = agent.assess(m, rows[-config.ASSESSMENT_WINDOW:], get_bundle())
        alerts = _alerts(conn, machine_id=machine_id, limit=50)
    return {"id": m["id"], "name": m["name"], "type": m["type"], "data_source": config.DATA_LABEL,
            "assessment": a, "readings": rows[-n:], "alerts": alerts}


def _alerts(conn, status: Optional[str] = None, machine_id: Optional[str] = None, limit: int = 100) -> list:
    sql = ("SELECT a.*, m.name AS machine_name FROM alerts a JOIN machines m ON m.id=a.machine_id WHERE 1=1")
    args = []
    if status:
        sql += " AND a.decision=?"; args.append(status)
    if machine_id:
        sql += " AND a.machine_id=?"; args.append(machine_id)
    sql += " ORDER BY a.id DESC LIMIT ?"; args.append(limit)
    out = []
    for r in conn.execute(sql, args):
        d = dict(r)
        d["explanation"] = json.loads(d["explanation"])
        out.append(d)
    return out


def list_alerts(status: Optional[str] = None, limit: int = 100) -> list:
    with connect() as conn:
        return _alerts(conn, status=status, limit=limit)


def decide(alert_id: int, decision: str, note: Optional[str]) -> dict:
    """Record the operator's approve/reject. This does not trigger any machine action."""
    if decision not in ("approve", "reject"):
        raise ValueError("decision must be 'approve' or 'reject'")
    with connect() as conn:
        row = conn.execute("SELECT decision FROM alerts WHERE id=?", (alert_id,)).fetchone()
        if not row:
            raise KeyError(alert_id)
        if row["decision"] != "pending":
            raise ValueError(f"Alert already {row['decision']}")
        conn.execute("UPDATE alerts SET decision=?, decided_at=?, operator_note=? WHERE id=?",
                     ("approved" if decision == "approve" else "rejected", _now(), note, alert_id))
        return next(a for a in _alerts(conn, limit=1000) if a["id"] == alert_id)
