"""HTTP-level tests. Skipped automatically if fastapi/httpx are not installed."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402


@pytest.fixture()
def client(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "SIM_TICK_SECONDS", 0)  # no background ticker during tests
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health_and_fleet(client):
    assert client.get("/api/health").json()["data_source"] == "SIMULATED"
    fleet = client.get("/api/fleet").json()
    assert fleet["total"] == 12 and fleet["data_source"] == "SIMULATED"


def test_machine_detail_and_404(client):
    d = client.get("/api/machines/M-03").json()
    assert d["assessment"]["status"] == "high_risk" and len(d["readings"]) > 20
    assert client.get("/api/machines/NOPE").status_code == 404


def test_decision_flow(client):
    alerts = client.get("/api/alerts", params={"status": "pending"}).json()
    assert alerts
    r = client.post(f"/api/alerts/{alerts[0]['id']}/decision", json={"decision": "approve", "note": "ok"})
    assert r.status_code == 200 and r.json()["decision"] == "approved"
    again = client.post(f"/api/alerts/{alerts[0]['id']}/decision", json={"decision": "reject"})
    assert again.status_code == 409
    assert client.post("/api/alerts/1/decision", json={"decision": "explode"}).status_code == 422


def test_post_reading_and_tick(client):
    assert client.post("/api/machines/M-01/readings", json={"temperature_c": 70}).status_code == 200
    assert client.post("/api/machines/NOPE/readings", json={}).status_code == 404
    assert client.post("/api/simulate/tick").json()["ok"] is True


def test_model_endpoint_labels_data_as_simulated(client):
    meta = client.get("/api/model").json()
    assert "simulated" in meta["data_provenance"].lower()
