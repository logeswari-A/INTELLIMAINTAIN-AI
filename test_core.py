"""Core logic tests: data validation, the agent's decisions, and the alert workflow.

These use the simulated fleet. They check behaviour (what the system says and refuses to say),
not real-world accuracy.
"""
import pytest

from app import agent, config, service, simulator
from app.model import get_bundle
from app.validation import usable_rows, validate


def good_rows(n=30, **overrides):
    rows = []
    for i in range(n):
        r = {"temperature_c": 65 + (i % 3) * 0.4, "vibration_mm_s": 2.5 + (i % 4) * 0.1,
             "rpm": 1480 + (i % 5) * 3, "operating_hours": 3000 + i * 0.02}
        r.update({k: v for k, v in overrides.items()})
        rows.append(r)
    return rows


MACHINE = {"id": "T-1", "name": "Test", "type": "Test"}


# ------------------------------------------------------------ validation

def test_clean_data_is_reliable():
    dq = validate(good_rows())
    assert dq["status"] == "reliable" and dq["issues"] == [] and dq["score"] == 100


def test_empty_history_is_unreliable():
    assert validate([])["status"] == "unreliable"


def test_sensor_offline_is_unreliable():
    rows = good_rows()
    for r in rows[-4:]:
        r["rpm"] = None
    dq = validate(rows)
    assert dq["status"] == "unreliable"
    assert any(i["code"] == "offline" and i["sensors"] == ["rpm"] for i in dq["issues"])
    assert dq["sensor_status"]["rpm"] == "bad"


def test_single_dropout_is_only_suspect():
    rows = good_rows()
    rows[-1]["vibration_mm_s"] = None
    dq = validate(rows)
    assert dq["status"] == "suspect"
    assert any(i["code"] == "dropout" for i in dq["issues"])


def test_out_of_range_is_unreliable():
    rows = good_rows()
    rows[-1]["temperature_c"] = 900
    assert validate(rows)["status"] == "unreliable"


def test_flatline_detected():
    rows = good_rows()
    for r in rows[-8:]:
        r["temperature_c"] = 71.0
    assert any(i["code"] == "flatline" for i in validate(rows)["issues"])


def test_spike_detected():
    rows = good_rows()
    rows[-1]["vibration_mm_s"] = 30.0
    assert any(i["code"] == "spike" for i in validate(rows)["issues"])


def test_gradual_rise_is_not_a_spike():
    rows = good_rows()
    for i, r in enumerate(rows):
        r["vibration_mm_s"] = 2.0 + i * 0.05
    assert not any(i["code"] == "spike" for i in validate(rows)["issues"])


def test_cross_sensor_inconsistency():
    rows = good_rows()
    rows[-1]["rpm"] = 2.0
    codes = [i["code"] for i in validate(rows)["issues"]]
    assert "inconsistent" in codes


def test_counter_decrease():
    rows = good_rows()
    rows[-1]["operating_hours"] = 10
    assert any(i["code"] == "counter_decrease" for i in validate(rows)["issues"])


def test_usable_rows_skip_incomplete_and_out_of_range():
    rows = good_rows()
    rows[-1]["rpm"] = None
    rows[-2]["temperature_c"] = 999
    assert all(r["rpm"] is not None and r["temperature_c"] < 200 for r in usable_rows(rows))


# ------------------------------------------------------------ agent

@pytest.fixture(scope="module")
def bundle():
    return get_bundle()


def test_agent_refuses_to_predict_on_unreliable_data(bundle):
    rows = good_rows()
    for r in rows[-12:]:
        r["rpm"] = None
    a = agent.assess(MACHINE, rows, bundle)
    assert a["risk_score"] is None and a["risk_level"] is None
    assert a["status"] == "data_check" and a["needs_human_inspection"] and not a["evidence_sufficient"]
    assert "manual inspection" in a["recommended_action"]


def test_agent_healthy_machine(bundle):
    a = agent.assess(MACHINE, good_rows(), bundle)
    assert a["status"] == "healthy" and a["risk_level"] == "Low"


def test_agent_high_risk_with_reliable_data_recommends_inspection(bundle):
    rows = good_rows(temperature_c=86.0, vibration_mm_s=8.0, rpm=1340.0)
    for i, r in enumerate(rows):  # add tiny variation so no flatline is flagged
        r["temperature_c"] += (i % 3) * 0.3
        r["vibration_mm_s"] += (i % 4) * 0.05
        r["rpm"] += (i % 5) * 2
    a = agent.assess(MACHINE, rows, bundle)
    assert a["status"] == "high_risk" and a["risk_level"] == "High"
    assert "technician should confirm" in a["recommended_action"]
    assert any("Vibration" in line for line in a["explanation"])


def test_agent_high_vibration_with_dirty_data_is_not_a_confident_alarm(bundle):
    """The brief's key case: high vibration + inconsistent data => explain uncertainty."""
    rows = good_rows(temperature_c=86.0, vibration_mm_s=8.0, rpm=1340.0)
    for i, r in enumerate(rows):
        r["temperature_c"] += (i % 3) * 0.3
        r["vibration_mm_s"] += (i % 4) * 0.05
        r["rpm"] += (i % 5) * 2
    for i in range(0, 30, 3):  # sensor drops out intermittently
        rows[i]["vibration_mm_s"] = None
    rows[-1]["vibration_mm_s"] = 8.2
    a = agent.assess(MACHINE, rows, bundle)
    assert a["status"] == "data_check"          # never "high_risk"
    assert a["needs_human_inspection"]
    assert "reference only" in " ".join(a["explanation"])


def test_agent_never_calls_a_prediction_certain(bundle):
    a = agent.assess(MACHINE, good_rows(), bundle)
    assert "not a guaranteed failure" in a["caveat"]


# ------------------------------------------------------------ simulated fleet end to end

def test_fleet_covers_every_scenario(tmp_db):
    service.init()
    f = service.fleet()
    assert f["total"] == 12 and f["data_source"] == "SIMULATED"
    statuses = {m["id"]: m for m in f["machines"]}
    assert statuses["M-03"]["status"] == "high_risk"
    assert statuses["M-11"]["risk_score"] is None                     # rpm offline: no prediction
    assert statuses["M-09"]["data_quality"]["status"] == "suspect"    # stuck temperature
    assert statuses["M-07"]["status"] == "data_check"                 # flaky vibration, elevated estimate
    assert set(f["counts"]) == {"healthy", "warning", "high_risk", "data_check"}
    assert sum(f["counts"].values()) == 12


def test_fleet_is_sorted_by_priority(tmp_db):
    service.init()
    pr = [m["priority"] for m in service.fleet()["machines"]]
    assert pr == sorted(pr)


def test_simulation_is_repeatable():
    m = simulator.FLEET[3]
    assert simulator.generate_reading(3, m, 17) == simulator.generate_reading(3, m, 17)


def test_alert_approval_workflow(tmp_db):
    service.init()
    pending = service.list_alerts(status="pending")
    assert pending
    a = service.decide(pending[0]["id"], "approve", "Looked at it")
    assert a["decision"] == "approved" and a["operator_note"] == "Looked at it"
    with pytest.raises(ValueError):
        service.decide(pending[0]["id"], "reject", None)       # already decided
    with pytest.raises(KeyError):
        service.decide(99999, "approve", None)


def test_flickering_state_does_not_flood_alerts(tmp_db):
    service.init()
    before = len(service.list_alerts(limit=500))
    for _ in range(60):
        service.tick()
    created = len(service.list_alerts(limit=500)) - before
    assert created <= 8, f"too many new alerts in 60 ticks: {created}"


def test_manual_reading_with_missing_fields_is_stored_and_flagged(tmp_db):
    service.init()
    service.ingest("M-01", {"temperature_c": 70.0, "vibration_mm_s": None, "rpm": 1500, "operating_hours": 1300})
    detail = service.machine_detail("M-01")
    assert detail["readings"][-1]["source"] == "manual"
    assert detail["readings"][-1]["vibration_mm_s"] is None
    with pytest.raises(KeyError):
        service.ingest("NOPE", {})


def test_reset_restores_seed_state(tmp_db):
    service.init()
    for _ in range(5):
        service.tick()
    service.reset()
    with service.connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM readings WHERE machine_id='M-01'").fetchone()[0]
    assert n == config.SEED_HISTORY
