"""Maintenance agent: turns readings + data quality + model output into a grounded recommendation.

This agent is rule-based and deterministic. Every number in its text comes from the readings,
the data-quality checks or the model output, so it cannot invent measurements. It is built so
an LLM could be added later to phrase the text, but the decision logic stays here.
"""
import numpy as np

from . import config
from .model import ModelBundle, predict_probability, risk_level
from .validation import usable_rows, validate

CAVEAT = (
    "This is an estimated risk from a model trained on simulated data. "
    "It is not a confirmed fault and not a guaranteed failure."
)

TYPICAL_CHECKS = {
    "vibration_mm_s": "bearings, shaft alignment, balance and mounting bolts",
    "temperature_c": "lubrication, cooling airflow and load",
    "rpm": "drive belt or coupling, load and motor supply",
    "operating_hours": "whether the service interval is due",
}


def _drivers(values: dict, baseline: dict) -> list:
    out = []
    for f in config.FEATURES:
        mu, sd = baseline[f]["mean"], max(baseline[f]["std"], 1e-6)
        z = (values[f] - mu) / sd
        if (f == "rpm" and z <= -1.5) or (f != "rpm" and z >= 1.5):
            out.append((f, z, mu))
    return sorted(out, key=lambda t: abs(t[1]), reverse=True)


def _driver_text(f: str, value: float, mu: float) -> str:
    label, unit = config.FIELD_LABELS[f], config.FIELD_UNITS[f]
    direction = "below" if f == "rpm" else "above"
    return f"{label} is {value:.1f}{unit}, {direction} the healthy baseline of about {mu:.1f}{unit}."


def _trend_notes(rows: list, sensor_status: dict) -> list:
    notes = []
    for f, min_delta, unit in (("vibration_mm_s", 0.5, "mm/s"), ("temperature_c", 3.0, "°C")):
        if sensor_status.get(f) != "ok":
            continue
        vals = [r.get(f) for r in rows[-40:] if r.get(f) is not None]
        if len(vals) < 20:
            continue
        third = len(vals) // 3
        delta = float(np.mean(vals[-third:]) - np.mean(vals[:third]))
        if delta >= min_delta:
            notes.append(f"{config.FIELD_LABELS[f]} has risen by about {delta:.1f} {unit} over the last {len(vals)} readings.")
    return notes


def _sensor_names(issues: list) -> str:
    names = []
    for i in issues:
        for s in i["sensors"]:
            lab = config.FIELD_LABELS[s].lower()
            if lab not in names:
                names.append(lab)
    return ", ".join(names) if names else "sensors"


def assess(machine: dict, rows: list, bundle: ModelBundle) -> dict:
    dq = validate(rows)
    usable = usable_rows(rows)
    latest = rows[-1] if rows else {}

    result = {
        "machine_id": machine["id"],
        "data_quality": dq,
        "risk_score": None,
        "risk_level": None,
        "status": "data_check",
        "priority": 1,
        "evidence_sufficient": False,
        "needs_human_inspection": True,
        "headline": "",
        "explanation": [],
        "recommended_action": "",
        "caveat": CAVEAT,
    }

    # 1. Data too poor or too thin to predict: say so and do not predict.
    if dq["status"] == "unreliable" or len(usable) < 3:
        problems = [i["message"] for i in dq["issues"] if i["severity"] == "critical"]
        if not problems:
            problems = ["There are not enough complete, in-range readings to make a reliable estimate."]
        result.update(
            headline="Cannot assess: sensor data unavailable or implausible",
            explanation=["No failure prediction was made, because acting on bad data could be worse than acting on none."] + problems,
            recommended_action=(
                f"Check the {_sensor_names(dq['issues'])} sensor: wiring, power, connector and data gateway. "
                "If it cannot be restored promptly, arrange a manual inspection of the machine, "
                "because the system currently has no evidence either way."
            ),
        )
        return result

    values = {f: float(np.median([r[f] for r in usable])) for f in config.FEATURES}
    p = predict_probability(bundle, values)
    level = risk_level(p)
    result.update(risk_score=round(p, 3), risk_level=level, evidence_sufficient=True)

    drivers = _drivers(values, bundle.meta["baseline"])
    driver_lines = [_driver_text(f, values[f], mu) for f, _, mu in drivers]
    trend_lines = _trend_notes(rows, dq["sensor_status"])
    checks = "; ".join(f"{config.FIELD_LABELS[f].lower()}: {TYPICAL_CHECKS[f]}" for f, _, _ in drivers[:2])
    checks_line = f" Typical checks for this pattern ({checks})." if checks else ""

    # 2. Data usable but questionable: show the estimate for reference, push verification.
    if dq["status"] == "suspect":
        issue_lines = [i["message"] for i in dq["issues"]]
        elevated = level in ("Medium", "High")
        result.update(
            status="data_check",
            priority=2 if elevated else 4,
            needs_human_inspection=True,
            headline=("Elevated risk estimate, but the data is questionable" if elevated
                      else "Risk looks low, but sensor data needs checking"),
            explanation=(
                [f"Estimated failure risk is {p:.0%} ({level}), but it may be distorted by the data issues below, "
                 "so treat it as a reference only."]
                + issue_lines + driver_lines + trend_lines
            ),
            recommended_action=(
                f"Check the {_sensor_names(dq['issues'])} sensor and inspect the machine in person before acting on this estimate."
                if elevated else
                f"Check the {_sensor_names(dq['issues'])} sensor at the next routine round. No urgent machine action is indicated."
            ),
        )
        return result

    # 3. Reliable data: act on the estimate.
    base = [f"Sensor data is complete and consistent. Estimated failure risk is {p:.0%} ({level})."]
    if level == "High":
        result.update(
            status="high_risk", priority=0, needs_human_inspection=True,
            headline="High failure risk: inspect this machine first",
            explanation=base + driver_lines + trend_lines,
            recommended_action=(
                "Prioritise this machine for inspection at the next safe stop. A technician should confirm the "
                "condition before any repair or part replacement." + checks_line
            ),
        )
    elif level == "Medium":
        result.update(
            status="warning", priority=3, needs_human_inspection=False,
            headline="Elevated risk: schedule an inspection",
            explanation=base + driver_lines + trend_lines,
            recommended_action="Schedule an inspection in the next planned maintenance window and keep watching the trend." + checks_line,
        )
    else:
        result.update(
            status="healthy", priority=5, needs_human_inspection=False,
            headline="Operating normally",
            explanation=base + (driver_lines or ["All readings are close to the healthy baseline."]),
            recommended_action="No action needed. Continue routine monitoring.",
        )
    return result
