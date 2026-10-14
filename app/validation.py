"""Sensor data-quality checks.

Runs BEFORE any prediction. Input is a list of reading dicts ordered oldest -> newest,
each with temperature_c, vibration_mm_s, rpm, operating_hours (None = missing).
"""
import math
from typing import Optional

import numpy as np

from .config import FEATURES, FIELD_LABELS, SENSOR_FIELDS

# Physically plausible sensor ranges. Outside these the value is treated as a fault.
BOUNDS = {
    "temperature_c": (-20.0, 200.0),
    "vibration_mm_s": (0.0, 50.0),
    "rpm": (0.0, 6000.0),
    "operating_hours": (0.0, 200000.0),
}
# Smallest jump (in sensor units) that can count as a spike, whatever the recent noise level.
SPIKE_FLOORS = {"temperature_c": 6.0, "vibration_mm_s": 2.0, "rpm": 250.0}

OFFLINE_AFTER = 3        # consecutive missing samples before a sensor counts as offline
FLATLINE_LEN = 8         # identical consecutive values that suggest a stuck sensor
INTERMITTENT_WINDOW = 20
INTERMITTENT_RATE = 0.15

CRITICAL_PENALTY = 40
WARNING_PENALTY = 15


def _val(row: dict, field: str) -> Optional[float]:
    v = row.get(field)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _in_bounds(v: Optional[float], field: str) -> bool:
    if v is None:
        return False
    lo, hi = BOUNDS[field]
    return lo <= v <= hi


def usable_rows(rows: list, n: int = 5, lookback: int = 10) -> list:
    """Most recent complete, in-range readings (oldest first) that are safe to feed the model."""
    out = []
    for r in reversed(rows[-lookback:]):
        if all(_in_bounds(_val(r, f), f) for f in FEATURES):
            out.append(r)
        if len(out) == n:
            break
    return list(reversed(out))


def _trailing_missing(vals: list) -> int:
    count = 0
    for v in reversed(vals):
        if v is None:
            count += 1
        else:
            break
    return count


def _issue(sensors, code, severity, message):
    return {"sensors": list(sensors), "code": code, "severity": severity, "message": message}


def validate(rows: list) -> dict:
    if not rows:
        return {
            "status": "unreliable",
            "score": 0,
            "issues": [_issue(FEATURES, "no_data", "critical", "No readings have been received for this machine.")],
            "sensor_status": {f: "bad" for f in FEATURES},
            "readings_checked": 0,
        }

    issues = []
    series = {f: [_val(r, f) for r in rows] for f in FEATURES}

    for f in FEATURES:
        label = FIELD_LABELS[f]
        vals = series[f]
        latest = vals[-1]
        trailing = _trailing_missing(vals)

        # Missing / offline / intermittent
        recent = vals[-INTERMITTENT_WINDOW:]
        missing_rate = sum(v is None for v in recent) / len(recent)
        if trailing >= OFFLINE_AFTER:
            issues.append(_issue([f], "offline", "critical",
                f"{label} has had no reading for the last {trailing} samples."))
        elif len(recent) >= 10 and missing_rate > INTERMITTENT_RATE:
            n_missing = sum(v is None for v in recent)
            issues.append(_issue([f], "intermittent", "warning",
                f"{label} dropped out in {n_missing} of the last {len(recent)} samples."))
        elif latest is None:
            issues.append(_issue([f], "dropout", "warning",
                f"{label} is missing from the latest sample."))

        if latest is None:
            continue

        # Physically implausible value
        if not _in_bounds(latest, f):
            lo, hi = BOUNDS[f]
            issues.append(_issue([f], "out_of_range", "critical",
                f"{label} reads {latest:g}, outside the plausible range {lo:g} to {hi:g}."))
            continue

        if f in SENSOR_FIELDS:
            # Flatline (stuck sensor): real sensors always show some noise.
            tail = vals[-FLATLINE_LEN:]
            if len(tail) == FLATLINE_LEN and all(v is not None for v in tail) and max(tail) == min(tail):
                issues.append(_issue([f], "flatline", "warning",
                    f"{label} has shown the identical value ({latest:g}) for {FLATLINE_LEN} samples in a row, "
                    "which suggests a stuck sensor."))
            else:
                # Sudden jump compared with the recent past
                prev = [v for v in vals[-16:-1] if v is not None]
                if len(prev) >= 8:
                    med = float(np.median(prev))
                    threshold = max(5.0 * float(np.std(prev)), SPIKE_FLOORS[f])
                    if abs(latest - med) > threshold:
                        issues.append(_issue([f], "spike", "warning",
                            f"{label} jumped to {latest:g} from a recent level of about {med:.1f}, "
                            "faster than the machine normally changes."))

    # Cross-sensor consistency: vibration while the speed sensor says the machine is stopped
    rpm_now, vib_now = series["rpm"][-1], series["vibration_mm_s"][-1]
    if (rpm_now is not None and vib_now is not None
            and _in_bounds(rpm_now, "rpm") and _in_bounds(vib_now, "vibration_mm_s")
            and rpm_now < 100 and vib_now > 1.5):
        issues.append(_issue(["rpm", "vibration_mm_s"], "inconsistent", "warning",
            f"Speed reads {rpm_now:g} rpm (stopped) while vibration is {vib_now:g} mm/s. "
            "Either the speed sensor or the vibration sensor is wrong."))

    # Operating-hours counter should never go backwards
    hours = [v for v in series["operating_hours"] if v is not None]
    if len(hours) >= 2 and hours[-1] < max(hours[:-1]) - 1e-9:
        issues.append(_issue(["operating_hours"], "counter_decrease", "warning",
            "The operating-hours counter went down, which means a maintenance reset or a counter fault."))

    score = 100
    for i in issues:
        score -= CRITICAL_PENALTY if i["severity"] == "critical" else WARNING_PENALTY
    score = max(0, score)

    if any(i["severity"] == "critical" for i in issues):
        status = "unreliable"
    elif issues:
        status = "suspect"
    else:
        status = "reliable"

    sensor_status = {f: "ok" for f in FEATURES}
    for i in issues:
        for f in i["sensors"]:
            if i["severity"] == "critical":
                sensor_status[f] = "bad"
            elif sensor_status[f] != "bad":
                sensor_status[f] = "suspect"

    return {
        "status": status,
        "score": score,
        "issues": issues,
        "sensor_status": sensor_status,
        "readings_checked": len(rows),
    }
