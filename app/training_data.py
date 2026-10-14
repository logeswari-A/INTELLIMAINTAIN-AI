"""Generate the SIMULATED training set for the failure-risk model."""
import numpy as np
import pandas as pd

from .physics import failure_probability, sensor_signature


def make_training_data(n: int = 8000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 9000, n)
    wear = hours / 9000.0
    condition = np.clip(rng.normal(0.10 + 0.30 * wear, 0.18), 0.0, 1.2)
    # A small share of severely degraded machines so the model sees the extreme end too.
    severe = rng.random(n) < 0.05
    condition = np.where(severe, rng.uniform(0.6, 1.2, n), condition)
    temp, vib, rpm = sensor_signature(condition, hours, rng, noise_scale=1.0)
    failure = rng.random(n) < failure_probability(condition)
    return pd.DataFrame(
        {
            "temperature_c": temp.round(1),
            "vibration_mm_s": vib.round(2),
            "rpm": rpm.round(0),
            "operating_hours": hours.round(0),
            "condition": condition,
            "failure": failure.astype(int),
        }
    )
