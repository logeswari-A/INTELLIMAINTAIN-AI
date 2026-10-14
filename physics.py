"""Shared toy physics for the *simulated* data.

A hidden condition value d (0 = like new, ~1 = severely degraded) drives all sensors.
Both the model's training data and the live machine simulator use this function, so the
simulator is a stand-in for real plant data and nothing more.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def failure_probability(d):
    """Simulated ground truth: chance of failure within the monitoring horizon."""
    return sigmoid(9.0 * (np.asarray(d, dtype=float) - 0.6))


def sensor_signature(d, hours, rng, noise_scale=1.0):
    """Return (temperature_c, vibration_mm_s, rpm) for condition d at the given hours."""
    d = np.asarray(d, dtype=float)
    hours = np.asarray(hours, dtype=float)
    size = d.shape if d.ndim else None
    wear = np.clip(hours / 9000.0, 0.0, 1.2)
    rpm = 1500.0 - 150.0 * d + rng.normal(0.0, 60.0 * noise_scale, size)
    temp = 62.0 + 22.0 * d + 4.0 * wear + rng.normal(0.0, 2.5 * noise_scale, size)
    vib = (
        2.0
        + 5.5 * np.power(np.clip(d, 0.0, None), 1.3)
        + 0.8 * wear
        + rng.normal(0.0, 0.45 * noise_scale, size)
    )
    vib = np.clip(vib, 0.05, None)
    return temp, vib, rpm
