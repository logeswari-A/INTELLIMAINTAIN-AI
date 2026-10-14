"""SIMULATED machine fleet. Nothing here talks to real equipment.

Readings are a pure function of (seed, machine, step), so the simulation is repeatable and
survives restarts. Each machine follows a scenario so the demo shows every case the system
must handle: healthy, slowly worn, fast-degrading, and several kinds of bad sensor data.
"""
import numpy as np

from . import config
from .physics import sensor_signature

# scenario -> what it demonstrates
FLEET = [
    {"id": "M-01", "name": "CNC mill A", "type": "CNC mill", "scenario": "healthy", "base_hours": 1200, "base_d": 0.08},
    {"id": "M-02", "name": "Hydraulic press B", "type": "Hydraulic press", "scenario": "healthy", "base_hours": 5400, "base_d": 0.12},
    {"id": "M-03", "name": "Conveyor motor C", "type": "Conveyor motor", "scenario": "degrading", "base_hours": 6100, "base_d": 0.15},
    {"id": "M-04", "name": "Air compressor D", "type": "Air compressor", "scenario": "healthy", "base_hours": 2300, "base_d": 0.10},
    {"id": "M-05", "name": "Coolant pump E", "type": "Pump", "scenario": "mild_wear", "base_hours": 7200, "base_d": 0.52},
    {"id": "M-06", "name": "Packaging line F", "type": "Packaging line", "scenario": "healthy", "base_hours": 800, "base_d": 0.06},
    {"id": "M-07", "name": "Gearbox G", "type": "Gearbox", "scenario": "flaky_vibration", "base_hours": 3900, "base_d": 0.58},
    {"id": "M-08", "name": "Extruder H", "type": "Extruder", "scenario": "healthy", "base_hours": 4600, "base_d": 0.14},
    {"id": "M-09", "name": "Boiler feed pump I", "type": "Pump", "scenario": "stuck_temp", "base_hours": 3100, "base_d": 0.11},
    {"id": "M-10", "name": "Lathe J", "type": "Lathe", "scenario": "slow_degrade", "base_hours": 5000, "base_d": 0.30},
    {"id": "M-11", "name": "Welding robot K", "type": "Robot", "scenario": "rpm_offline", "base_hours": 2700, "base_d": 0.09},
    {"id": "M-12", "name": "Exhaust fan L", "type": "Fan", "scenario": "rpm_inconsistent", "base_hours": 3300, "base_d": 0.13},
]

HOURS_PER_STEP = 0.02


def condition_at(machine: dict, step: int) -> float:
    s, d0 = machine["scenario"], machine["base_d"]
    if s == "degrading":
        return min(1.1, d0 + 0.0050 * step)
    if s == "slow_degrade":
        return min(1.1, d0 + 0.0020 * step)
    if s == "mild_wear":
        return d0 + 0.0003 * step
    return d0


def generate_reading(machine_index: int, machine: dict, step: int) -> dict:
    """Return one simulated reading. Faulty-sensor scenarios start at fixed steps."""
    rng = np.random.default_rng([config.SIM_SEED, machine_index, step])
    hours = machine["base_hours"] + HOURS_PER_STEP * step
    d = condition_at(machine, step)
    temp, vib, rpm = sensor_signature(d, hours, rng, noise_scale=0.35)
    temp, vib, rpm = float(temp), float(vib), float(rpm)
    scenario = machine["scenario"]

    if scenario == "flaky_vibration" and step >= 90:
        roll = rng.random()
        if roll < 0.30:
            vib = None
        elif roll < 0.38:
            vib += float(rng.uniform(15, 25))   # occasional electrical spike
    elif scenario == "stuck_temp" and step >= 100:
        frozen, *_ = sensor_signature(d, machine["base_hours"] + HOURS_PER_STEP * 100,
                                      np.random.default_rng([config.SIM_SEED, machine_index, 100]), 0.35)
        temp = float(frozen)                   # same value every sample from step 100
        temp = round(temp, 1)
    elif scenario == "rpm_offline" and step >= 140:
        rpm = None
    elif scenario == "rpm_inconsistent" and step >= 120:
        rpm = float(abs(rng.normal(3, 2)))     # speed sensor reads ~0 while machine runs

    return {
        "temperature_c": None if temp is None else round(temp, 1),
        "vibration_mm_s": None if vib is None else round(vib, 2),
        "rpm": None if rpm is None else round(rpm, 0),
        "operating_hours": round(hours, 2),
    }
