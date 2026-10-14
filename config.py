"""Central configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DB_PATH = Path(os.getenv("INTELLIMAINTAIN_DB", DATA_DIR / "intellimaintain.db"))
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
MODEL_META_PATH = MODEL_DIR / "risk_model_meta.json"

# Model inputs. Temperature, vibration and rpm come from sensors;
# operating_hours is a controller counter that should only ever increase.
FEATURES = ["temperature_c", "vibration_mm_s", "rpm", "operating_hours"]
SENSOR_FIELDS = ["temperature_c", "vibration_mm_s", "rpm"]

FIELD_LABELS = {
    "temperature_c": "Temperature",
    "vibration_mm_s": "Vibration",
    "rpm": "Rotational speed",
    "operating_hours": "Operating hours",
}
FIELD_UNITS = {
    "temperature_c": " °C",
    "vibration_mm_s": " mm/s",
    "rpm": " rpm",
    "operating_hours": " h",
}

# Risk bands applied to the model's estimated failure probability.
RISK_MEDIUM = float(os.getenv("RISK_MEDIUM", "0.30"))
RISK_HIGH = float(os.getenv("RISK_HIGH", "0.60"))

# Simulation. SIM_TICK_SECONDS=0 disables the background ticker (use POST /api/simulate/tick).
SIM_TICK_SECONDS = float(os.getenv("SIM_TICK_SECONDS", "5"))
SIM_SEED = int(os.getenv("SIM_SEED", "42"))
SEED_HISTORY = 150          # readings generated per machine on first start
ASSESSMENT_WINDOW = 60      # most recent readings used for validation and trends

DATA_LABEL = "SIMULATED"    # shown in the UI and API so nobody mistakes it for plant data
