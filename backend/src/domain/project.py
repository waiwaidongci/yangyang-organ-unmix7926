import numpy as np

from .fixtures import demo_rows
from .inputs import normalize_rows, normalize_settings
from .reporting import build_report
from .schema import CALIBRATION, MIC_KEYS, SPEC
from .separation import separate

__all__ = ["SPEC", "demo_rows", "normalize_rows", "analyze"]


def analyze(rows: list[dict], settings: dict) -> dict:
    records = normalize_rows(rows)
    parameters = normalize_settings(settings)
    calibration_count = sum(row["split"] == CALIBRATION for row in records)
    microphones = np.asarray([[row[key] for key in MIC_KEYS] for row in records], dtype=float)
    separation = separate(microphones, calibration_count, **parameters)
    return build_report(records, microphones, calibration_count, separation)
