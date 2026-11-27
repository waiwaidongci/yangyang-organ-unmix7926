from itertools import combinations

import numpy as np
from scipy.stats import rankdata

from .schema import CALIBRATION, HOLDOUT, MIC_KEYS, MIC_LABELS
from .separation import MIN_SIGNAL_RMS, Separation

REVIEW_WINDOW_MS = 10
RESIDUAL_REVIEW_PERCENT = 10.0
CORRELATION_REVIEW_LEVEL = 0.35


def rms(values: np.ndarray, axis: int | None = None):
    return np.sqrt(np.mean(np.square(values), axis=axis))


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0:
        return None
    return float(np.clip(np.dot(left_centered, right_centered) / denominator, -1.0, 1.0))


def dependence_rows(scores: np.ndarray, calibration_count: int, components: list[str]) -> list[dict]:
    records = []
    for split, values in ((CALIBRATION, scores[:calibration_count]), (HOLDOUT, scores[calibration_count:])):
        # Tied ranks keep repeated amplitudes from acquiring spurious order.
        amplitude_ranks = rankdata(np.abs(values), axis=0)
        varying_signal = np.ptp(values, axis=0) > MIN_SIGNAL_RMS
        varying_amplitude = np.ptp(np.abs(values), axis=0) > MIN_SIGNAL_RMS
        for left, right in combinations(range(len(components)), 2):
            records.append({
                "split": split,
                "left": components[left],
                "right": components[right],
                "pearson": _correlation(values[:, left], values[:, right])
                if varying_signal[left] and varying_signal[right] else None,
                "amplitude_spearman": _correlation(amplitude_ranks[:, left], amplitude_ranks[:, right])
                if varying_amplitude[left] and varying_amplitude[right] else None,
            })
    return records


def residual_rows(microphones: np.ndarray, separation: Separation, calibration_count: int) -> list[dict]:
    calibration_scale = rms(microphones[:calibration_count] - separation.mean, axis=0)
    calibration_error = rms(separation.residual[:calibration_count], axis=0)
    holdout_error = rms(separation.residual[calibration_count:], axis=0)
    records = []
    for key, scale, fit, held in zip(MIC_KEYS, calibration_scale, calibration_error, holdout_error):
        records.append({
            "channel": MIC_LABELS[key],
            "calibration_signal_rms": float(scale),
            "calibration_residual_rms": float(fit),
            "holdout_residual_rms": float(held),
            "holdout_residual_pct": float(100 * held / scale) if scale > MIN_SIGNAL_RMS else None,
        })
    return records


def review_rows(rows: list[dict], residual: np.ndarray, calibration_count: int, scale: float) -> list[dict]:
    rate = rows[0]["sample_rate_hz"]
    window_samples = round(rate * REVIEW_WINDOW_MS / 1000)
    records = []
    for start in range(calibration_count, len(rows), window_samples):
        end = min(len(rows), start + window_samples)
        window = residual[start:end]
        first_index = rows[start]["sample_index"]
        last_index = rows[end - 1]["sample_index"]
        channel_rms = rms(window, axis=0)
        strongest_channel = int(np.argmax(channel_rms))
        residual_pct = round(100 * float(rms(window)) / scale, 6)
        records.append({
            "priority": 0,
            "first_id": rows[start]["id"],
            "first_sample": first_index,
            "last_sample": last_index,
            "start_ms": 1000 * first_index / rate,
            "end_ms_exclusive": 1000 * (last_index + 1) / rate,
            "sample_count": end - start,
            "residual_pct": residual_pct,
            "peak_channel": MIC_LABELS[MIC_KEYS[strongest_channel]] if residual_pct > 0 else None,
        })
    records.sort(key=lambda record: (-record["residual_pct"], record["first_sample"]))
    for priority, record in enumerate(records, 1):
        record["priority"] = priority
    return records
