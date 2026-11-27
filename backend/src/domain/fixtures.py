import numpy as np

from .schema import CALIBRATION, HOLDOUT, MIC_KEYS

DEMO_SEED = 6407
DEMO_RATE = 4_000
CALIBRATION_SAMPLES = 160
DEMO_SAMPLES = 240


def demo_rows() -> list[dict]:
    """A short synchronized bench mock-up with four mixtures and no source labels."""
    rng = np.random.default_rng(DEMO_SEED)
    time = np.arange(DEMO_SAMPLES) / DEMO_RATE
    pipe = 0.23 * (np.sin(2 * np.pi * 173 * time) + 0.19 * np.sin(2 * np.pi * 346 * time))
    wind = 0.065 * np.convolve(rng.laplace(0.0, 1.0, DEMO_SAMPLES), [0.2, 0.6, 0.2], mode="same")
    strikes = np.zeros(DEMO_SAMPLES)
    strike_indices = np.sort(rng.choice(DEMO_SAMPLES - 7, size=20, replace=False))
    strikes[strike_indices] = rng.uniform(0.16, 0.32, len(strike_indices))
    action = np.convolve(strikes, [1.0, -0.5, -0.24, -0.11], mode="full")[:DEMO_SAMPLES]
    room = 0.021 * rng.uniform(-1.0, 1.0, DEMO_SAMPLES)
    sources = np.column_stack((pipe, wind, action, room))
    mixing = np.array([
        [1.00, 0.27, 0.22, 0.13],
        [0.43, 0.95, 0.17, 0.46],
        [0.36, 0.34, 0.89, 0.21],
        [0.61, 0.45, 0.39, 0.92],
    ])
    microphones = sources @ mixing.T + np.array([0.006, -0.004, 0.003, -0.002])
    return [
        {
            "id": f"LHY-G07-064-{index:04d}",
            "sample_index": index,
            "sample_rate_hz": DEMO_RATE,
            "split": CALIBRATION if index < CALIBRATION_SAMPLES else HOLDOUT,
            **{key: float(round(value, 8)) for key, value in zip(MIC_KEYS, sample)},
        }
        for index, sample in enumerate(microphones)
    ]
