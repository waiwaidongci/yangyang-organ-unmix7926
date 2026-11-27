from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import FastICA

ICA_SEED = 227
ICA_TOLERANCE = 1e-5
RELATIVE_RANK_TOLERANCE = 1e-8
MIN_SIGNAL_RMS = 1e-10


@dataclass(frozen=True)
class Separation:
    scores: np.ndarray
    mixing: np.ndarray
    unmixing: np.ndarray
    mean: np.ndarray
    residual: np.ndarray
    rank: int
    iterations: int
    iteration_limit: int

    @property
    def iteration_limit_reached(self) -> bool:
        return self.iterations >= self.iteration_limit


def separate(microphones: np.ndarray, calibration_count: int, n_components: int, max_iter: int) -> Separation:
    fitted = microphones[:calibration_count]
    centered = fitted - fitted.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    rank_floor = max(RELATIVE_RANK_TOLERANCE * singular_values[0], MIN_SIGNAL_RMS * np.sqrt(calibration_count))
    rank = int(np.count_nonzero(singular_values > rank_floor))
    if n_components > rank:
        raise ValueError(f"校准段有效秩为 {rank}，无法拆出 {n_components} 个成分。请减少成分数或重新采样。")

    model = FastICA(
        n_components=n_components,
        algorithm="parallel",
        whiten="unit-variance",
        whiten_solver="svd",
        fun="logcosh",
        random_state=ICA_SEED,
        max_iter=max_iter,
        tol=ICA_TOLERANCE,
    )
    model.fit(fitted)
    mixing = model.mixing_
    order = np.argsort(-np.sum(mixing**2, axis=0), kind="stable")
    peak_microphones = np.argmax(np.abs(mixing), axis=0)
    polarity = np.where(mixing[peak_microphones, np.arange(n_components)] < 0, -1.0, 1.0)[order]
    mixing = mixing[:, order] * polarity
    unmixing = model.components_[order] * polarity[:, None]
    scores = (microphones - model.mean_) @ unmixing.T
    residual = microphones - (scores @ mixing.T + model.mean_)
    if not all(np.isfinite(values).all() for values in (mixing, unmixing, scores, residual)):
        raise ValueError("拆分产生无效数值，请检查采样是否接近静音或通道是否重复。")
    return Separation(scores, mixing, unmixing, model.mean_, residual, rank, model.n_iter_, max_iter)
