from __future__ import annotations

import numpy as np


def generate_scenarios(
    point_prediction: np.ndarray,
    residuals: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hourly_std = residuals.std(axis=0)
    hourly_std = np.where(hourly_std < 1e-6, 1.0, hourly_std)
    noise = rng.normal(0, hourly_std, size=(count, point_prediction.shape[0]))
    return np.maximum(0, point_prediction[None, :] + noise)

