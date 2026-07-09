from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "temperature_c", "load_kw", "pv_kw", "event_flag"}


def generate_synthetic_data(days: int = 70, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=days * 24, freq="h")
    rows = []
    for day in range(days):
        is_event_day = day % 7 in {4, 5} or rng.random() < 0.15
        event_scale = rng.uniform(160, 260) if is_event_day else 0.0
        for hour in range(24):
            temp = 22 + 8 * np.sin(np.pi * (hour - 6) / 12) + rng.normal(0, 1.2)
            base_load = 120 + 35 * np.sin(np.pi * hour / 12) + 1.8 * max(temp - 24, 0)
            evening_event = event_scale if 18 <= hour <= 21 else 0.0
            load = max(40, base_load + evening_event + rng.normal(0, 12))
            pv = max(0, 420 * np.sin(np.pi * (hour - 6) / 12) + rng.normal(0, 18))
            if hour <= 6 or hour >= 19:
                pv = 0.0
            rows.append(
                {
                    "timestamp": timestamps[day * 24 + hour],
                    "temperature_c": temp,
                    "load_kw": load,
                    "pv_kw": pv,
                    "event_flag": int(is_event_day),
                }
            )
    return pd.DataFrame(rows)


def load_real_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要字段: {', '.join(sorted(missing))}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def make_daily_samples(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    clean = df.copy()
    clean["hour"] = clean["timestamp"].dt.hour
    usable_rows = (len(clean) // 24) * 24
    clean = clean.iloc[:usable_rows].reset_index(drop=True)
    if usable_rows < 48:
        raise ValueError("至少需要 48 小时数据，建议 30 天以上。")

    days = usable_rows // 24
    features = []
    load_targets = []
    pv_targets = []
    for day in range(days):
        block = clean.iloc[day * 24 : (day + 1) * 24]
        x = np.vstack(
            [
                block["temperature_c"].to_numpy(dtype=float),
                block["hour"].to_numpy(dtype=float) / 23.0,
                block["event_flag"].to_numpy(dtype=float),
                block["load_kw"].shift(1).bfill().to_numpy(dtype=float),
            ]
        )
        features.append(x)
        load_targets.append(block["load_kw"].to_numpy(dtype=float))
        pv_targets.append(block["pv_kw"].to_numpy(dtype=float))
    return np.stack(features), np.stack(load_targets), np.stack(pv_targets), clean


def build_next_day_features(clean: pd.DataFrame) -> np.ndarray:
    last_day = clean.iloc[-24:].copy()
    next_features = np.vstack(
        [
            last_day["temperature_c"].to_numpy(dtype=float),
            np.arange(24, dtype=float) / 23.0,
            last_day["event_flag"].to_numpy(dtype=float),
            last_day["load_kw"].to_numpy(dtype=float),
        ]
    )
    return next_features

