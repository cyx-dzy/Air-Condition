from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DispatchConfig, ModelConfig
from .data import build_next_day_features, generate_synthetic_data, load_real_data, make_daily_samples
from .dispatch import optimize_dispatch
from .model import predict, train_forecaster, training_predictions
from .scenarios import generate_scenarios
from .visualize import save_dispatch_plot, save_prediction_plot


def run_pipeline(
    data_path: str | None,
    output_dir: str | Path = "outputs",
    synthetic_days: int = 70,
    model_config: ModelConfig | None = None,
    dispatch_config: DispatchConfig | None = None,
) -> dict[str, Path]:
    model_config = model_config or ModelConfig()
    dispatch_config = dispatch_config or DispatchConfig()
    output = Path(output_dir)
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    df = load_real_data(data_path) if data_path else generate_synthetic_data(days=synthetic_days, seed=model_config.seed)
    features, y_load, y_pv, clean = make_daily_samples(df)
    next_features = build_next_day_features(clean)

    load_model, load_scaler = train_forecaster(features, y_load, model_config)
    pv_model, pv_scaler = train_forecaster(features, y_pv, model_config)
    load_pred = np.maximum(0, predict(load_model, load_scaler, next_features))
    pv_pred = np.maximum(0, predict(pv_model, pv_scaler, next_features))

    train_load_pred = training_predictions(load_model, load_scaler, features)
    train_pv_pred = training_predictions(pv_model, pv_scaler, features)
    load_scenarios = generate_scenarios(
        load_pred,
        y_load - train_load_pred,
        dispatch_config.scenarios,
        model_config.seed,
    )
    pv_scenarios = generate_scenarios(
        pv_pred,
        y_pv - train_pv_pred,
        dispatch_config.scenarios,
        model_config.seed + 1,
    )

    schedule, metrics = optimize_dispatch(load_scenarios, pv_scenarios, dispatch_config)
    predictions = pd.DataFrame({"hour": np.arange(1, 25), "load_kw": load_pred, "pv_kw": pv_pred})
    predictions.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(load_scenarios, columns=[f"h{h}" for h in range(1, 25)]).to_csv(
        output / "scenarios_load.csv", index=False
    )
    pd.DataFrame(pv_scenarios, columns=[f"h{h}" for h in range(1, 25)]).to_csv(
        output / "scenarios_pv.csv", index=False
    )
    schedule.to_csv(output / "schedule.csv", index=False)
    pd.Series(metrics).to_csv(output / "metrics.csv", header=["value"])
    save_prediction_plot(load_pred, pv_pred, load_scenarios, pv_scenarios, figures / "prediction_scenarios.png")
    save_dispatch_plot(schedule, figures / "dispatch_schedule.png")

    return {
        "predictions": output / "predictions.csv",
        "load_scenarios": output / "scenarios_load.csv",
        "pv_scenarios": output / "scenarios_pv.csv",
        "schedule": output / "schedule.csv",
        "metrics": output / "metrics.csv",
        "prediction_plot": figures / "prediction_scenarios.png",
        "dispatch_plot": figures / "dispatch_schedule.png",
    }

