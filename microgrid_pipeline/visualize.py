from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_prediction_plot(
    load_pred: np.ndarray,
    pv_pred: np.ndarray,
    load_scenarios: np.ndarray,
    pv_scenarios: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hours = np.arange(1, 25)
    plt.figure(figsize=(11, 5.5))
    for scenario in load_scenarios:
        plt.plot(hours, scenario, color="#cc6666", alpha=0.25, linewidth=1)
    for scenario in pv_scenarios:
        plt.plot(hours, scenario, color="#d6b834", alpha=0.25, linewidth=1)
    plt.plot(hours, load_pred, "o-", color="#b42318", linewidth=2, label="Load forecast")
    plt.plot(hours, pv_pred, "s-", color="#b7791f", linewidth=2, label="PV forecast")
    plt.title("Forecast scenarios")
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_dispatch_plot(schedule: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hours = schedule["hour"]
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.bar(hours, schedule["p_charge_kw"], color="#2f855a", alpha=0.75, label="Charge")
    ax1.bar(hours, -schedule["p_discharge_kw"], color="#c53030", alpha=0.75, label="Discharge")
    ax1.plot(hours, schedule["battery_energy_kwh"], "o-", color="#2b6cb0", linewidth=2, label="Battery energy")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("kW / kWh")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.step(hours, schedule["price_yuan_per_kwh"], where="mid", color="#2d3748", linestyle="--", label="Price")
    ax2.set_ylabel("Yuan/kWh")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.title("Risk-aware battery dispatch")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

