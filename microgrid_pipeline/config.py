from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 32
    num_heads: int = 4
    dropout: float = 0.1
    epochs: int = 12
    learning_rate: float = 0.01
    seed: int = 2026


@dataclass(frozen=True)
class DispatchConfig:
    horizon: int = 24
    scenarios: int = 10
    battery_capacity_max: float = 1200.0
    battery_capacity_min: float = 120.0
    battery_initial: float = 300.0
    battery_power_max: float = 300.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    grid_power_max: float = 1500.0
    alpha: float = 0.95
    risk_weight: float = 0.5


def default_grid_price() -> list[float]:
    return [0.35] * 7 + [0.85] * 4 + [1.25] * 3 + [0.85] * 4 + [1.25] * 4 + [0.35] * 2

