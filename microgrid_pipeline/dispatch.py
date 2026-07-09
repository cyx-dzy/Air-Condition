from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.optimize._milp import LinearConstraint as _LinearConstraint

from .config import DispatchConfig, default_grid_price


class VarIndex:
    def __init__(self, scenarios: int, horizon: int):
        self.scenarios = scenarios
        self.horizon = horizon
        self.offsets: dict[str, int] = {}
        cursor = 0
        for name, size in [
            ("p_ch", horizon),
            ("p_dis", horizon),
            ("e_bat", horizon),
            ("u_ch", horizon),
            ("u_dis", horizon),
            ("grid_buy", scenarios * horizon),
            ("grid_sell", scenarios * horizon),
            ("cost", scenarios),
            ("var", 1),
            ("z", scenarios),
        ]:
            self.offsets[name] = cursor
            cursor += size
        self.size = cursor

    def h(self, name: str, t: int) -> int:
        return self.offsets[name] + t

    def sh(self, name: str, s: int, t: int) -> int:
        return self.offsets[name] + s * self.horizon + t

    def s(self, name: str, s: int) -> int:
        return self.offsets[name] + s

    @property
    def var(self) -> int:
        return self.offsets["var"]


def optimize_dispatch(
    load_scenarios: np.ndarray,
    pv_scenarios: np.ndarray,
    config: DispatchConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    scenarios, horizon = load_scenarios.shape
    idx = VarIndex(scenarios, horizon)
    price_grid = np.asarray(default_grid_price(), dtype=float)
    price_sell = np.full(horizon, 0.3)
    pi = np.full(scenarios, 1.0 / scenarios)

    c = np.zeros(idx.size)
    risk_weight = config.risk_weight
    for s in range(scenarios):
        c[idx.s("cost", s)] = (1 - risk_weight) * pi[s]
        c[idx.s("z", s)] = risk_weight * pi[s] / (1 - config.alpha)
    c[idx.var] = risk_weight

    lower = np.zeros(idx.size)
    upper = np.full(idx.size, np.inf)
    integrality = np.zeros(idx.size)
    lower[idx.var] = -np.inf

    for t in range(horizon):
        upper[idx.h("p_ch", t)] = config.battery_power_max
        upper[idx.h("p_dis", t)] = config.battery_power_max
        lower[idx.h("e_bat", t)] = config.battery_capacity_min
        upper[idx.h("e_bat", t)] = config.battery_capacity_max
        upper[idx.h("u_ch", t)] = 1
        upper[idx.h("u_dis", t)] = 1
        integrality[idx.h("u_ch", t)] = 1
        integrality[idx.h("u_dis", t)] = 1

    for s in range(scenarios):
        for t in range(horizon):
            upper[idx.sh("grid_buy", s, t)] = config.grid_power_max
            upper[idx.sh("grid_sell", s, t)] = config.grid_power_max

    rows = []
    lb = []
    ub = []

    def add_eq(coefs: dict[int, float], rhs: float) -> None:
        row = np.zeros(idx.size)
        for col, value in coefs.items():
            row[col] = value
        rows.append(row)
        lb.append(rhs)
        ub.append(rhs)

    def add_le(coefs: dict[int, float], rhs: float) -> None:
        row = np.zeros(idx.size)
        for col, value in coefs.items():
            row[col] = value
        rows.append(row)
        lb.append(-np.inf)
        ub.append(rhs)

    def add_ge(coefs: dict[int, float], rhs: float) -> None:
        row = np.zeros(idx.size)
        for col, value in coefs.items():
            row[col] = value
        rows.append(row)
        lb.append(rhs)
        ub.append(np.inf)

    for t in range(horizon):
        add_le({idx.h("u_ch", t): 1, idx.h("u_dis", t): 1}, 1)
        add_le({idx.h("p_ch", t): 1, idx.h("u_ch", t): -config.battery_power_max}, 0)
        add_le({idx.h("p_dis", t): 1, idx.h("u_dis", t): -config.battery_power_max}, 0)
        if t == 0:
            add_eq(
                {
                    idx.h("e_bat", t): 1,
                    idx.h("p_ch", t): -config.eta_charge,
                    idx.h("p_dis", t): 1 / config.eta_discharge,
                },
                config.battery_initial,
            )
        else:
            add_eq(
                {
                    idx.h("e_bat", t): 1,
                    idx.h("e_bat", t - 1): -1,
                    idx.h("p_ch", t): -config.eta_charge,
                    idx.h("p_dis", t): 1 / config.eta_discharge,
                },
                0,
            )

    add_eq({idx.h("e_bat", horizon - 1): 1}, config.battery_initial)

    for s in range(scenarios):
        cost_coefs = {idx.s("cost", s): 1}
        for t in range(horizon):
            add_eq(
                {
                    idx.sh("grid_buy", s, t): 1,
                    idx.sh("grid_sell", s, t): -1,
                    idx.h("p_dis", t): 1,
                    idx.h("p_ch", t): -1,
                },
                load_scenarios[s, t] - pv_scenarios[s, t],
            )
            cost_coefs[idx.sh("grid_buy", s, t)] = -price_grid[t]
            cost_coefs[idx.sh("grid_sell", s, t)] = price_sell[t]
        add_eq(cost_coefs, 0)
        add_ge({idx.s("z", s): 1, idx.s("cost", s): -1, idx.var: 1}, 0)

    constraints = LinearConstraint(np.vstack(rows), np.asarray(lb), np.asarray(ub))
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"time_limit": 60, "mip_rel_gap": 0.02},
    )
    if not result.success:
        raise RuntimeError(f"璋冨害浼樺寲澶辫触: {result.message}")

    x = result.x
    schedule = pd.DataFrame(
        {
            "hour": np.arange(1, horizon + 1),
            "p_charge_kw": [x[idx.h("p_ch", t)] for t in range(horizon)],
            "p_discharge_kw": [x[idx.h("p_dis", t)] for t in range(horizon)],
            "battery_energy_kwh": [x[idx.h("e_bat", t)] for t in range(horizon)],
            "price_yuan_per_kwh": price_grid,
            "avg_grid_buy_kw": [
                np.mean([x[idx.sh("grid_buy", s, t)] for s in range(scenarios)]) for t in range(horizon)
            ],
            "avg_grid_sell_kw": [
                np.mean([x[idx.sh("grid_sell", s, t)] for s in range(scenarios)]) for t in range(horizon)
            ],
        }
    )
    metrics = {
        "objective": float(result.fun),
        "expected_cost": float(np.mean([x[idx.s("cost", s)] for s in range(scenarios)])),
        "var": float(x[idx.var]),
        "cvar_proxy": float(
            x[idx.var] + sum(pi[s] * x[idx.s("z", s)] for s in range(scenarios)) / (1 - config.alpha)
        ),
    }
    return schedule, metrics


