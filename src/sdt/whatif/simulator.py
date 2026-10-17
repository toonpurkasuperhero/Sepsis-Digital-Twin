"""Semi-synthetic ICU world with KNOWN treatment effects (plan, Phase 5 validation 1).

A latent severity s drives the observed vitals and also drives treatment
decisions (sicker patients get more fluids and vasopressors): this is the
time-varying confounding that makes naive "what-if" models wrong. Because the
world is simulated, the true counterfactual under any plan can be computed
with common random numbers and compared with each estimator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class World:
    fluid_map_effect: float = 3.0       # mmHg rise in MAP per bolus hour (direct)
    vaso_map_effect: float = 6.0        # mmHg rise in MAP while a vasopressor runs
    fluid_severity_effect: float = -0.08
    vaso_severity_effect: float = -0.05
    drift: float = 0.04
    noise_obs: float = 0.6              # observation noise (in severity-scaled units)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(n: int, T: int, world: World, seed: int, plan: dict | None = None,
             plan_start: int | None = None) -> pd.DataFrame:
    """Simulates n stays for T hours. `plan` = {"fluid": 0/1, "vaso": 0/1} forces
    treatments from `plan_start` onward (None = clinicians' confounded policy).
    The same seed gives the same patients and noise, so factual and
    counterfactual runs differ only by the plan."""
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 1, n)
    eps_s = rng.normal(0, 0.15, (T, n))
    eps_o = rng.normal(0, 1, (T, 3, n))
    u_fluid, u_vaso = rng.random((T, n)), rng.random((T, n))
    vaso_on = np.zeros(n)
    rows = []
    for t in range(T):
        map_ = 80 - 8 * s + world.vaso_map_effect * vaso_on + world.noise_obs * 4 * eps_o[t, 0]
        lac = np.exp(0.4 + 0.35 * s + world.noise_obs * 0.15 * eps_o[t, 1])
        hr = 85 + 9 * s + world.noise_obs * 5 * eps_o[t, 2]
        forced = plan is not None and t >= plan_start
        if forced:
            fluid = np.full(n, float(plan.get("fluid", 0)))
            vaso_on = np.full(n, float(plan.get("vaso", 0)))
        else:                                   # confounded clinician policy
            fluid = (u_fluid[t] < _sigmoid(-2.5 + 0.12 * (72 - map_))).astype(float)
            start = u_vaso[t] < _sigmoid(-4.0 + 0.18 * (65 - map_))
            vaso_on = np.where(vaso_on == 1, (u_vaso[t] < 0.92).astype(float), start.astype(float))
        map_ = map_ + world.fluid_map_effect * fluid
        rows.append(pd.DataFrame({"stay_id": np.arange(n), "hour": t, "map": map_, "lactate": lac, "hr": hr,
                                  "fluid": fluid, "vaso": vaso_on, "severity": s}))
        s = s + world.drift + world.fluid_severity_effect * fluid + world.vaso_severity_effect * vaso_on + eps_s[t]
    return pd.concat(rows, ignore_index=True)


def true_effect(n, T, world, seed, plan_a, plan_b, start, outcome="map", horizon=6) -> np.ndarray:
    """Per-stay true effect of plan_a vs plan_b on the outcome `horizon` hours after `start`."""
    ya = simulate(n, T, world, seed, plan_a, start)
    yb = simulate(n, T, world, seed, plan_b, start)
    t = start + horizon - 1          # engine hour_ahead=horizon is hour start+horizon-1
    return (ya[ya.hour == t][outcome].to_numpy() - yb[yb.hour == t][outcome].to_numpy())
