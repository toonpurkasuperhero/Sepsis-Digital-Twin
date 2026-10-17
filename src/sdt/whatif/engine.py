"""Phase 5: counterfactual what-if engine (G-computation over patient history).

Method: sequential G-computation (the approach behind G-Net, Li et al. 2021).
A transition model learns next-hour observations from the recent history AND
the treatments given. A plan is answered by rolling the patient forward hour by
hour with the plan's treatments, sampling model residuals, under a bootstrap
ensemble. The result is the predicted trajectory with an interval, for the plan
versus an alternative.

Guardrails (plan, "Counterfactual what-if engine"):
  * positivity: if similar patients almost never received the requested
    treatment (propensity < `min_propensity`), the engine refuses to answer;
  * every answer carries the no-unmeasured-confounding assumption;
  * the engine never recommends a dose; it only compares plans a clinician asks about.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ASSUMPTION = ("Assumes no unmeasured confounding: every factor that drove past treatment decisions "
              "and affects the outcome is in the recorded history. Decision support only.")


@dataclass
class EngineConfig:
    obs: list[str]
    treatments: list[str]
    lags: int = 3
    n_boot: int = 5
    n_samples: int = 100
    min_propensity: float = 0.02
    transition: str = "gbm"          # "gbm" (non-linear) or "linear"; choose by held-out one-step error
    seed: int = 0


def _features(df: pd.DataFrame, cfg: EngineConfig) -> pd.DataFrame:
    """History features at hour t: obs at t..t-lags+1, treatments at t..t-lags+1,
    and the treatment given during hour t+1 (whose effect shows in obs t+1)."""
    g = df.groupby("stay_id", sort=False)
    cols = {}
    for lag in range(cfg.lags):
        for c in cfg.obs + cfg.treatments:
            cols[f"{c}_l{lag}"] = g[c].shift(lag)
    for a in cfg.treatments:
        cols[f"{a}_next"] = g[a].shift(-1)
    cols["hour"] = df["hour"]
    return pd.DataFrame(cols, index=df.index)


@dataclass
class WhatIfEngine:
    cfg: EngineConfig
    models: list = field(default_factory=list)          # bootstrap ensemble: list of {obs: (model, residuals)}
    propensity: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "WhatIfEngine":
        df = df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
        X = _features(df, self.cfg)
        Y = df.groupby("stay_id", sort=False)[self.cfg.obs].shift(-1)
        ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
        X, Y, sid = X[ok], Y[ok], df.loc[ok, "stay_id"]
        rng = np.random.default_rng(self.cfg.seed)
        stays = sid.unique()
        self.models = []
        for b in range(self.cfg.n_boot):
            pick = rng.choice(stays, len(stays))
            counts = pd.Series(pick).value_counts()
            w = sid.map(counts).fillna(0).to_numpy()      # bootstrap as stay-level weights
            member = {}
            for o in self.cfg.obs:
                m = (HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=b)
                     if self.cfg.transition == "gbm" else Ridge(alpha=1.0))
                m.fit(X, Y[o], sample_weight=w)
                member[o] = (m, (Y[o] - m.predict(X)).to_numpy()[w > 0])
            self.models.append(member)
        # positivity: probability of each treatment during the next hour given history
        for a in self.cfg.treatments:
            Xa = X.drop(columns=[f"{t}_next" for t in self.cfg.treatments])
            self.propensity[a] = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(
                Xa, X[f"{a}_next"].round().astype(int))
        self._columns = list(X.columns)
        return self

    def positivity(self, history: pd.DataFrame, plan: dict) -> dict:
        x = _features(history, self.cfg).iloc[[-1]]
        out = {}
        for a, v in plan.items():
            p1 = float(self.propensity[a].predict_proba(
                x.drop(columns=[f"{t}_next" for t in self.cfg.treatments]))[0, 1])
            out[a] = p1 if v else 1 - p1
        return out

    def rollout(self, history: pd.DataFrame, plan: dict, horizon: int) -> np.ndarray:
        """Samples [n_boot * n_samples, horizon, n_obs] future observations under `plan`
        (treatments held at the plan's values for every future hour)."""
        rng = np.random.default_rng(self.cfg.seed)
        h = history.sort_values("hour").tail(self.cfg.lags).copy()
        out = []
        for member in self.models:
            sims = pd.concat([h] * self.cfg.n_samples, keys=range(self.cfg.n_samples), names=["sample"]).reset_index(level=0).reset_index(drop=True)
            sims["stay_id"] = sims["sample"]
            traj = np.zeros((self.cfg.n_samples, horizon, len(self.cfg.obs)))
            for k in range(horizon):
                last = sims.groupby("stay_id").tail(1)
                X = _features(sims, self.cfg).loc[last.index]
                for a, v in plan.items():              # treatment during the coming hour
                    X[f"{a}_next"] = v
                X = X[self._columns]
                new = last.copy()
                new["hour"] = last["hour"] + 1
                for a, v in plan.items():              # the plan starts at the next hour; history is untouched
                    new[a] = v
                for j, o in enumerate(self.cfg.obs):
                    model, resid = member[o]
                    new[o] = model.predict(X) + rng.choice(resid, len(X))
                    traj[:, k, j] = new[o].to_numpy()
                sims = pd.concat([sims, new], ignore_index=True)
            out.append(traj)
        return np.concatenate(out)

    def query(self, history: pd.DataFrame, plan: dict, alternative: dict, horizon: int = 6) -> dict:
        pos = self.positivity(history, plan)
        if min(pos.values()) < self.cfg.min_propensity:
            return {"answered": False, "reason": (
                f"Positivity check failed: similar patients rarely received this plan "
                f"(propensity {min(pos.values()):.3f} < {self.cfg.min_propensity}). "
                "The engine will not extrapolate."), "propensity": pos}
        a = self.rollout(history, plan, horizon)
        b = self.rollout(history, alternative, horizon)
        summary = {}
        for j, o in enumerate(self.cfg.obs):
            d = a[:, :, j] - b[:, :, j]
            summary[o] = pd.DataFrame({
                "hour_ahead": np.arange(1, horizon + 1),
                "plan_mean": a[:, :, j].mean(0), "alternative_mean": b[:, :, j].mean(0),
                "difference_mean": d.mean(0),
                "difference_p05": np.percentile(d, 5, axis=0), "difference_p95": np.percentile(d, 95, axis=0)})
        return {"answered": True, "assumption": ASSUMPTION, "propensity": pos, "trajectories": summary}


def naive_effect(df: pd.DataFrame, cfg: EngineConfig, outcome: str, horizon: int,
                 start_rows: pd.DataFrame, plan_a: dict, plan_b: dict) -> np.ndarray:
    """What an ordinary predictive model would answer: outcome at t+horizon regressed on the
    CURRENT observations and the treatments received over the window, then plans swapped in.
    It ignores that treatments were chosen because of the patient's state (confounded)."""
    df = df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    g = df.groupby("stay_id", sort=False)
    X = df[cfg.obs].copy()
    for a in cfg.treatments:
        X[f"{a}_window"] = sum(g[a].shift(-k) for k in range(1, horizon + 1)) / horizon
    y = g[outcome].shift(-horizon)
    ok = X.notna().all(axis=1) & y.notna()
    m = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X[ok], y[ok])
    xa, xb = start_rows[cfg.obs].copy(), start_rows[cfg.obs].copy()
    for a in cfg.treatments:
        xa[f"{a}_window"], xb[f"{a}_window"] = plan_a.get(a, 0), plan_b.get(a, 0)
    return m.predict(xa) - m.predict(xb)
