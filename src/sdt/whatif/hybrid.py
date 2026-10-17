"""What-if engine v2: baseline course from REAL ICU data + published treatment effects.

A plan is compared with usual care:

    y_plan(t+k) = y_base(t+k) + theta_NE * (NE_plan - NE_now) + theta_F * w_(k-1) * bolus

  y_base   course of MAP / HR under usual care, from a gradient-boosting model trained on
           hourly eICU-demo + MIMIC-IV-demo histories. It is calibrated (conformal) against what
           actually happened, which already includes clinicians' routine titration; adding effect
           terms for the treatments actually given makes factual forecasts worse (validation A),
           because each recorded dose change answers a deterioration the hourly data cannot see.
  theta_NE immediate, sustained effect of a norepinephrine-equivalent dose change
           (per 0.1 mcg/kg/min); lasts while the dose is held (half-life is minutes).
  theta_F  effect of a bolus (per 500 mL) in the hour it is given, decaying with
           w = [1, 0.43, 0.15]: median MAP +7 mmHg at bolus end and +3 at 60 min
           (Glassford et al. 2014, Crit Care), so 3/7 = 0.43 one hour later.

Why effect sizes come from the literature, not from the records: double machine learning
(Chernozhukov et al. 2018) on the same records estimates almost no effect of either
treatment (|effect| < 0.2 mmHg), contradicting physiology. Vasopressors are titrated to a
MAP target within the same charting hour, so dose and pressure move together in a control
loop that observational hourly data cannot separate; the semi-synthetic check shows the
same estimator recovers known effects when the data allow it. The DML estimates are kept
as a reported diagnostic (`effects[...].data_estimate`), not used for answers.

Intervals: split conformal calibration (Vovk et al. 2005; Lei et al. 2018) of factual
multi-hour forecasts on held-out stays, combined with the published effect uncertainty.

Assumption, stated in every answer: no unmeasured confounding; additive effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ASSUMPTION = ("Assumes no unmeasured confounding: everything that drove past treatment and affects "
              "the outcome is in the recorded history. Decision support only; never a dosing recommendation.")
FLUID_KERNEL = np.array([1.0, 3 / 7, 0.15])
F_UNIT, D_UNIT = 500.0, 0.1                      # effects are per 500 mL and per 0.1 mcg/kg/min

# Published priors: (mean, sd) of the immediate effect, per unit above.
PRIORS = {
    "map": {"fluid": (5.0, 3.0,
                      "Glassford 2014: median +7 mmHg at bolus end (range 1-15), +3 at 60 min; "
                      "hour-average of the bolus hour ~ +5"),
            "ne": (20.0, 7.0,
                   "Adda 2021: NE 0.32 -> 0.26 mcg/kg/min lowered MAP 88 -> 76 mmHg (IQR 7-20 %), "
                   "i.e. ~20 (10-30) mmHg per 0.1")},
    "hr": {"fluid": (-2.0, 2.5, "Glassford 2014: median HR change -2 bpm (range -10 to 0)"),
           "ne": (0.0, 5.0, "no reliable published estimate: weak prior centred on no effect")},
}
OUTCOMES = ["map", "hr"]


# --------------------------------------------------------------------------- data

def _lags(df: pd.DataFrame, cols: list[str], n: int) -> pd.DataFrame:
    g = df.groupby("stay_id", sort=False)
    return pd.DataFrame({f"{c}_l{k}": g[c].shift(k) for c in cols for k in range(n)}, index=df.index)


def fluid_exposure(df: pd.DataFrame) -> pd.Series:
    g = df.groupby("stay_id", sort=False)["fluid_ml"]
    return sum(w * g.shift(k).fillna(0) for k, w in enumerate(FLUID_KERNEL)) / F_UNIT


@dataclass
class Effect:
    data_mean: float
    data_se: float
    prior_mean: float
    prior_sd: float
    prior_source: str

    @property
    def mean(self) -> float:                       # answers use the published effect
        return self.prior_mean

    @property
    def sd(self) -> float:
        return self.prior_sd

    def as_dict(self) -> dict:
        return {"used_mean": self.mean, "used_sd": self.sd, "source": self.prior_source,
                "data_estimate_diagnostic": self.data_mean, "data_se": self.data_se}


def dml_effects(df: pd.DataFrame, outcome: str, folds: int = 5, seed: int = 0,
                design: str = "lagged") -> tuple[np.ndarray, np.ndarray]:
    """Partially linear DML (Robinson 1988; Chernozhukov et al. 2018), cross-fitted by stay.

    design="lagged" (default): treatment = bolus volume and dose change during hour t, outcome =
      y in hour t+1, controls = everything known by the end of hour t (including y_t).
      Hourly ICU charting records titration that responds to the SAME hour's blood pressure;
      using the next hour's outcome while adjusting for the current one removes that
      within-hour reverse causation. Any effect already visible within hour t is absorbed by
      the y_t control, so this design is conservative (biased toward zero), never sign-flipped.
    design="same_hour": treatment in hour t+1 vs outcome in hour t+1 (reported for comparison).
    Returns (theta[fluid per 500 mL, NE per 0.1 mcg/kg/min], stay-clustered covariance)."""
    g = df.groupby("stay_id", sort=False)
    other = "hr" if outcome == "map" else "map"
    if design == "lagged":
        hist = _lags(df, [outcome, other], 3)
        hist = hist.join(pd.DataFrame({f"{c}_l{k}": g[c].shift(k) for c in ("fluid_ml", "ne_dose") for k in (1, 2, 3)}))
        T = pd.DataFrame({"F": df["fluid_ml"] / F_UNIT, "D": (df["ne_dose"] - g["ne_dose"].shift(1)) / D_UNIT})
    else:
        hist = _lags(df, [outcome, other, "fluid_ml", "ne_dose"], 3)
        T = pd.DataFrame({"F": g["fluid_ml"].shift(-1) / F_UNIT, "D": (g["ne_dose"].shift(-1) - df["ne_dose"]) / D_UNIT})
    hist["lactate"] = df["lactate"]; hist["age"] = df["age"]; hist["hour"] = df["hour"]
    y = g[outcome].shift(-1)
    ok = y.notna() & hist[f"{outcome}_l0"].notna() & T.notna().all(axis=1)
    X, y, T, sid = hist[ok], y[ok].to_numpy(), T[ok].to_numpy(), df.loc[ok, "stay_id"].to_numpy()
    stays = np.unique(sid)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(stays, rng.integers(0, folds, len(stays))))
    f = np.array([fold_of[s] for s in sid])
    ry, rT = np.zeros_like(y), np.zeros_like(T)
    for k in range(folds):
        tr, te = f != k, f == k
        reg = lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=seed)
        ry[te] = y[te] - reg().fit(X[tr], y[tr]).predict(X[te])
        for j in range(T.shape[1]):
            rT[te, j] = T[te, j] - reg().fit(X[tr], T[tr, j]).predict(X[te])
    A = rT.T @ rT
    theta = np.linalg.solve(A, rT.T @ ry)
    resid = ry - rT @ theta
    meat = np.zeros((2, 2))
    order = np.argsort(sid, kind="stable")
    bounds = np.flatnonzero(np.r_[True, sid[order][1:] != sid[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):          # cluster-robust by stay
        idx = order[a:b]
        u = rT[idx].T @ resid[idx]
        meat += np.outer(u, u)
    Ainv = np.linalg.inv(A)
    return theta, Ainv @ meat @ Ainv


# --------------------------------------------------------------------------- engine

@dataclass
class HybridEngine:
    seed: int = 0
    horizon: int = 6
    effects: dict = field(default_factory=dict)        # outcome -> {"fluid": Effect, "ne": Effect}
    course: dict = field(default_factory=dict)          # outcome -> baseline course model
    resid_q90: dict = field(default_factory=dict)       # outcome -> conformal half-width per hour ahead
    propensity: dict = field(default_factory=dict)
    support: dict = field(default_factory=dict)

    # ---- fitting
    def fit(self, train: pd.DataFrame, calib: pd.DataFrame, log=print) -> "HybridEngine":
        train, calib = _prep(train), _prep(calib)
        for o in OUTCOMES:
            theta, cov = dml_effects(train, o, seed=self.seed)
            pr = PRIORS[o]
            self.effects[o] = {
                "fluid": Effect(float(theta[0]), float(np.sqrt(cov[0, 0])), *pr["fluid"]),
                "ne": Effect(float(theta[1]), float(np.sqrt(cov[1, 1])), *pr["ne"])}
            log(f"{o}: fluid {self.effects[o]['fluid'].as_dict()}")
            log(f"{o}: NE    {self.effects[o]['ne'].as_dict()}")
            self.course[o] = self._fit_course(train, o)
        self._fit_propensity(train)
        self.support = {"ne_max": float(train.loc[train["ne_dose"] > 0, "ne_dose"].quantile(0.99)),
                        "ne_step_max": float(train.groupby("stay_id")["ne_dose"].diff().abs().quantile(0.999)),
                        "fluid_max": 1000.0}
        self.resid_q90 = self._conformal(calib)
        log(f"conformal 90% half-widths: { {o: np.round(v, 1).tolist() for o, v in self.resid_q90.items()} }")
        return self

    @staticmethod
    def _features(df: pd.DataFrame) -> pd.DataFrame:
        X = _lags(df, ["map", "hr"], 3)
        X = X.join(_lags(df, ["fluid_ml"], 3)).join(_lags(df, ["ne_dose"], 2))
        X["lactate"], X["age"], X["hour"] = df["lactate"], df["age"], df["hour"]
        return X

    def _fit_course(self, df, o):
        X = self._features(df)
        y = df.groupby("stay_id", sort=False)[o].shift(-1)
        ok = y.notna() & X[f"{o}_l0"].notna()
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=self.seed).fit(X[ok], y[ok])

    def _fit_propensity(self, df):
        X = self._features(df)
        g = df.groupby("stay_id", sort=False)
        nxt_ne, nxt_f = g["ne_dose"].shift(-1), g["fluid_ml"].shift(-1)
        ok = nxt_ne.notna()
        y_ne, y_bolus = (nxt_ne[ok] > 0).astype(int), (nxt_f[ok] >= 250).astype(int)
        self.propensity["ne_on"] = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(X[ok], y_ne)
        self.propensity["bolus"] = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(X[ok], y_bolus)
        off = ok & (df["ne_dose"] == 0)
        # Refusal thresholds relative to how common each action is: a plan is refused when this
        # patient's probability is under a tenth of the base rate ("almost never in patients like this").
        self.propensity["min_start"] = max(1e-3, float((nxt_ne[off] > 0).mean()) / 10)
        self.propensity["min_bolus"] = max(1e-3, float(y_bolus.mean()) / 10)

    # ---- trajectories
    def baseline(self, history: pd.DataFrame, hours: int) -> dict:
        """Course under usual care as practised in the training hospitals (their routine titration
        included): the reference that a plan is compared with."""
        h = _prep(history).copy()
        out = {o: [] for o in OUTCOMES}
        for _ in range(hours):
            X = self._features(h).iloc[[-1]]
            new = h.iloc[[-1]].copy()
            new["hour"] += 1
            new["fluid_ml"] = 0.0                     # dose column keeps the current dose
            for o in OUTCOMES:
                new[o] = float(self.course[o].predict(X)[0])
                out[o].append(new[o].iloc[0])
            h = pd.concat([h, new], ignore_index=True)
        return {o: np.array(v) for o, v in out.items()}

    def effect_path(self, o: str, ne_now: float, ne_path, fluid_path) -> tuple[np.ndarray, np.ndarray]:
        """Treatment-difference vs current care for each future hour: mean and sd."""
        e = self.effects[o]
        H = len(ne_path)
        dne = (np.asarray(ne_path, float) - ne_now) / D_UNIT
        fl = np.asarray(fluid_path, float) / F_UNIT
        exp = np.array([sum(FLUID_KERNEL[j] * fl[k - j] for j in range(len(FLUID_KERNEL)) if k - j >= 0)
                        for k in range(H)])
        mean = e["ne"].mean * dne + e["fluid"].mean * exp
        sd = np.sqrt((e["ne"].sd * dne) ** 2 + (e["fluid"].sd * exp) ** 2)
        return mean, sd

    def rollout(self, history: pd.DataFrame, fluid_plan, ne_plan) -> dict:
        base = self.baseline(history, len(fluid_plan))
        ne_now = float(_prep(history)["ne_dose"].iloc[-1])
        return {o: base[o] + self.effect_path(o, ne_now, ne_plan, fluid_plan)[0] for o in OUTCOMES}

    def _conformal(self, calib: pd.DataFrame, n_origins: int = 600) -> dict:
        """Absolute errors of factual forecasts (actual future treatments) on held-out stays."""
        rng = np.random.default_rng(self.seed)
        errs = {o: [[] for _ in range(self.horizon)] for o in OUTCOMES}
        stays = calib["stay_id"].unique()
        for sid in rng.choice(stays, min(n_origins, len(stays)), replace=False):
            d = calib[calib["stay_id"] == sid].reset_index(drop=True)
            if len(d) < self.horizon + 4:
                continue
            t = int(rng.integers(3, len(d) - self.horizon))
            fut = d.iloc[t + 1:t + 1 + self.horizon]
            # The baseline model describes usual care, including the titration clinicians actually
            # did, so it is calibrated against what happened without adding any effect term.
            pred = self.baseline(d.iloc[:t + 1], self.horizon)
            for o in OUTCOMES:
                for k in range(self.horizon):
                    if pd.notna(fut[o].iloc[k]):
                        errs[o][k].append(abs(pred[o][k] - fut[o].iloc[k]))
        return {o: np.array([np.quantile(e, min(1.0, 0.9 * (1 + 1 / max(len(e), 1)))) if e else np.nan
                             for e in errs[o]]) for o in OUTCOMES}

    # ---- guarded query
    def positivity(self, history: pd.DataFrame, fluid_ml: float, ne_target: float) -> dict:
        h = _prep(history)
        X = self._features(h).iloc[[-1]]
        p_ne = float(self.propensity["ne_on"].predict_proba(X)[0, 1])
        p_bolus = float(self.propensity["bolus"].predict_proba(X)[0, 1])
        ne_now = float(h["ne_dose"].iloc[-1])
        reasons = []
        if ne_target > self.support["ne_max"]:
            reasons.append(f"requested dose {ne_target:.2f} is above the doses seen in the training data "
                           f"(99th percentile {self.support['ne_max']:.2f} mcg/kg/min)")
        if abs(ne_target - ne_now) > max(self.support["ne_step_max"], 0.1):
            reasons.append(f"a change of {abs(ne_target - ne_now):.2f} mcg/kg/min at once is outside the range "
                           "where the published dose-response applies")
        if fluid_ml > self.support["fluid_max"]:
            reasons.append(f"more than {self.support['fluid_max']:.0f} mL in one hour is outside the data")
        if ne_target > 0 and ne_now == 0 and p_ne < self.propensity["min_start"]:
            reasons.append(f"patients like this one almost never start a vasopressor (probability {p_ne:.3f})")
        if fluid_ml >= 250 and p_bolus < self.propensity["min_bolus"]:
            reasons.append(f"patients like this one almost never receive a bolus (probability {p_bolus:.3f})")
        return {"ok": not reasons, "reasons": reasons, "p_start_vasopressor": p_ne, "p_bolus": p_bolus}

    def query(self, history: pd.DataFrame, fluid_ml: float, ne_target: float, hours: int | None = None) -> dict:
        """Plan (bolus `fluid_ml` in the next hour; NE held at `ne_target`) vs current care."""
        H = hours or self.horizon
        h = _prep(history)
        ne_now = float(h["ne_dose"].iloc[-1])
        pos = self.positivity(h, fluid_ml, ne_target)
        if not pos["ok"]:
            return {"answered": False, "reasons": pos["reasons"], "positivity": pos, "assumption": ASSUMPTION}
        base = self.baseline(h, H)
        traj = {}
        for o in OUTCOMES:
            m, sd = self.effect_path(o, ne_now, [ne_target] * H, [fluid_ml] + [0.0] * (H - 1))
            plan = base[o] + m
            half = np.sqrt(self.resid_q90[o][:H] ** 2 + (1.645 * sd) ** 2)
            base_half = self.resid_q90[o][:H]
            traj[o] = pd.DataFrame({"hour_ahead": np.arange(1, H + 1),
                                    "current_care": base[o], "current_low": base[o] - base_half,
                                    "current_high": base[o] + base_half,
                                    "plan": plan, "plan_low": plan - half, "plan_high": plan + half,
                                    "difference": m, "difference_low": m - 1.645 * sd,
                                    "difference_high": m + 1.645 * sd})
        return {"answered": True, "assumption": ASSUMPTION, "positivity": pos,
                "plan": {"fluid_ml_next_hour": fluid_ml, "ne_dose": ne_target, "ne_now": ne_now},
                "effects_used": {o: {k: v.as_dict() for k, v in e.items()} for o, e in self.effects.items()},
                "trajectories": traj}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Plausible-value filter (MAP 20-200 mmHg, HR 20-250 bpm), sorted by stay and hour."""
    d = df.sort_values(["stay_id", "hour"]).reset_index(drop=True).copy()
    d["map"] = d["map"].where(d["map"].between(20, 200))
    d["hr"] = d["hr"].where(d["hr"].between(20, 250))
    return d
