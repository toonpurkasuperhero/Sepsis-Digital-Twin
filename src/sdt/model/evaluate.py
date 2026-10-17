"""Phase 4 evaluation (plan: "Datasets, evaluation and validation").

All hour-level metrics use eligible hours only (before onset). Confidence
intervals come from a patient-level bootstrap, because hours of one patient
are not independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def flatten(stays, probs: list[np.ndarray], k: int) -> pd.DataFrame:
    """One row per eligible hour: stay, hour, y, p (task k: 0 = 6 h, 1 = 12 h)."""
    parts = []
    for s, p in zip(stays, probs):
        e = s.elig == 1
        h = np.arange(len(s.y))[e]
        parts.append(pd.DataFrame({"stay_id": s.stay_id, "hour": h, "y": s.y[e, k], "p": p[e, k],
                                   "group": s.group}))
    return pd.concat(parts, ignore_index=True)


def calibration(y, p) -> tuple[float, float]:
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    lr = LogisticRegression(C=1e6).fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def bootstrap(df: pd.DataFrame, fn, n: int = 200, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = {sid: idx for sid, idx in df.groupby("stay_id").indices.items()}
    keys = np.array(list(groups))
    vals = []
    for _ in range(n):
        pick = rng.choice(keys, len(keys))
        idx = np.concatenate([groups[k] for k in pick])
        d = df.iloc[idx]
        if d["y"].nunique() == 2:
            vals.append(fn(d["y"], d["p"]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def threshold_for_sensitivity(y, p, target: float = 0.8) -> float:
    pos = np.sort(p[y == 1])
    return float(pos[int(np.floor((1 - target) * len(pos)))])


def alert_metrics(df: pd.DataFrame, thr: float, onsets: pd.Series) -> dict:
    alert = df["p"] >= thr
    y = df["y"] == 1
    tp, fp = (alert & y).sum(), (alert & ~y).sum()
    out = {"threshold": thr, "sensitivity": float(tp / max(y.sum(), 1)),
           "specificity": float(((~alert) & ~y).sum() / max((~y).sum(), 1)),
           "ppv": float(tp / max(tp + fp, 1)),
           "alerts_per_patient_day": float(alert.sum() / (len(df) / 24))}
    # Lead time: septic stays with an onset inside the record; first alert in the 12 h before onset.
    d = df.assign(alert=alert, onset=df["stay_id"].map(onsets))
    win = d[d["onset"].notna() & (d["hour"] >= d["onset"] - 12) & (d["hour"] < d["onset"])]
    first = win[win["alert"]].groupby("stay_id")["hour"].min()
    n_sep = win["stay_id"].nunique()
    lead = (win.drop_duplicates("stay_id").set_index("stay_id")["onset"] - first).dropna()
    out.update(septic_stays_detected_pct=float(100 * len(first) / max(n_sep, 1)),
               median_lead_time_h=float(lead.median()) if len(lead) else np.nan)
    # Decision-curve net benefit at this threshold.
    n = len(df)
    out["net_benefit"] = float(tp / n - fp / n * thr / (1 - thr))
    out["net_benefit_treat_all"] = float(y.mean() - (1 - y.mean()) * thr / (1 - thr))
    return out


def full_report(df: pd.DataFrame, iso: IsotonicRegression, thr: float, onsets: pd.Series,
                n_boot: int = 200) -> dict:
    y, p_raw = df["y"].to_numpy(), df["p"].to_numpy()
    p = iso.predict(p_raw)
    cal = df.assign(p=p)
    slope, intercept = calibration(y, p)
    rep = {"hours": int(len(df)), "stays": int(df["stay_id"].nunique()), "positive_hours": int(y.sum()),
           "auroc": float(roc_auc_score(y, p_raw)), "auroc_ci95": bootstrap(df, roc_auc_score, n_boot),
           "auprc": float(average_precision_score(y, p_raw)),
           "auprc_ci95": bootstrap(df, average_precision_score, n_boot),
           "prevalence": float(y.mean()), "brier_calibrated": float(brier_score_loss(y, p)),
           "calibration_slope": slope, "calibration_intercept": intercept}
    rep.update(alert_metrics(cal, thr, onsets))
    return rep


def subgroup_auroc(df: pd.DataFrame, labels: pd.Series, name: str) -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(df["stay_id"].map(labels)):
        if d["y"].nunique() == 2 and d["y"].sum() >= 20:
            rows.append({"subgroup": name, "value": g, "stays": d["stay_id"].nunique(),
                         "auroc": roc_auc_score(d["y"], d["p"]), "auprc": average_precision_score(d["y"], d["p"])})
    return pd.DataFrame(rows)


def fit_isotonic(df: pd.DataFrame) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(df["p"], df["y"])
