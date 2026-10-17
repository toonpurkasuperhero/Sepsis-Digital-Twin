"""Turns the hourly timeline into per-stay sequences for Phases 3-4.

Inputs are rebuilt from OBSERVED values only (value where `__mask == 1`), so
carry-forward, masks and time-since-last can be recomputed after values are
hidden for mask-and-recover tests, exactly as the live system would see them.

Streams:
  E1  primary vitals: observed matrix X1 [T, P1] + Layer 1 derived D1 [T, Q1]
  E2  labs as irregular tokens (hour, param, value) + Layer 2 derived D2 [T, Q2]
  E3  static baseline vector
  ENV environment inputs (masked; capped effect in the model)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

AGE_CAP_H = 48.0


@dataclass
class Schema:
    primary: list[str]
    secondary: list[str]
    derived1: list[str]
    derived2: list[str]
    static: list[str]
    env: list[str]
    max_age: dict[str, float] = field(default_factory=dict)


@dataclass
class Stay:
    stay_id: int
    x1: np.ndarray          # [T, P1] observed (NaN = not measured)
    d1: np.ndarray          # [T, Q1]
    tok_t: np.ndarray       # [N] hour of each lab token
    tok_p: np.ndarray       # [N] param index
    tok_v: np.ndarray       # [N] raw value
    x2: np.ndarray          # [T, P2] observed labs (for E2 pre-training targets)
    d2: np.ndarray          # [T, Q2]
    static: np.ndarray      # [S]
    env: np.ndarray         # [E]
    y: np.ndarray           # [T, 2] targets 6 h, 12 h
    elig: np.ndarray        # [T]
    group: str = ""


def read_lake(root: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(root / "lakes" / name)


def make_schema(timeline: pd.DataFrame, primary: list[str], secondary: list[str],
                static_cols: list[str], env: list[str], max_age: dict) -> Schema:
    measured = set(primary) | set(secondary)
    base = {c.split("__")[0] for c in timeline.columns if c.endswith("__mask")}
    derived = [c for c in timeline.columns if "__" not in c and c not in measured
               and c not in ("hour_end", "eligible_sepsis") and not c.startswith("y_")
               and pd.api.types.is_numeric_dtype(timeline[c]) and c not in base
               and timeline[c].nunique(dropna=True) > 1]          # constant columns carry nothing
    d1_names = {"gcs", "gcs_unable", "fio2_eff", "fio2_eff__unc", "shock_index", "pulse_pressure", "map_any",
                "sf_ratio", "pf_estimate", "rox", "qsofa", "news2"}
    d1 = [c for c in derived if c in d1_names]
    d2 = [c for c in derived if c not in d1_names and not timeline[c].isna().all()]
    prim = [p for p in primary if timeline[f"{p}__mask"].sum() > 0]
    sec = [p for p in secondary if timeline[f"{p}__mask"].sum() > 0]
    return Schema(prim, sec, d1, d2, static_cols, env, {p: max_age.get(p, 24) for p in prim + sec})


def build_stays(timeline: pd.DataFrame, static: pd.DataFrame, sc: Schema,
                group_col: str | None = None) -> list[Stay]:
    tl = timeline.sort_index()
    obs1 = tl[sc.primary].where(tl[[f"{p}__mask" for p in sc.primary]].to_numpy() == 1)
    obs2 = tl[sc.secondary].where(tl[[f"{p}__mask" for p in sc.secondary]].to_numpy() == 1)
    D1, D2 = tl[sc.derived1].to_numpy(np.float32), tl[sc.derived2].to_numpy(np.float32)
    X1, X2 = obs1.to_numpy(np.float32), obs2.to_numpy(np.float32)
    Y = tl[["y_sepsis_6h", "y_sepsis_12h"]].to_numpy(np.float32)
    E = tl["eligible_sepsis"].to_numpy(np.float32)
    S = static[sc.static].astype(float).to_numpy(np.float32)
    ENV = static[sc.env].astype(float).to_numpy(np.float32) if sc.env else np.zeros((len(static), 0), np.float32)
    srow = {sid: i for i, sid in enumerate(static.index)}

    ids = tl.index.get_level_values("stay_id").to_numpy()
    bounds = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    stays = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sid = int(ids[a])
        x2 = X2[a:b]
        t, p = np.nonzero(~np.isnan(x2))
        stays.append(Stay(sid, X1[a:b], D1[a:b], t.astype(np.int32), p.astype(np.int32), x2[t, p],
                          x2, D2[a:b], S[srow[sid]], ENV[srow[sid]], Y[a:b], E[a:b],
                          str(static.loc[sid, group_col]) if group_col else ""))
    return stays


# ------------------------------------------------------------ normalisation

@dataclass
class Normalizer:
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]

    @classmethod
    def fit(cls, stays: list[Stay]) -> "Normalizer":
        def stats(arr):
            m = np.nanmean(arr, axis=0)
            s = np.nanstd(arr, axis=0)
            return np.nan_to_num(m), np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
        out_m, out_s = {}, {}
        for k in ("x1", "d1", "x2", "d2"):
            out_m[k], out_s[k] = stats(np.concatenate([getattr(s, k) for s in stays]))
        out_m["static"], out_s["static"] = stats(np.stack([s.static for s in stays]))
        return cls(out_m, out_s)

    def __call__(self, key: str, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean[key]) / self.std[key]


# ------------------------------------------------------------ causal features

def locf_features(x_obs: np.ndarray, max_age: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From observed values [T, P]: carried value (NaN past max age), mask, hours since last."""
    T, P = x_obs.shape
    mask = ~np.isnan(x_obs)
    idx = np.where(mask, np.arange(T)[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    age = np.where(idx >= 0, np.arange(T)[:, None] - idx, np.inf).astype(np.float32)
    val = np.take_along_axis(np.nan_to_num(x_obs), np.maximum(idx, 0), axis=0)
    val = np.where((idx >= 0) & (age <= max_age[None, :]), val, np.nan).astype(np.float32)
    return val, mask.astype(np.float32), age
