"""Sepsis-3 and septic-shock labels on the hourly timeline.

Suspected infection (Seymour et al. 2016, JAMA; mimic-code `suspicion_of_infection`):
  antibiotic, then a culture within 72 h  -> suspicion time = antibiotic time
  culture, then an antibiotic within 24 h -> suspicion time = culture time
Sepsis-3 (Singer et al. 2016): suspected infection AND SOFA >= 2 (baseline
assumed 0) at some hour from 48 h before to 24 h after the suspicion time.
Onset time = min(suspicion time, first such SOFA hour), the convention of the
PhysioNet/CinC 2019 challenge (Reyna et al. 2020).
Labels use `sofa_label`, i.e. measured values only.

Septic shock (Singer et al. 2016): sepsis + vasopressor running + lactate
> 2 mmol/L. Whether fluids were "adequate" cannot be checked reliably in EHR
data, so this is the common EHR approximation (no fluid criterion).

Prediction targets at hour h (horizon H):
  y_sepsis_{H}h = 1 if onset falls in hours (h, h + H]
  eligible_sepsis = hour is before onset (after onset the task is undefined)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def suspicion_times(abx: pd.DataFrame, cultures: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """abx: hadm_id, starttime ; cultures: hadm_id, culture_time.
    Returns hadm_id, suspicion_time, antibiotic_time, culture_time."""
    a = abx[["hadm_id", "starttime"]].dropna().drop_duplicates()
    c = cultures[["hadm_id", "culture_time"]].dropna().drop_duplicates()
    pairs = a.merge(c, on="hadm_id")
    dt = (pairs["culture_time"] - pairs["starttime"]).dt.total_seconds() / 3600
    abx_first = (dt >= 0) & (dt <= cfg["abx_then_culture_h"])
    cult_first = (dt < 0) & (-dt <= cfg["culture_then_abx_h"])
    pairs = pairs[abx_first | cult_first].copy()
    pairs["suspicion_time"] = np.where(abx_first[abx_first | cult_first], pairs["starttime"], pairs["culture_time"])
    pairs["suspicion_time"] = pd.to_datetime(pairs["suspicion_time"])
    return pairs.rename(columns={"starttime": "antibiotic_time"}).drop_duplicates(["hadm_id", "suspicion_time"])


def sepsis_onsets(sofa_label: pd.Series, hour_end: pd.Series, stays: pd.DataFrame,
                  susp: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per stay: suspicion time, SOFA time, onset time and onset hour."""
    st = stays.set_index("stay_id")
    frame = pd.DataFrame({"sofa": sofa_label, "hour_end": hour_end}).reset_index()
    frame = frame[frame["sofa"] >= cfg["sofa_threshold"] + cfg["baseline_sofa"]]
    frame["hadm_id"] = frame["stay_id"].map(st["hadm_id"])
    cand = frame.merge(susp[["hadm_id", "suspicion_time"]], on="hadm_id")
    lo = cand["suspicion_time"] - pd.Timedelta(hours=cfg["sofa_window_before_h"])
    hi = cand["suspicion_time"] + pd.Timedelta(hours=cfg["sofa_window_after_h"])
    cand = cand[(cand["hour_end"] >= lo) & (cand["hour_end"] <= hi)]
    first = (cand.sort_values("hour_end")
             .groupby(["stay_id", "suspicion_time"], as_index=False)["hour_end"].first()
             .rename(columns={"hour_end": "sofa_time"}))
    first["onset_time"] = first[["suspicion_time", "sofa_time"]].min(axis=1)
    best = first.sort_values("onset_time").groupby("stay_id").first()
    out = pd.DataFrame(index=st.index)
    out = out.join(best[["suspicion_time", "sofa_time", "onset_time"]])
    hours = (out["onset_time"] - st["intime"]).dt.total_seconds() / 3600
    out["onset_hour"] = np.ceil(hours) - 1          # same binning as the timeline
    out["sepsis3"] = out["onset_time"].notna().astype("int8")
    out["sepsis_at_admission"] = (out["onset_hour"] < 0).astype("int8")
    return out


def hourly_labels(index: pd.MultiIndex, onsets: pd.DataFrame, shock_hour: pd.Series,
                  horizons: list[int]) -> pd.DataFrame:
    sid = index.get_level_values("stay_id")
    hour = np.asarray(index.get_level_values("hour"), dtype=float)
    onset = np.asarray(sid.map(onsets["onset_hour"]), dtype=float)
    shock = np.asarray(sid.map(shock_hour), dtype=float)
    out = pd.DataFrame(index=index)
    out["eligible_sepsis"] = (np.isnan(onset) | (hour < onset)).astype("int8")
    out["eligible_shock"] = (np.isnan(shock) | (hour < shock)).astype("int8")
    for H in horizons:
        out[f"y_sepsis_{H}h"] = ((onset > hour) & (onset <= hour + H)).astype("int8")
        out[f"y_shock_{H}h"] = ((shock > hour) & (shock <= hour + H)).astype("int8")
    return out


def shock_onset_hour(h: pd.DataFrame, onsets: pd.DataFrame, cfg: dict) -> pd.Series:
    """First hour with any vasopressor running and lactate > threshold, in a
    sepsis-positive stay, no earlier than 24 h before sepsis onset."""
    pressor_cols = ["vaso_norepinephrine", "vaso_epinephrine", "vaso_dopamine",
                    "vaso_phenylephrine", "vaso_vasopressin"]
    pressor = (h[pressor_cols] > 0).any(axis=1)
    lactate = h["lactate"] > cfg["shock_lactate_mmol"]
    sid = h.index.get_level_values("stay_id")
    hour = pd.Series(h.index.get_level_values("hour"), index=h.index, dtype=float)
    onset = pd.Series(np.asarray(sid.map(onsets["onset_hour"]), dtype=float), index=h.index)
    ok = pressor & lactate & onset.notna() & (hour >= onset - 24)
    return hour[ok].groupby(level="stay_id").min().rename("shock_hour")
