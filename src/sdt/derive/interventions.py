"""Stream 4: interventions on the hourly timeline.

Columns produced per (stay_id, hour):
  vaso_<drug>             highest infusion rate running in the hour (mcg/kg/min;
                          vasopressin in units/min)
  ne_equivalent           norepinephrine-equivalent dose (Goradia et al. 2021)
  iv_fluid_ml             crystalloid + albumin volume given in the hour
  vent_invasive / vent_noninvasive   1 if running during the hour
  rx_<class>              1 if a drug of that class was active in the hour
  antibiotic_start        1 if an antibiotic order started in the hour
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from sdt.derive.hourly import stay_grid


def _overlap_hours(events: pd.DataFrame, stays: pd.DataFrame, start="starttime", end="endtime") -> pd.DataFrame:
    """Explodes interval events into one row per ICU hour they overlap, with the
    overlapping fraction of the event's duration (`frac`)."""
    if events.empty:
        return events.assign(hour=pd.Series(dtype=int), frac=pd.Series(dtype=float))
    ev = events.dropna(subset=[start]).copy()
    ev[end] = ev[end].fillna(ev[start])
    intime = ev["stay_id"].map(stays.set_index("stay_id")["intime"])
    s = (ev[start] - intime).dt.total_seconds() / 3600
    e = (ev[end] - intime).dt.total_seconds() / 3600
    e = np.maximum(e, s + 1e-6)
    first, last = np.floor(s).astype(int), np.ceil(e).astype(int) - 1
    reps = (last - first + 1).clip(lower=1).to_numpy()
    out = ev.loc[ev.index.repeat(reps)].copy()
    offs = np.concatenate([np.arange(r) for r in reps])
    out["hour"] = np.repeat(first.to_numpy(), reps) + offs
    lo = np.maximum(out["hour"].to_numpy(), np.repeat(s.to_numpy(), reps))
    hi = np.minimum(out["hour"].to_numpy() + 1, np.repeat(e.to_numpy(), reps))
    out["frac"] = np.clip(hi - lo, 0, None) / np.repeat((e - s).to_numpy(), reps)
    return out


def vasopressors(inputs: pd.DataFrame, stays: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    by_id = {v["itemid"]: (k, v) for k, v in cfg["vasopressors"].items()}
    ev = inputs[inputs["itemid"].isin(by_id) & inputs["rate"].notna()].copy()
    ev["drug"] = ev["itemid"].map(lambda i: by_id[i][0])
    unit = ev["rateuom"].fillna("").str.lower()
    w = ev["patientweight"].where(ev["patientweight"] > 0)
    # Normalise to mcg/kg/min (vasopressin: units/min).
    ev.loc[unit == "mcg/min", "rate"] = ev["rate"] / w
    ev.loc[unit == "mg/kg/min", "rate"] = ev["rate"] * 1000
    ev.loc[unit == "units/hour", "rate"] = ev["rate"] / 60
    ev = ev.dropna(subset=["rate"])
    hours = _overlap_hours(ev, stays)
    hours = hours[hours["frac"] > 0]
    wide = hours.pivot_table(index=["stay_id", "hour"], columns="drug", values="rate", aggfunc="max")
    wide.columns = [f"vaso_{c}" for c in wide.columns]
    for k in cfg["vasopressors"]:
        if f"vaso_{k}" not in wide:
            wide[f"vaso_{k}"] = np.nan
    factors = {f"vaso_{k}": v["ne_factor"] for k, v in cfg["vasopressors"].items()}
    wide["ne_equivalent"] = sum(wide[c].fillna(0) * f for c, f in factors.items())
    return wide


def iv_fluids(inputs: pd.DataFrame, stays: pd.DataFrame, cfg: dict) -> pd.Series:
    ids = set(cfg["iv_fluids"]["itemids"])
    ev = inputs[inputs["itemid"].isin(ids) & (inputs["amountuom"].str.lower() == "ml")]
    ev = ev[ev["amount"] > 0]
    hours = _overlap_hours(ev, stays)
    hours["ml"] = hours["amount"] * hours["frac"]
    return hours.groupby(["stay_id", "hour"])["ml"].sum().rename("iv_fluid_ml")


def ventilation(procs: pd.DataFrame, stays: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ids = {v["itemid"]: f"vent_{k}" for k, v in cfg["ventilation"].items()}
    ev = procs[procs["itemid"].isin(ids)].assign(kind=lambda d: d["itemid"].map(ids))
    hours = _overlap_hours(ev, stays)
    wide = hours.assign(on=1).pivot_table(index=["stay_id", "hour"], columns="kind", values="on", aggfunc="max")
    for c in ids.values():
        if c not in wide:
            wide[c] = np.nan
    return wide


def classify_drugs(rx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Adds `drug_class` to prescriptions using whole-word name matching."""
    rx = rx[~rx["route"].fillna("").str.upper().isin(cfg["excluded_routes"])]
    excl = "|".join(map(re.escape, cfg["excluded_drug_words"]))
    rx = rx[~rx["drug"].str.contains(excl, regex=True, na=False)]
    parts = []
    for cls, names in cfg["drug_classes"].items():
        pat = r"\b(?:" + "|".join(map(re.escape, names)) + r")\b"
        hit = rx[rx["drug"].str.contains(pat, regex=True, na=False)]
        if cls == "anticoagulant":
            bad = "|".join(map(re.escape, cfg["anticoagulant_exclude_words"]))
            hit = hit[~hit["drug"].str.contains(bad, regex=True, na=False)]
        parts.append(hit.assign(drug_class=cls))
    return pd.concat(parts, ignore_index=True) if parts else rx.assign(drug_class=[])


def drug_hours(rx_classed: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    ev = rx_classed.merge(stays[["hadm_id", "stay_id", "intime", "outtime"]], on="hadm_id")
    ev = ev[(ev["starttime"] <= ev["outtime"]) & (ev["stoptime"].fillna(ev["starttime"]) >= ev["intime"] - pd.Timedelta(hours=24))]
    hours = _overlap_hours(ev, stays, start="starttime", end="stoptime")
    wide = hours.assign(on=1).pivot_table(index=["stay_id", "hour"], columns="drug_class", values="on", aggfunc="max")
    wide.columns = [f"rx_{c}" for c in wide.columns]
    abx = ev[ev["drug_class"] == "antibiotic"]
    start_h = np.floor((abx["starttime"] - abx["intime"]).dt.total_seconds() / 3600).astype(int)
    starts = pd.Series(1, index=pd.MultiIndex.from_arrays([abx["stay_id"], start_h], names=["stay_id", "hour"]))
    wide = wide.join(starts.groupby(level=[0, 1]).max().rename("antibiotic_start"), how="outer")
    return wide


def build_interventions(src_inputs, src_procs, rx, stays, cfg) -> pd.DataFrame:
    grid = stay_grid(stays, pre_hours=0).set_index(["stay_id", "hour"])
    parts = [vasopressors(src_inputs, stays, cfg), iv_fluids(src_inputs, stays, cfg),
             ventilation(src_procs, stays, cfg), drug_hours(classify_drugs(rx, cfg), stays)]
    out = grid[[]].join(parts, how="left")
    rate_cols = [c for c in out.columns if c.startswith("vaso_")]
    flag_cols = [c for c in out.columns if c.startswith(("vent_", "rx_")) or c == "antibiotic_start"]
    out[rate_cols + ["ne_equivalent", "iv_fluid_ml"]] = out[rate_cols + ["ne_equivalent", "iv_fluid_ml"]].fillna(0.0)
    out[flag_cols] = out[flag_cols].fillna(0).astype("int8")
    out["iv_fluid_24h_ml"] = out["iv_fluid_ml"].groupby(level="stay_id").rolling(24, min_periods=1).sum().droplevel(0)
    return out
