"""Data quality gate: every measured value passes these checks before a lake.

Checks (numbering follows the plan's "Data quality gate" section):
  1 schema and units      2 duplicates and time window
  3 hard physiological limits (-> impossible, removed)
  4 waveform signal quality (needs raw ECG/PPG; not in MIMIC-IV tables)
  5 artifact classifier (needs expert-labelled alerts; not available yet)
  6 extreme values (kept) and transient spikes (-> suspect)
  7 cross-parameter consistency (SBP > MAP > DBP)

Nothing that is physiologically possible is deleted: extremes stay in and are
used everywhere, suspects stay in the lake but are kept out of derivations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sdt import schema

_CONVERSIONS = {
    "f_to_c": lambda v: (v - 32.0) * 5.0 / 9.0,
    "percent_to_fraction_if_gt1": lambda v: np.where(v > 1.0, v / 100.0, v),
}
# Itemid families that are charted together (same cuff or same arterial line).
_BP_FAMILIES = {
    "arterial": {220050, 220051, 220052, 225312},
    "noninvasive": {220179, 220180, 220181},
}


def map_and_convert(raw: pd.DataFrame, specs: dict) -> pd.DataFrame:
    """Check 1: attach parameter names and standard units; convert where needed."""
    itemid_to_param = {i: name for name, s in specs.items() for i in s["itemids"]}
    df = raw[raw["itemid"].isin(itemid_to_param)].copy()
    df["param"] = df["itemid"].map(itemid_to_param)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value", "charttime"])
    for name, s in specs.items():
        for itemid, fn in (s.get("convert") or {}).items():
            m = df["itemid"] == itemid
            df.loc[m, "value"] = _CONVERSIONS[fn](df.loc[m, "value"].to_numpy())
    df["unit"] = df["param"].map({n: s["unit"] for n, s in specs.items()})
    return df


def run_gate(df: pd.DataFrame, specs: dict, stays: pd.DataFrame,
             pre_icu_hours: dict[str, int] | None = None) -> pd.DataFrame:
    """Returns `df` with `flag` and `qc_reason` columns. Removed rows are kept
    in the returned frame (flagged) so the gate itself can be audited."""
    df = df.copy()
    df["flag"] = schema.OK
    df["qc_reason"] = ""

    # --- 2. time window: vitals inside the stay; labs may precede it
    win = stays.set_index("stay_id")[["intime", "outtime"]]
    intime = df["stay_id"].map(win["intime"])
    outtime = df["stay_id"].map(win["outtime"])
    lead = df["param"].map(pre_icu_hours or {}).fillna(0)
    early = df["charttime"] < intime - pd.to_timedelta(lead, unit="h")
    late = df["charttime"] > outtime + pd.Timedelta(hours=1)
    _flag(df, early | late, schema.OUT_OF_WINDOW, "outside ICU stay window")

    # --- 2. exact duplicates (same parameter, time and value)
    dup = df.duplicated(["stay_id", "param", "charttime", "value"], keep="first")
    _flag(df, dup, schema.DUPLICATE, "duplicate row")

    for name, s in specs.items():
        m = df["param"] == name
        v = df.loc[m, "value"]
        # --- 3. hard limits
        lo, hi = s.get("hard_limits", (-np.inf, np.inf))
        _flag(df, m & ((df["value"] < lo) | (df["value"] > hi)), schema.IMPOSSIBLE,
              f"outside hard limits [{lo}, {hi}]")
        # --- 6. extreme but possible: kept
        if "extreme" in s:
            elo, ehi = s["extreme"]
            ext = m & (df["flag"] == schema.OK) & ((df["value"] < elo) | (df["value"] > ehi))
            _flag(df, ext, schema.EXTREME, f"outside typical range [{elo}, {ehi}]")
        # --- 6. transient spikes
        if "max_jump" in s and v.size > 2:
            _flag_spikes(df, m, s["max_jump"], s.get("jump_window_min", 5))

    # --- 7. cross-parameter consistency
    _flag_bp_consistency(df)
    return df


def _flag(df: pd.DataFrame, mask: pd.Series, flag: str, reason: str) -> None:
    # A stronger verdict (removed) is never overwritten by a weaker one.
    if flag not in schema.REMOVED_FLAGS:
        mask = mask & ~df["flag"].isin(schema.REMOVED_FLAGS)
    df.loc[mask, "flag"] = flag
    df.loc[mask, "qc_reason"] = reason


def _flag_spikes(df: pd.DataFrame, m: pd.Series, max_jump: float, window_min: float) -> None:
    """A value that jumps away from BOTH neighbours by more than `max_jump`
    within `window_min`, in the same direction, is a transient spike."""
    sub = df[m & df["flag"].isin(schema.USABLE_FLAGS)].sort_values(["stay_id", "charttime"])
    g = sub.groupby("stay_id")
    d_prev = sub["value"] - g["value"].shift(1)
    d_next = sub["value"] - g["value"].shift(-1)
    t_prev = (sub["charttime"] - g["charttime"].shift(1)).dt.total_seconds() / 60
    t_next = (g["charttime"].shift(-1) - sub["charttime"]).dt.total_seconds() / 60
    spike = ((d_prev.abs() > max_jump) & (d_next.abs() > max_jump)
             & (np.sign(d_prev) == np.sign(d_next))
             & (t_prev <= window_min) & (t_next <= window_min))
    idx = spike[spike].index
    df.loc[idx, "flag"] = schema.SUSPECT
    df.loc[idx, "qc_reason"] = f"transient spike > {max_jump} within {window_min} min"


def _flag_bp_consistency(df: pd.DataFrame) -> None:
    bp = df[df["param"].isin(["sbp", "dbp", "map"]) & df["flag"].isin(schema.USABLE_FLAGS)].copy()
    if bp.empty:
        return
    fam = {i: f for f, ids in _BP_FAMILIES.items() for i in ids}
    bp["family"] = bp["itemid"].map(fam).fillna("single_source")   # sources without itemids
    wide = bp.pivot_table(index=["stay_id", "charttime", "family"], columns="param",
                          values="value", aggfunc="first")
    for col in ("sbp", "dbp", "map"):
        if col not in wide:
            wide[col] = np.nan
    bad_pair = wide["sbp"] <= wide["dbp"]
    bad_map = (wide["map"] < wide["dbp"]) | (wide["map"] > wide["sbp"])
    keys = bp.set_index(["stay_id", "charttime", "family"]).index
    pair_hit = keys.isin(wide.index[bad_pair.fillna(False)])
    map_hit = keys.isin(wide.index[bad_map.fillna(False)])
    rows = bp.index
    pair_rows = rows[pair_hit & bp["param"].isin(["sbp", "dbp"]).to_numpy()]
    map_rows = rows[map_hit & (bp["param"] == "map").to_numpy()]
    df.loc[pair_rows, ["flag", "qc_reason"]] = [schema.SUSPECT, "SBP <= DBP at same time"]
    df.loc[map_rows, ["flag", "qc_reason"]] = [schema.SUSPECT, "MAP outside DBP..SBP at same time"]


def gate_report(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per parameter and flag, plus share removed."""
    rep = df.pivot_table(index="param", columns="flag", values="value", aggfunc="size", fill_value=0)
    rep["total"] = rep.sum(axis=1)
    removed = [c for c in rep.columns if c in schema.REMOVED_FLAGS]
    rep["pct_removed"] = (rep[removed].sum(axis=1) / rep["total"] * 100).round(3) if removed else 0.0
    rep["sqi_check"] = "not evaluated: no raw waveforms in source"
    rep["artifact_model"] = "not evaluated: needs expert-labelled alerts"
    return rep.reset_index()


def to_lake_rows(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """Measured values that survived the gate, in lake format."""
    keep = df[~df["flag"].isin(schema.REMOVED_FLAGS)].copy()
    keep["source"] = schema.MEASURED
    keep["uncertainty"] = np.nan
    keep["inputs"] = ""
    keep["pipeline_version"] = version
    return keep[schema.LAKE_COLUMNS].reset_index(drop=True)
