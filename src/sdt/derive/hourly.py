"""Causal hourly patient timeline.

Hour h of a stay covers (intime + h, intime + h + 1]. Every value attached to
hour h was observed no later than the END of hour h, so a model reading hour h
never sees the future. Pre-ICU hours (negative h) are built only so that lab
values drawn before admission can be carried into the first ICU hours; they
are dropped from the output.

For each parameter p the timeline holds:
  p          last value, carried forward at most `max_age_h` hours (else NaN)
  p__min     lowest value observed within the hour   (for SOFA "worst")
  p__max     highest value observed within the hour
  p__mask    1 if p was measured within the hour      (GRU-D mask)
  p__age     hours since p was last measured          (GRU-D time interval)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sdt import schema

PRE_ICU_HOURS = 24


def hour_index(times: pd.Series, intime: pd.Series) -> pd.Series:
    """Hour bin whose end is the first hour boundary at or after `times`."""
    hours = (times - intime).dt.total_seconds() / 3600.0
    return np.ceil(hours).astype("Int64") - 1


def stay_grid(stays: pd.DataFrame, pre_hours: int = PRE_ICU_HOURS) -> pd.DataFrame:
    n = np.ceil(stays["los_h"].to_numpy()).astype(int)
    grid = pd.DataFrame({
        "stay_id": np.repeat(stays["stay_id"].to_numpy(), n + pre_hours),
        "hour": np.concatenate([np.arange(-pre_hours, k) for k in n]),
    })
    intime = grid["stay_id"].map(stays.set_index("stay_id")["intime"])
    grid["hour_end"] = intime + pd.to_timedelta(grid["hour"] + 1, unit="h")
    return grid


def build_hourly(lake: pd.DataFrame, stays: pd.DataFrame, specs: dict) -> pd.DataFrame:
    usable = lake[lake["flag"].isin(schema.USABLE_FLAGS)]
    intime = usable["stay_id"].map(stays.set_index("stay_id")["intime"])
    obs = usable.assign(hour=hour_index(usable["charttime"], intime))
    obs = obs[obs["hour"] >= -PRE_ICU_HOURS].sort_values("charttime")

    agg = obs.groupby(["stay_id", "param", "hour"])["value"].agg(["last", "min", "max"])
    grid = stay_grid(stays).set_index(["stay_id", "hour"])

    out = {}
    for name, spec in specs.items():
        if name not in agg.index.get_level_values("param"):
            sub = pd.DataFrame(index=grid.index, columns=["last", "min", "max"], dtype=float)
        else:
            sub = agg.xs(name, level="param").reindex(grid.index)
        mask = sub["last"].notna().astype("int8")
        last_seen = _hours_since(mask)
        limit = int(spec.get("max_age_h", 24))
        carried = sub["last"].groupby(level="stay_id").ffill()
        carried = carried.where(last_seen <= limit)
        out[name] = carried
        out[f"{name}__min"] = sub["min"]
        out[f"{name}__max"] = sub["max"]
        out[f"{name}__mask"] = mask
        out[f"{name}__age"] = last_seen
    return pd.concat({"hour_end": grid["hour_end"], **out}, axis=1)


def _hours_since(mask: pd.Series) -> pd.Series:
    """Hours since the last hour with mask == 1, per stay (NaN before the first)."""
    hours = pd.Series(mask.index.get_level_values("hour"), index=mask.index, dtype=float)
    last = hours.where(mask == 1).groupby(level="stay_id").ffill()
    return hours - last


def icu_hours(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.index.get_level_values("hour") >= 0]


def rolling_hours(s: pd.Series, hours: int, fn: str) -> pd.Series:
    """Per-stay trailing window over the last `hours` hourly rows (inclusive)."""
    r = s.groupby(level="stay_id").rolling(hours, min_periods=1)
    return getattr(r, fn)().droplevel(0)
