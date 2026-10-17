"""Phase 6: wearable data ingestion into Data Lake 1.

Input files (one per device export, CSV):
  samples.csv  stay_id, timestamp, signal, value     signal in gsr | actigraphy |
                                                      toe_temperature | perfusion_index
  beats.csv    stay_id, timestamp, rr_ms             beat-to-beat intervals (ECG or PPG)

Every value passes the same quality gate as bedside data; HRV windows with a
low signal-quality index are flagged `suspect`. Rows enter the lake as
`measured` with the device named in `inputs`. Two Layer 1 derivations become
possible with wearables (plan, Layer 1 table):
  central-to-toe gradient = core temperature - toe temperature; >= 7 degC = poor perfusion (Lima et al. 2002)
  perfusion index < 1.4 % = poor peripheral perfusion (Lima et al. 2002)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sdt import schema
from sdt.quality import gate
from sdt.wearable.hrv import hrv_windows

SIGNALS = {"gsr", "actigraphy", "toe_temperature", "perfusion_index"}


def wearable_specs(parameters: dict) -> dict:
    keys = SIGNALS | {"hrv_rmssd", "hrv_sdnn"}
    return {k: {**parameters["primary"][k], "itemids": {}} for k in keys}


def ingest(samples: pd.DataFrame, beats: pd.DataFrame, stays: pd.DataFrame, parameters: dict,
           device: str, version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (lake rows, gate audit)."""
    specs = wearable_specs(parameters)
    s = samples.rename(columns={"timestamp": "charttime", "signal": "param"})
    s = s[s["param"].isin(SIGNALS)].assign(itemid=pd.NA)
    h = hrv_windows(beats.rename(columns={"timestamp": "time"}))
    hrv_rows = []
    if len(h):
        for col, param in (("rmssd_ms", "hrv_rmssd"), ("sdnn_ms", "hrv_sdnn")):
            part = h.dropna(subset=[col])
            hrv_rows.append(pd.DataFrame({
                "stay_id": part["stay_id"], "param": param, "value": part[col], "itemid": pd.NA,
                # a window's value is known at the END of the window
                "charttime": part["window_start"] + pd.Timedelta(minutes=5), "sqi": part["sqi"],
                "hrv_flag": part["flag"]}))
    df = pd.concat([s, *hrv_rows], ignore_index=True)
    df["unit"] = df["param"].map({k: v["unit"] for k, v in specs.items()})
    audit = gate.run_gate(df, specs, stays)
    low_sqi = audit.get("hrv_flag", pd.Series(index=audit.index, dtype=object)) == "suspect"
    audit.loc[low_sqi & ~audit["flag"].isin(schema.REMOVED_FLAGS), ["flag", "qc_reason"]] = \
        [schema.SUSPECT, "HRV window signal quality below threshold"]
    lake = gate.to_lake_rows(audit, version)
    lake["inputs"] = f"device:{device}"
    return lake, audit


def perfusion_derivations(core_temp: pd.Series, toe_temp: pd.Series, perfusion_index: pd.Series) -> pd.DataFrame:
    """Hourly aligned series in, derived flags out."""
    grad = core_temp - toe_temp
    return pd.DataFrame({
        "central_toe_gradient": grad,
        "poor_perfusion_gradient": (grad >= 7).astype(float).where(grad.notna()),
        "poor_perfusion_pi": (perfusion_index < 1.4).astype(float).where(perfusion_index.notna()),
    })
