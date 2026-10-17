"""Phase 6: heart-rate variability from beat-to-beat intervals.

HRV needs beat-to-beat (RR / inter-beat) intervals from ECG or PPG, not 1-5 min
summaries (plan, Layer 1 table). Time-domain metrics per window:
  RMSSD  root mean square of successive differences (ms)
  SDNN   standard deviation of normal-to-normal intervals (ms)
  pNN50  share of successive differences > 50 ms (%)
Beat cleaning: intervals outside 300-2000 ms are dropped, as are intervals
differing > 20 % from the previous accepted one (ectopic beats / artifact).
The share of beats kept is the window's signal-quality index (SQI); windows
with SQI < `min_sqi` are flagged `suspect` and not used downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_rr(rr_ms: np.ndarray, max_change: float = 0.2) -> np.ndarray:
    """Boolean mask of accepted intervals."""
    rr = np.asarray(rr_ms, dtype=float)
    ok = (rr >= 300) & (rr <= 2000)
    keep = np.zeros(len(rr), bool)
    prev = np.nan
    for i, (v, good) in enumerate(zip(rr, ok)):
        if not good:
            continue
        if np.isnan(prev) or abs(v - prev) <= max_change * prev:
            keep[i] = True
            prev = v
    return keep


def hrv_windows(beats: pd.DataFrame, window: str = "5min", min_sqi: float = 0.8,
                min_beats: int = 60) -> pd.DataFrame:
    """beats: columns stay_id, time (timestamp of beat), rr_ms. Returns one row per
    (stay_id, window) with RMSSD, SDNN, pNN50, SQI and a quality flag."""
    rows = []
    for sid, g in beats.sort_values("time").groupby("stay_id"):
        for start, w in g.groupby(pd.Grouper(key="time", freq=window)):
            if len(w) == 0:
                continue
            keep = clean_rr(w["rr_ms"].to_numpy())
            rr = w["rr_ms"].to_numpy()[keep]
            sqi = keep.mean()
            d = np.diff(rr)
            # successive differences only between consecutive accepted beats
            consecutive = np.diff(np.flatnonzero(keep)) == 1
            d = d[consecutive] if len(d) else d
            enough = len(rr) >= min_beats and len(d) > 1
            rows.append({
                "stay_id": sid, "window_start": start, "n_beats": len(w), "sqi": float(sqi),
                "rmssd_ms": float(np.sqrt(np.mean(d ** 2))) if enough else np.nan,
                "sdnn_ms": float(np.std(rr, ddof=1)) if enough else np.nan,
                "pnn50_pct": float(100 * np.mean(np.abs(d) > 50)) if enough else np.nan,
                "flag": "ok" if (sqi >= min_sqi and enough) else "suspect",
            })
    return pd.DataFrame(rows)
