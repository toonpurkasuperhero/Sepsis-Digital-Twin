"""Layer 1: parameters derived from primary vitals (plan, "Layer 1" table A).

All functions take the hourly timeline (see `hourly.py`) joined with the
intervention stream and return new columns. Columns whose source can vary
row-by-row (FiO2, MAP) get a companion `<name>__source` column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sdt import schema


def pao2_from_spo2(spo2_pct) -> np.ndarray:
    """Non-linear SpO2 -> PaO2 (Ellis inversion of the Severinghaus curve), the
    method recommended by Brown et al. 2016 (Chest) and validated prospectively
    by Brown et al. 2017 (Crit Care Med). Only valid for SpO2 <= 96 %."""
    s = np.asarray(spo2_pct, dtype=float) / 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        a = 11700.0 / (1.0 / s - 1.0)
        b = np.sqrt(50.0 ** 3 + a ** 2)
        return np.cbrt(a + b) + np.cbrt(a - b)


def gcs_total(h: pd.DataFrame) -> pd.DataFrame:
    """GCS sum. MIMIC codes an intubated verbal score as 0; following the
    MIT-LCP mimic-code convention the total is then set to 15 (unassessable),
    and `gcs_unable` records that so models can tell the difference."""
    unable = h["gcs_verbal"] == 0
    total = h["gcs_eye"] + h["gcs_verbal"] + h["gcs_motor"]
    total = total.where(~unable, 15.0)
    return pd.DataFrame({"gcs": total, "gcs_unable": unable.astype("int8")})


def fio2_effective(h: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Measured FiO2 when charted; otherwise a nasal-cannula estimate from O2
    flow (approximation, stored as a model estimate with uncertainty); otherwise
    room air, but only when no supplemental oxygen is documented at all."""
    per_lpm, max_lpm = cfg["nasal_cannula_fio2_per_lpm"], cfg["nasal_cannula_max_lpm"]
    measured = h["fio2"]
    flow = h["o2_flow"]
    est = (0.21 + per_lpm * flow).where((flow > 0) & (flow <= max_lpm))
    room_air = pd.Series(0.21, index=h.index).where(flow.isna() | (flow == 0))
    if not cfg.get("assume_room_air", True):      # sources where a missing FiO2 means "not charted"
        room_air[:] = np.nan
    fio2 = measured.fillna(est).fillna(room_air)
    src = np.select(
        [measured.notna(), est.notna(), room_air.notna()],
        [schema.MEASURED, schema.MODEL_ESTIMATE, schema.ASSUMED], default="")
    unc = np.select([measured.notna(), est.notna(), room_air.notna()], [0.0, 0.05, 0.02], default=np.nan)
    return pd.DataFrame({"fio2_eff": fio2, "fio2_eff__source": src, "fio2_eff__unc": unc})


def news2(h: pd.DataFrame) -> pd.Series:
    """National Early Warning Score 2 (Royal College of Physicians, 2017), SpO2 scale 1.
    Consciousness: GCS < 15 is scored as not Alert (3)."""
    rr, spo2, sbp, hr, t = h["resp_rate"], h["spo2"], h["sbp"], h["heart_rate"], h["temperature"]
    on_o2 = (h["fio2_eff"] > 0.21).astype(float)
    s = (
        _bands(rr, [8, 11, 20, 24], [3, 1, 0, 2, 3])
        + _bands(spo2, [91, 93, 95], [3, 2, 1, 0])
        + on_o2 * 2
        + _bands(sbp, [90, 100, 110, 219], [3, 2, 1, 0, 3])
        + _bands(hr, [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
        + (h["gcs"] < 15).astype(float) * 3
        + _bands(t, [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])
    )
    needed = pd.concat([rr, spo2, sbp, hr, t, h["gcs"]], axis=1).notna().all(axis=1)
    return s.where(needed)


def _bands(x: pd.Series, upper_edges: list[float], points: list[int]) -> pd.Series:
    """Points for x falling in (-inf, e0], (e0, e1], ..., (e_last, inf)."""
    idx = np.searchsorted(np.asarray(upper_edges), x.to_numpy(), side="left")
    out = np.asarray(points, dtype=float)[np.clip(idx, 0, len(points) - 1)]
    return pd.Series(np.where(x.isna(), np.nan, out), index=x.index)


def derive_layer1(h: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = pd.DataFrame(index=h.index)
    d[["gcs", "gcs_unable"]] = gcs_total(h)
    d[["fio2_eff", "fio2_eff__source", "fio2_eff__unc"]] = fio2_effective(h, cfg)
    hh = h.join(d)

    d["shock_index"] = h["heart_rate"] / h["sbp"].where(h["sbp"] > 0)                         # Berger et al. 2013
    d["pulse_pressure"] = h["sbp"] - h["dbp"]
    calc_map = h["dbp"] + (h["sbp"] - h["dbp"]) / 3
    d["map_any"] = h["map"].fillna(calc_map)
    d["map_any__source"] = np.where(h["map"].notna(), schema.MEASURED,
                                    np.where(calc_map.notna(), schema.DERIVED_FORMULA, ""))
    d["sf_ratio"] = h["spo2"] / d["fio2_eff"].where(d["fio2_eff"] > 0)                              # Rice et al. 2007
    valid = h["spo2"] <= cfg["spo2_max_for_pf_imputation"]
    d["pf_estimate"] = (pao2_from_spo2(h["spo2"]) / d["fio2_eff"]).where(valid)  # Brown et al. 2016
    d["rox"] = d["sf_ratio"] / h["resp_rate"].where(h["resp_rate"] > 0)                              # Roca et al. 2019
    d["qsofa"] = ((h["resp_rate"] >= 22).astype(float) + (h["sbp"] <= 100).astype(float)
                  + (d["gcs"] < 15).astype(float)).where(
        h[["resp_rate", "sbp"]].notna().all(axis=1) & d["gcs"].notna())     # Seymour et al. 2016
    d["news2"] = news2(hh)
    return d


# Lake metadata for each derived column: (unit, source, inputs).
LAYER1_META = {
    "gcs": ("points", schema.DERIVED_FORMULA, "gcs_eye,gcs_verbal,gcs_motor"),
    "fio2_eff": ("fraction", None, "fio2,o2_flow"),
    "shock_index": ("ratio", schema.DERIVED_FORMULA, "heart_rate,sbp"),
    "pulse_pressure": ("mmHg", schema.DERIVED_FORMULA, "sbp,dbp"),
    "map_any": ("mmHg", None, "map|sbp,dbp"),
    "sf_ratio": ("ratio", schema.DERIVED_FORMULA, "spo2,fio2_eff"),
    "pf_estimate": ("mmHg", schema.MODEL_ESTIMATE, "spo2,fio2_eff (Ellis/Brown 2016)"),
    "rox": ("ratio", schema.DERIVED_FORMULA, "sf_ratio,resp_rate"),
    "qsofa": ("points", schema.DERIVED_FORMULA, "resp_rate,sbp,gcs"),
    "news2": ("points", schema.DERIVED_FORMULA, "rr,spo2,o2,sbp,hr,gcs,temp"),
}
