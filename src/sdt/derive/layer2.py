"""Layer 2: parameters derived from labs (plan, "Layer 2" table A).

SOFA is computed twice:
  sofa_label    measured values only (PaO2 from blood gases). Used to build
                the Sepsis-3 training labels, so labels never depend on an
                estimate.
  sofa          the same, but the respiratory score falls back to the Layer 1
                SpO2-based PaO2/FiO2 estimate when no blood gas exists. Used
                as a model feature only.
Both use the worst value of the previous 24 h, as in the MIT-LCP mimic-code
SOFA concept (Vincent et al. 1996 scoring).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sdt import schema
from sdt.derive.hourly import rolling_hours

# ---------------------------------------------------------------- SOFA scores

def _cut(x: pd.Series, edges: list[float], points: list[int], right: bool = False) -> pd.Series:
    """Points by interval: edges are ascending boundaries; `right` closes on the right."""
    idx = np.searchsorted(np.asarray(edges), x.to_numpy(), side="left" if right else "right")
    out = np.asarray(points, dtype=float)[idx]
    return pd.Series(np.where(x.isna(), np.nan, out), index=x.index)


def sofa_resp(pf: pd.Series, ventilated: pd.Series) -> pd.Series:
    s = _cut(pf, [100, 200, 300, 400], [4, 3, 2, 1, 0])
    # Scores 3 and 4 require respiratory support.
    return s.where(ventilated.astype(bool) | s.isna(), np.minimum(s, 2))


def sofa_coag(platelets: pd.Series) -> pd.Series:
    return _cut(platelets, [20, 50, 100, 150], [4, 3, 2, 1, 0])


def sofa_liver(bili: pd.Series) -> pd.Series:
    return _cut(bili, [1.2, 2.0, 6.0, 12.0], [0, 1, 2, 3, 4])


def sofa_cns(gcs: pd.Series) -> pd.Series:
    return _cut(gcs, [6, 10, 13, 15], [4, 3, 2, 1, 0])


def sofa_cv(map_min: pd.Series, iv: pd.DataFrame) -> pd.Series:
    dop, dob = iv["vaso_dopamine"], iv["vaso_dobutamine"]
    epi, ne = iv["vaso_epinephrine"], iv["vaso_norepinephrine"]
    s = pd.Series(np.where(map_min < 70, 1.0, np.where(map_min.notna(), 0.0, np.nan)), index=map_min.index)
    s = s.mask((dop > 0) & (dop <= 5) | (dob > 0), 2)
    s = s.mask((dop > 5) | ((epi > 0) & (epi <= 0.1)) | ((ne > 0) & (ne <= 0.1)), 3)
    s = s.mask((dop > 15) | (epi > 0.1) | (ne > 0.1), 4)
    return s


def sofa_renal(creat_max: pd.Series, uo_24h: pd.Series) -> pd.Series:
    c = _cut(creat_max, [1.2, 2.0, 3.5, 5.0], [0, 1, 2, 3, 4])
    u = pd.Series(np.select([uo_24h < 200, uo_24h < 500], [4.0, 3.0], default=0.0), index=uo_24h.index)
    u = u.where(uo_24h.notna())
    return pd.concat([c, u], axis=1).max(axis=1, skipna=True)


def urine_24h(h: pd.DataFrame) -> pd.Series:
    """24 h urine output. Valid only once 24 full ICU hours exist and at least
    one output was charted in the window."""
    ml = h["urine_ml"].fillna(0)
    total = rolling_hours(ml, 24, "sum")
    charted = rolling_hours(h["urine_ml"].notna().astype(float), 24, "sum")
    hour = pd.Series(h.index.get_level_values("hour"), index=h.index)
    return total.where((hour >= 23) & (charted > 0))


def sofa_scores(h: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    vent = (h["vent_invasive"] > 0) | (h["vent_noninvasive"] > 0)
    pf_meas_hour = h["pao2__min"] / d1["fio2_eff"]
    pf_any_hour = pf_meas_hour.fillna(d1["pf_estimate"])
    uo = urine_24h(h)
    hourly = {
        "resp_label": sofa_resp(pf_meas_hour, vent),
        "resp": sofa_resp(pf_any_hour, vent),
        "coag": sofa_coag(h["platelets__min"]),
        "liver": sofa_liver(h["bilirubin__max"]),
        "cv": sofa_cv(h["map__min"].fillna(d1["map_any"]), h),
        "cns": sofa_cns(d1["gcs"]),
        "renal": sofa_renal(h["creatinine__max"], uo),
    }
    out = pd.DataFrame({f"sofa_{k}": rolling_hours(v, 24, "max") for k, v in hourly.items()})
    organs = ["coag", "liver", "cv", "cns", "renal"]
    base = out[[f"sofa_{o}" for o in organs]].fillna(0).sum(axis=1)
    out["sofa_label"] = base + out["sofa_resp_label"].fillna(0)
    out["sofa"] = base + out["sofa_resp"].fillna(0)
    out["sofa_n_observed"] = out[[f"sofa_{o}" for o in organs + ["resp"]]].notna().sum(axis=1)
    out["pf_measured"] = pf_meas_hour
    out["urine_24h_ml"] = uo
    return out


# ----------------------------------------------------------- renal (KDIGO)

def egfr_ckd_epi_2021(creat, age, female) -> pd.Series:
    """CKD-EPI 2021 race-free equation (Inker et al. 2021, NEJM). Valid only at
    steady state; during AKI it overestimates function."""
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    r = creat / kappa
    return (142 * np.minimum(r, 1) ** alpha * np.maximum(r, 1) ** -1.200
            * 0.9938 ** age * np.where(female, 1.012, 1.0))


def kdigo(h: pd.DataFrame, weight_kg: pd.Series) -> pd.DataFrame:
    creat = h["creatinine"]
    measured = h["creatinine__max"]
    # Baseline: lowest creatinine seen so far (Cooper et al., Kidney Int Rep:
    # in acute infection this beats the fixed GFR-75 back-calculation).
    baseline = measured.groupby(level="stay_id").cummin().groupby(level="stay_id").ffill()
    min_48h = rolling_hours(measured, 48, "min")
    ratio = creat / baseline
    stage_cr = pd.Series(np.select(
        [(ratio >= 3) | (creat >= 4.0), ratio >= 2, (ratio >= 1.5) | (creat - min_48h >= 0.3)],
        [3.0, 2.0, 1.0], default=0.0), index=h.index).where(creat.notna())

    w = h.index.get_level_values("stay_id").map(weight_kg)
    rate = h["urine_ml"].fillna(0) / np.asarray(w, dtype=float)       # mL/kg/h
    hour = pd.Series(h.index.get_level_values("hour"), index=h.index)
    m6, m12, m24 = (rolling_hours(rate, n, "mean") for n in (6, 12, 24))
    s12 = rolling_hours(h["urine_ml"].fillna(0), 12, "sum")
    stage_uo = pd.Series(np.select(
        [((m24 < 0.3) & (hour >= 23)) | ((s12 == 0) & (hour >= 11)),
         (m12 < 0.5) & (hour >= 11), (m6 < 0.5) & (hour >= 5)],
        [3.0, 2.0, 1.0], default=0.0), index=h.index).where(np.isfinite(rate))
    return pd.DataFrame({
        "creatinine_baseline": baseline, "kdigo_creat_stage": stage_cr,
        "urine_ml_kg_h": rate.where(np.isfinite(rate)), "kdigo_uo_stage": stage_uo,
        "kdigo_stage": pd.concat([stage_cr, stage_uo], axis=1).max(axis=1),
    })


# --------------------------------------------------------- coagulopathy

def sic_score(platelets, inr, sofa4) -> pd.DataFrame:
    """Sepsis-induced coagulopathy (Iba et al. 2017, BMJ Open). `sofa4` is the
    sum of respiratory, cardiovascular, hepatic and renal SOFA."""
    p = _cut(platelets, [100, 150], [2, 1, 0])
    i = _cut(inr, [1.2, 1.4], [0, 1, 2], right=True)
    s = np.minimum(sofa4, 2)
    total = (p + i + s).where(platelets.notna() & inr.notna())
    return pd.DataFrame({"sic_score": total, "sic_positive": ((total >= 4) & (p + i > 2)).astype(float).where(total.notna())})


def isth_dic(platelets, d_dimer, pt, fibrinogen_mgdl, cfg) -> pd.Series:
    """ISTH overt DIC score (Taylor et al. 2001). D-dimer bands are assay
    specific: thresholds come from config and must be set locally."""
    p = pd.Series(np.select([platelets < 50, platelets <= 100], [2.0, 1.0], 0.0), index=platelets.index)
    dd = _cut(d_dimer, [cfg["dic_d_dimer_moderate_ngml"], cfg["dic_d_dimer_strong_ngml"]], [0, 2, 3])
    prolong = pt - cfg["dic_normal_pt_s"]
    t = pd.Series(np.select([prolong > 6, prolong >= 3], [2.0, 1.0], 0.0), index=pt.index)
    f = (fibrinogen_mgdl <= 100).astype(float)   # 1 g/L = 100 mg/dL
    ok = pd.concat([platelets, d_dimer, pt, fibrinogen_mgdl], axis=1).notna().all(axis=1)
    return (p + dd + t + f).where(ok)


# --------------------------------------------------------------- other labs

def lactate_clearance(lake: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    """(L_prev - L_now) / L_prev x 100 between consecutive lactates 1-24 h apart
    (Nguyen et al. 2004). Attached to the hour of the newer measurement."""
    from sdt.derive.hourly import hour_index
    lac = lake[(lake["param"] == "lactate") & lake["flag"].isin(schema.USABLE_FLAGS)]
    lac = lac.sort_values(["stay_id", "charttime"])
    g = lac.groupby("stay_id")
    prev, prev_t = g["value"].shift(1), g["charttime"].shift(1)
    gap = (lac["charttime"] - prev_t).dt.total_seconds() / 3600
    clr = ((prev - lac["value"]) / prev * 100).where((gap >= 1) & (gap <= 24))
    intime = lac["stay_id"].map(stays.set_index("stay_id")["intime"])
    out = pd.DataFrame({"stay_id": lac["stay_id"], "hour": hour_index(lac["charttime"], intime),
                        "lactate_clearance_pct": clr}).dropna()
    return out.groupby(["stay_id", "hour"]).last()


def derive_layer2(h: pd.DataFrame, d1: pd.DataFrame, lake: pd.DataFrame, stays: pd.DataFrame,
                  cfg: dict) -> pd.DataFrame:
    d = sofa_scores(h, d1)
    st = stays.set_index("stay_id")
    sid = h.index.get_level_values("stay_id")
    age = np.asarray(sid.map(st["age"]), dtype=float)
    female = np.asarray(sid.map(st["gender"]) == "F")
    weight = st["weight_kg"] if "weight_kg" in st else pd.Series(dtype=float)

    d["anion_gap"] = h["sodium"] - (h["chloride"] + h["bicarbonate"])
    d["anion_gap_albumin_corr"] = d["anion_gap"] + 2.5 * (4.0 - h["albumin"])     # Figge et al. 1998
    d["nlr"] = h["neutrophils_pct"] / h["lymphocytes_pct"].where(h["lymphocytes_pct"] > 0)
    d["bun_creatinine_ratio"] = h["bun"] / h["creatinine"].where(h["creatinine"] > 0)
    d["egfr_ckd_epi_2021"] = egfr_ckd_epi_2021(h["creatinine"], age, female)
    d = d.join(kdigo(h, weight))
    sofa4 = d[["sofa_resp", "sofa_cv", "sofa_liver", "sofa_renal"]].fillna(0).sum(axis=1)
    d = d.join(sic_score(h["platelets"], h["inr"], sofa4))
    d["isth_dic_score"] = isth_dic(h["platelets"], h["d_dimer"], h["pt"], h["fibrinogen"], cfg)
    d = d.join(lactate_clearance(lake, stays), how="left")
    return d


LAYER2_META = {
    "sofa_label": ("points", schema.DERIVED_FORMULA, "measured values only; label use"),
    "sofa": ("points", schema.MODEL_ESTIMATE, "resp uses SpO2-based P/F when no blood gas"),
    "pf_measured": ("mmHg", schema.DERIVED_FORMULA, "pao2,fio2_eff"),
    "urine_24h_ml": ("mL", schema.DERIVED_FORMULA, "urine_output"),
    "anion_gap": ("mmol/L", schema.DERIVED_FORMULA, "sodium,chloride,bicarbonate"),
    "anion_gap_albumin_corr": ("mmol/L", schema.DERIVED_FORMULA, "anion_gap,albumin"),
    "nlr": ("ratio", schema.DERIVED_FORMULA, "neutrophils_pct,lymphocytes_pct"),
    "bun_creatinine_ratio": ("ratio", schema.DERIVED_FORMULA, "bun,creatinine"),
    "egfr_ckd_epi_2021": ("mL/min/1.73m2", schema.DERIVED_FORMULA, "creatinine,age,sex"),
    "creatinine_baseline": ("mg/dL", schema.DERIVED_FORMULA, "lowest creatinine so far"),
    "kdigo_stage": ("stage", schema.DERIVED_FORMULA, "creatinine,baseline,urine_ml_kg_h"),
    "urine_ml_kg_h": ("mL/kg/h", schema.DERIVED_FORMULA, "urine_output,weight"),
    "sic_score": ("points", schema.DERIVED_FORMULA, "platelets,inr,sofa4"),
    "isth_dic_score": ("points", schema.DERIVED_FORMULA, "platelets,d_dimer,pt,fibrinogen"),
    "lactate_clearance_pct": ("%", schema.DERIVED_FORMULA, "consecutive lactates 1-24 h apart"),
}
