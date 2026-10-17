"""eICU Collaborative Research Database DEMO (open access, ~2,500 ICU stays, 186 hospitals).

Two products:
  1. an hourly frame in PhysioNet/CinC 2019 column format, so the same quality
     gate, derivations, lakes and risk model run unchanged;
  2. an hourly TREATMENT panel (IV fluid volume, norepinephrine-equivalent dose),
     the real-treatment data the what-if engine is trained on.

Offsets in eICU are minutes from ICU admission; hour h covers [60h, 60h+60).
Dose conversion assumptions (stored with the panel):
  * mcg/min -> mcg/kg/min using admission weight;
  * norepinephrine charted in ml/hr assumes the common 4 mg / 250 mL bag (16 mcg/mL);
    other vasopressors charted in ml/hr are NOT converted (concentration unknown);
  * norepinephrine-equivalents follow Goradia et al. 2021 (as in config/interventions.yaml);
  * a charted infusion rate persists until the next charting of that drug, at most 4 h.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

MAX_HOURS = 336            # same cap as the PhysioNet 2019 records
NE_MCG_PER_ML = 16.0       # 4 mg in 250 mL
RATE_CARRY_H = 4

LAB_MAP = {
    "lactate": "Lactate", "pH": "pH", "HCO3": "HCO3", "paCO2": "PaCO2", "Base Excess": "BaseExcess",
    "O2 Sat (%)": "SaO2", "FiO2": "FiO2", "AST (SGOT)": "AST", "BUN": "BUN", "alkaline phos.": "Alkalinephos",
    "calcium": "Calcium", "chloride": "Chloride", "creatinine": "Creatinine", "direct bilirubin": "Bilirubin_direct",
    "glucose": "Glucose", "magnesium": "Magnesium", "phosphate": "Phosphate", "potassium": "Potassium",
    "total bilirubin": "Bilirubin_total", "troponin - I": "TroponinI", "Hct": "Hct", "Hgb": "Hgb", "PTT": "PTT",
    "WBC x 1000": "WBC", "fibrinogen": "Fibrinogen", "platelets x 1000": "Platelets",
}
FLUID_INCLUDE = re.compile(r"crystalloid|\bns\b|\blr\b|lactated|ringer|sodium chloride|saline|d5ns|0\.45 ns|"
                           r"ivf|albumin|bolus|d5 ?1/2|plasmalyte|plasma-lyte", re.I)
FLUID_EXCLUDE = re.compile(r"continuous infusion meds|other meds|ivpb|enteral|feeding|flush|water|propofol|"
                           r"diltiazem|heparin|amiodarone|insulin|tpn|blood|platelet|plasma\b|ffp|cryo|"
                           r"nicardipine|vasopressin|norepinephrine|phenylephrine|precedex|dexmedetomidine|fentanyl|"
                           r"midazolam|nitroglycerin|esmolol|labetalol|potassium|magnesium|antibiotic|vanco|zosyn",
                           re.I)
# drug -> (NE-equivalent factor, accepted units)
VASO = {"norepinephrine": 1.0, "epinephrine": 1.0, "dopamine": 0.01, "phenylephrine": 0.1, "vasopressin": 2.5}


def _q(sql: str) -> pd.DataFrame:
    return duckdb.sql(sql).df()


def patients(root: Path) -> pd.DataFrame:
    p = _q(f"SELECT * FROM '{root}/patient.csv.gz'")
    p["age_n"] = pd.to_numeric(p["age"].replace({"> 89": "90"}), errors="coerce")
    p["los_h"] = np.minimum(np.ceil(p["unitdischargeoffset"] / 60), MAX_HOURS)
    return p[(p["age_n"] >= 18) & (p["los_h"] >= 4)].reset_index(drop=True)


def hourly_frame(root: Path) -> pd.DataFrame:
    """PhysioNet-2019-format hourly rows for every adult stay >= 4 h."""
    p = patients(root)
    ids = ",".join(map(str, p["patientunitstayid"]))
    vp = _q(f"""
        SELECT patientunitstayid AS stay_id, floor(observationoffset / 60) AS hour,
               avg(heartrate) HR, avg(sao2) O2Sat, avg(respiration) Resp, avg(temperature) "Temp",
               avg(systemicsystolic) SBP_a, avg(systemicdiastolic) DBP_a, avg(systemicmean) MAP_a, avg(etco2) EtCO2
        FROM '{root}/vitalPeriodic.csv.gz'
        WHERE observationoffset >= 0 AND patientunitstayid IN ({ids}) GROUP BY 1, 2""")
    va = _q(f"""
        SELECT patientunitstayid AS stay_id, floor(observationoffset / 60) AS hour,
               avg(noninvasivesystolic) SBP_n, avg(noninvasivediastolic) DBP_n, avg(noninvasivemean) MAP_n
        FROM '{root}/vitalAperiodic.csv.gz'
        WHERE observationoffset >= 0 AND patientunitstayid IN ({ids}) GROUP BY 1, 2""")
    v = vp.merge(va, on=["stay_id", "hour"], how="outer")
    for k in ("SBP", "DBP", "MAP"):                       # arterial line preferred, cuff otherwise
        v[k] = v[f"{k}_a"].fillna(v[f"{k}_n"])
    v.loc[v["Temp"] > 45, "Temp"] = (v["Temp"] - 32) * 5 / 9   # Fahrenheit charted in the Celsius field
    names = ",".join(f"'{n}'" for n in LAB_MAP)
    labs = _q(f"""
        SELECT patientunitstayid AS stay_id, floor(labresultoffset / 60) AS hour, labname, avg(labresult) AS v
        FROM '{root}/lab.csv.gz'
        WHERE labresultoffset >= 0 AND labname IN ({names}) AND patientunitstayid IN ({ids}) GROUP BY 1, 2, 3""")
    labs["col"] = labs["labname"].map(LAB_MAP)
    lw = labs.pivot_table(index=["stay_id", "hour"], columns="col", values="v").reset_index()
    df = v[["stay_id", "hour", "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]].merge(
        lw, on=["stay_id", "hour"], how="outer")
    st = p.set_index("patientunitstayid")
    df = df[df["hour"] < df["stay_id"].map(st["los_h"])]
    # one row for every ICU hour, as in the challenge files
    grid = pd.DataFrame({"stay_id": np.repeat(st.index, st["los_h"].astype(int)),
                         "hour": np.concatenate([np.arange(int(n)) for n in st["los_h"]])})
    df = grid.merge(df, on=["stay_id", "hour"], how="left")
    sid = df["stay_id"]
    df["Age"] = sid.map(st["age_n"])
    df["Gender"] = (sid.map(st["gender"]) == "Male").astype(int)
    df["Unit1"] = (sid.map(st["unittype"]) == "MICU").astype(float)
    df["Unit2"] = (sid.map(st["unittype"]) == "SICU").astype(float)
    df["HospAdmTime"] = sid.map(st["hospitaladmitoffset"]) / 60.0
    df["ICULOS"] = df["hour"] + 1
    df["SepsisLabel"] = 0          # no Sepsis-3 labels are derived for eICU: risk is shown, not scored
    df["hospital_set"] = "eICU"
    df["hour"] = df["hour"].astype(int)
    return df.sort_values(["stay_id", "hour"]).reset_index(drop=True)


def treatment_panel(root: Path) -> pd.DataFrame:
    """Hourly IV fluid volume (mL) and norepinephrine-equivalent dose (mcg/kg/min)."""
    p = patients(root).set_index("patientunitstayid")
    inf = _q(f"""SELECT patientunitstayid AS stay_id, infusionoffset AS off, lower(drugname) AS drug,
                        drugrate, patientweight FROM '{root}/infusiondrug.csv.gz' WHERE infusionoffset >= 0""")
    inf = inf[inf["stay_id"].isin(p.index)].copy()
    inf["rate"] = pd.to_numeric(inf["drugrate"], errors="coerce")
    inf["w"] = pd.to_numeric(inf["patientweight"], errors="coerce").fillna(inf["stay_id"].map(p["admissionweight"]))
    inf["w"] = inf["w"].where(inf["w"].between(30, 300))
    rows = []
    for drug, factor in VASO.items():
        d = inf[inf["drug"].str.contains(drug) & ~inf["drug"].str.contains("volume")].copy()
        if drug == "epinephrine":
            d = d[~d["drug"].str.contains("norepinephrine")]
        unit = d["drug"].str.extract(r"\(([^)]*)\)")[0].fillna("")
        dose = pd.Series(np.nan, index=d.index)
        dose[unit == "mcg/kg/min"] = d["rate"]
        dose[unit == "mcg/min"] = d["rate"] / d["w"]
        if drug == "norepinephrine":
            dose[unit == "ml/hr"] = d["rate"] * NE_MCG_PER_ML / 60 / d["w"]
        if drug == "vasopressin":
            dose[unit == "units/min"] = d["rate"]
            dose[unit == "units/hr"] = d["rate"] / 60
        d["ne"] = dose * factor
        d["drugkey"] = drug
        rows.append(d.dropna(subset=["ne"]))
    vaso = pd.concat(rows, ignore_index=True)
    vaso = vaso[vaso["ne"].between(0, 5)]
    vaso["hour"] = (vaso["off"] // 60).astype(int)
    # charted rate persists until the next charting of that drug (max RATE_CARRY_H hours)
    per_drug = vaso.groupby(["stay_id", "drugkey", "hour"])["ne"].last().reset_index()
    filled = []
    for (sid, drug), g in per_drug.groupby(["stay_id", "drugkey"]):
        hours = np.arange(g["hour"].min(), g["hour"].max() + RATE_CARRY_H + 1)
        s = g.set_index("hour")["ne"].reindex(hours).ffill(limit=RATE_CARRY_H)
        filled.append(pd.DataFrame({"stay_id": sid, "hour": hours, "ne": s.to_numpy()}))
    ne = (pd.concat(filled).groupby(["stay_id", "hour"])["ne"].sum(min_count=1) if filled
          else pd.Series(dtype=float))

    io = _q(f"""SELECT patientunitstayid AS stay_id, intakeoutputoffset AS off, cellpath, cellvaluenumeric AS ml
                FROM '{root}/intakeOutput.csv.gz' WHERE intakeoutputoffset >= 0 AND lower(cellpath) LIKE '%intake%'""")
    io = io[io["stay_id"].isin(p.index) & (io["ml"] > 0) & (io["ml"] < 5000)]
    leaf = io["cellpath"].str.split("|").str[-1]
    keep = io["cellpath"].str.contains(FLUID_INCLUDE) & ~leaf.str.contains(FLUID_EXCLUDE)
    io = io[keep]
    fl = io.assign(hour=(io["off"] // 60).astype(int)).groupby(["stay_id", "hour"])["ml"].sum()

    grid = pd.DataFrame({"stay_id": np.repeat(p.index, p["los_h"].astype(int)),
                         "hour": np.concatenate([np.arange(int(n)) for n in p["los_h"]])}).set_index(["stay_id", "hour"])
    out = grid.join(ne.rename("ne_dose")).join(fl.rename("fluid_ml"))
    out["ne_dose"] = out["ne_dose"].fillna(0.0)
    out["fluid_ml"] = out["fluid_ml"].fillna(0.0)
    has_infusion_feed = inf.groupby("stay_id").size()
    out["infusion_documented"] = out.index.get_level_values("stay_id").isin(has_infusion_feed.index).astype("int8")
    return out
