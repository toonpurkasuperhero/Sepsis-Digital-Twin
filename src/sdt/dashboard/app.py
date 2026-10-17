"""Sepsis Digital Twin dashboard (FastAPI backend + static single-page front end).

Two replay cohorts, both external to the training hospital:
  physionet_b  PhysioNet 2019 set B (no treatment records: what-if disabled)
  eicu         eICU-CRD demo, 186 hospitals, with infusion and fluid records: what-if enabled
Each stay is replayed hour by hour through the trained fusion model, as in silent mode. A replay
clock sets "now"; nothing after "now" is ever sent to the browser, so the
screen shows only what a clinician would have known at that hour.

  python -m sdt.dashboard.app            # http://127.0.0.1:8050
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sdt.config import PROJECT_ROOT, load_config
from sdt.deploy.silent import ShadowRunner
from sdt.model.data import build_stays, read_lake
from sdt.pipeline_physionet import load_adapter
from sdt.sources import physionet2019 as pn

STATIC = Path(__file__).parent / "static"
VITALS = ["heart_rate", "map", "sbp", "resp_rate", "spo2", "temperature"]
DERIVED_SHOWN = {"shock_index": ("Shock index", "HR / SBP", 2), "sf_ratio": ("SpO2/FiO2", "ratio", 0),
                 "pf_estimate": ("Estimated PaO2/FiO2", "mmHg", 0), "pulse_pressure": ("Pulse pressure", "mmHg", 0),
                 "sofa": ("SOFA (partial)", "points", 0),
                 "egfr_ckd_epi_2021": ("eGFR (CKD-EPI 2021)", "mL/min/1.73m²", 0),
                 "kdigo_stage": ("KDIGO creatinine stage", "stage", 0),
                 "lactate_clearance_pct": ("Lactate clearance", "%", 0)}
LABELS = {"heart_rate": "Heart rate", "map": "Mean arterial pressure", "sbp": "Systolic BP",
          "dbp": "Diastolic BP", "resp_rate": "Respiratory rate", "spo2": "SpO2", "temperature": "Temperature",
          "fio2": "FiO2", "etco2": "EtCO2",
          "lactate": "Lactate", "ph": "pH", "bicarbonate": "Bicarbonate", "base_excess": "Base excess",
          "paco2": "PaCO2", "sao2": "SaO2", "creatinine": "Creatinine", "bun": "Urea nitrogen (BUN)",
          "bilirubin": "Bilirubin, total", "bilirubin_direct": "Bilirubin, direct", "ast": "AST",
          "alk_phos": "Alkaline phosphatase", "platelets": "Platelets", "wbc": "White cell count",
          "hemoglobin": "Haemoglobin", "hematocrit": "Haematocrit", "fibrinogen": "Fibrinogen",
          "ptt": "PTT", "glucose_lab": "Glucose", "potassium": "Potassium", "chloride": "Chloride",
          "calcium": "Calcium", "magnesium": "Magnesium", "phosphate": "Phosphate",
          "troponin_i": "Troponin I"}


COHORTS = {
    "physionet_b": {"label": "Hospital B (PhysioNet 2019)", "root": None, "treatments": False,
                    "note": "External test hospital. Risk model calibrated on hospital A."},
    "eicu": {"label": "eICU demo, 186 hospitals (treatments recorded)", "root": "data/eicu_demo",
             "treatments": True,
             "note": "Risk model not recalibrated for these hospitals; Sepsis-3 labels not derived."},
}
ENGINE_PATH = PROJECT_ROOT / "data" / "whatif" / "engine_v2.pkl"


def _num(x, nd=2):
    if x is None or pd.isna(x):
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


class Twin:
    """Loads the model and a replay cohort once; serves causal views of it."""

    def __init__(self, cohort: str = "physionet_b", n_stays: int = 240, n_septic: int = 60, seed: int = 0):
        cfg = load_config()
        adapter = load_adapter()
        _, model_root = pn.adapter_paths(adapter)
        self.cohort = cohort
        self.meta_c = COHORTS[cohort]
        root = model_root if self.meta_c["root"] is None else PROJECT_ROOT / self.meta_c["root"]
        self.runner = ShadowRunner(model_root / "models", model_root / "reports" / "phase3_4_report.json")
        self.report = json.loads((model_root / "reports" / "phase3_4_report.json").read_text())
        sc = self.schema = self.runner.schema
        prim, sec = pn.specs_for(cfg.parameters, adapter)
        self.specs = {**prim, **sec}

        static = read_lake(root, "lake3_tertiary")
        static = static.join(pd.get_dummies(static["unit"], prefix="unit", dtype=float))
        static["hosp_adm_time_h"] = static["hosp_adm_time_h"].clip(-2000, 0)
        for c in sc.static:
            if c not in static:
                static[c] = 0.0
        outcomes = read_lake(root, "outcomes")
        if cohort == "physionet_b":
            b = static[static["hospital_set"] == "B"].join(outcomes[["onset_hour"]])
            septic = b[b["onset_hour"] >= 6].sample(n_septic, random_state=seed)
            other = b[b["onset_hour"].isna()].sample(n_stays - n_septic, random_state=seed)
            ids = septic.index.append(other.index)
            self.treat = None
        else:
            iv = read_lake(root, "interventions")
            per = iv.groupby(level="stay_id").agg(ne=("ne_dose", "max"), fl=("fluid_ml", "max"))
            treated = per[(per["ne"] > 0) | (per["fl"] >= 250)].index.intersection(static.index)
            untreated = per.index.difference(treated).intersection(static.index)
            n_t = min(len(treated), int(n_stays * 0.75))
            ids = (pd.Index(treated).to_series().sample(n_t, random_state=seed).index
                   .append(pd.Index(untreated).to_series().sample(n_stays - n_t, random_state=seed).index))
            self.treat = iv
        tl = read_lake(root, "timeline")
        tl = tl[tl.index.get_level_values("stay_id").isin(ids)]
        self.timeline = tl
        stays = build_stays(tl, static.loc[ids], sc, group_col="hospital_set")
        self.stays = {s.stay_id: s for s in stays}
        self.risk, self.gates = {}, {}
        for s, (cal, g) in zip(stays, self.runner.risks(stays)):
            self.risk[s.stay_id], self.gates[s.stay_id] = cal, g
        self.static = static.loc[ids]
        self.onset = outcomes["onset_hour"].reindex(ids)
        self.thr = self.runner.threshold
        self.max_hour = int(max(len(s.y) for s in stays)) - 1
        self.engine = None
        if self.meta_c["treatments"] and ENGINE_PATH.exists():
            self.engine = pickle.loads(ENGINE_PATH.read_bytes())

    def panel(self, sid: int, upto: int) -> pd.DataFrame:
        """Hourly MAP/HR/lactate/treatment history of one stay up to hour `upto` (what-if input)."""
        tl = self.timeline.xs(sid, level="stay_id")
        iv = self.treat.xs(sid, level="stay_id").reindex(tl.index)
        d = pd.DataFrame({"stay_id": str(sid), "hour": tl.index.to_numpy(),
                          "map": tl["map"].where(tl["map__mask"] == 1).to_numpy(),
                          "hr": tl["heart_rate"].where(tl["heart_rate__mask"] == 1).to_numpy(),
                          "lactate": tl["lactate"].to_numpy(),
                          "ne_dose": iv["ne_dose"].fillna(0).to_numpy(), "fluid_ml": iv["fluid_ml"].fillna(0).to_numpy(),
                          "age": float(self.static.loc[sid, "age"])})
        return d[d["hour"] <= upto]

    def whatif(self, sid: int, clock: int, fluid_ml: float | None, ne_dose: float | None) -> dict:
        if self.engine is None:
            return {"available": False, "reason": (
                "No treatment records exist for this cohort, so there is no treatment history to simulate from. "
                "Switch to the eICU cohort, where infusions and fluids are recorded.")}
        s = self.stays.get(sid)
        if s is None:
            raise HTTPException(404, "unknown patient")
        h = min(clock, len(s.y) - 1)
        hist = self.panel(sid, h)
        if hist["map"].notna().sum() < 2:
            return {"available": False, "reason": "Too few blood-pressure measurements so far to simulate from."}
        ne_now = float(hist["ne_dose"].iloc[-1])
        eng = self.engine
        setup = {"available": True, "hour": h, "ne_now": _num(ne_now, 3),
                 "ne_max": _num(eng.support["ne_max"], 2), "fluid_options": [0, 250, 500, 1000],
                 "assumption": "", "effects": {o: {k: v.as_dict() for k, v in e.items()} for o, e in eng.effects.items()}}
        if fluid_ml is None and ne_dose is None:
            return setup
        r = eng.query(hist, float(fluid_ml or 0.0), float(ne_now if ne_dose is None else ne_dose))
        setup["assumption"] = r["assumption"]
        setup["answered"] = r["answered"]
        if not r["answered"]:
            setup["reasons"] = r["reasons"]
            return setup
        setup["plan"] = r["plan"]
        setup["trajectories"] = {o: {c: [_num(v, 1) for v in df[c]] for c in df.columns}
                                 for o, df in r["trajectories"].items()}
        past = hist.tail(12)
        setup["recent"] = {"hours": past["hour"].tolist(), "map": [_num(v, 0) for v in past["map"]],
                           "hr": [_num(v, 0) for v in past["hr"]]}
        return setup

    # ------------------------------------------------------------ views
    def ward(self, clock: int) -> list[dict]:
        rows = []
        for sid, s in self.stays.items():
            last = len(s.y) - 1
            h = min(clock, last)
            r = self.risk[sid]
            st = self.static.loc[sid]
            hr = s.x1[:h + 1, self.schema.primary.index("heart_rate")]
            mp = s.x1[:h + 1, self.schema.primary.index("map")]
            rows.append({
                "stay_id": sid, "age": int(st["age"]), "sex": "F" if st["female"] else "M", "unit": st["unit"],
                "hour": h, "in_icu": clock <= last,
                "risk_6h": _num(r[h, 0], 5), "risk_12h": _num(r[h, 1], 5),
                "alert": bool(r[h, 0] >= self.thr["6h"]),
                "trend": [_num(v, 5) for v in r[max(0, h - 11):h + 1, 0]],
                "hr": _num(_last(hr), 0), "map": _num(_last(mp), 0),
            })
        return sorted(rows, key=lambda x: (not x["in_icu"], -(x["risk_6h"] or 0)))

    def patient(self, sid: int, clock: int, show_outcome: bool) -> dict:
        s = self.stays.get(sid)
        if s is None:
            raise HTTPException(404, "unknown patient")
        h = min(clock, len(s.y) - 1)
        sc, r, g = self.schema, self.risk[sid], self.gates[sid]
        st = self.static.loc[sid]
        vitals = {}
        for p in VITALS:
            j = sc.primary.index(p)
            vitals[p] = {"label": LABELS[p], "unit": self.specs[p]["unit"],
                         "values": [_num(v, 1) for v in s.x1[:h + 1, j]]}
        labs = []
        for j, p in enumerate(sc.secondary):
            col = s.x2[:h + 1, j]
            idx = np.flatnonzero(~np.isnan(col))
            if not len(idx):
                continue
            v = float(col[idx[-1]])
            lo, hi = self.specs[p].get("extreme", (None, None))
            labs.append({"param": p, "label": LABELS.get(p, p), "value": _num(v, 2),
                         "unit": self.specs[p]["unit"], "hours_ago": int(h - idx[-1]),
                         "n_measured": int(len(idx)),
                         "flag": "extreme" if (lo is not None and (v < lo or v > hi)) else ""})
        labs.sort(key=lambda x: (x["flag"] != "extreme", x["hours_ago"]))
        derived = []
        for name, (label, unit, nd) in DERIVED_SHOWN.items():
            src = s.d1 if name in sc.derived1 else s.d2 if name in sc.derived2 else None
            if src is None:
                continue
            col = src[:h + 1, (sc.derived1 if name in sc.derived1 else sc.derived2).index(name)]
            idx = np.flatnonzero(~np.isnan(col))
            if len(idx):
                derived.append({"label": label, "unit": unit, "value": _num(col[idx[-1]], nd),
                                "hours_ago": int(h - idx[-1])})
        out = {
            "stay_id": sid, "hour": h, "last_hour": len(s.y) - 1, "in_icu": clock <= len(s.y) - 1,
            "age": int(st["age"]), "sex": "F" if st["female"] else "M", "unit": st["unit"],
            "hospital_admission_h_before_icu": _num(-st["hosp_adm_time_h"], 1),
            "risk": {"6h": [_num(v, 5) for v in r[:h + 1, 0]], "12h": [_num(v, 5) for v in r[:h + 1, 1]]},
            "threshold": self.thr, "alert": bool(r[h, 0] >= self.thr["6h"]),
            "streams": [{"name": n, "weight": _num(w, 3)} for n, w in
                        zip(["Vital signs", "Laboratory results", "Baseline (age, unit, admission)"], g[h])],
            "vitals": vitals, "labs": labs, "derived": derived,
            "data_quality": {"vitals_measured_pct": _num(100 * np.mean(~np.isnan(s.x1[:h + 1, :len(VITALS)])), 0),
                             "labs_drawn": int(np.sum(~np.isnan(s.x2[:h + 1]))),
                             "missing_by_design": "GCS, urine output, PaO2 and treatments are not recorded in this dataset"},
        }
        if self.treat is not None:
            iv = self.treat.xs(sid, level="stay_id").reindex(range(h + 1))
            out["treatments"] = {"ne_dose": [_num(v, 3) for v in iv["ne_dose"].fillna(0)],
                                 "fluid_ml": [_num(v, 0) for v in iv["fluid_ml"].fillna(0)]}
        out["cohort"] = self.cohort
        if show_outcome:     # replay/teaching only: the recorded future is hidden by default
            on = self.onset.get(sid)
            out["recorded_onset_hour"] = None if pd.isna(on) else int(on)
        return out

    def model_card(self) -> dict:
        p4 = self.report["phase4"]
        ext, inn = p4["external_test_setB_6h"], p4["internal_test_6h"]
        return {
            "name": "Sepsis Digital Twin - fusion model v0.1",
            "intended_use": "Research prototype. Estimates the probability that Sepsis-3 begins within the next "
                            "6 and 12 hours for adult ICU patients. Decision support only; not a medical device.",
            "not_for": "Clinical decisions, patients outside the ICU, children, or hospitals it has not been calibrated for.",
            "training_data": "PhysioNet/CinC 2019 set A (20,336 ICU stays, hospital system A).",
            "validation": [
                {"set": "Internal test (hospital A)", "auroc": _num(inn["auroc"], 3), "auroc_ci95": [_num(x, 3) for x in inn["auroc_ci95"]],
                 "sensitivity": _num(inn["sensitivity"], 2), "ppv": _num(inn["ppv"], 3),
                 "alerts_per_patient_day": _num(inn["alerts_per_patient_day"], 1)},
                {"set": "External test (hospital B)", "auroc": _num(ext["auroc"], 3), "auroc_ci95": [_num(x, 3) for x in ext["auroc_ci95"]],
                 "sensitivity": _num(ext["sensitivity"], 2), "ppv": _num(ext["ppv"], 3),
                 "alerts_per_patient_day": _num(ext["alerts_per_patient_day"], 1)},
            ],
            "operating_point": "Alert when 6 h risk >= threshold set for 80 % sensitivity on hospital A validation data.",
            "limitations": [
                "Alert burden is high at this operating point (about 6.6 alert-hours per patient-day externally); "
                "the threshold must be agreed with clinicians.",
                "Risk is over-estimated at hospital B (calibration intercept -1.2): recalibrate before use at any new site.",
                "No treatment data in the training source: the what-if simulator is disabled.",
                "Replay cohort on this screen is enriched for sepsis (60 of 240 stays) to show alerts; "
                "real prevalence is about 6 %.",
            ],
        }


def _last(col):
    idx = np.flatnonzero(~np.isnan(col))
    return col[idx[-1]] if len(idx) else None


def create_app(twin: Twin | None = None) -> FastAPI:
    app = FastAPI(title="Sepsis Digital Twin", docs_url="/api/docs")
    twins: dict[str, Twin] = {} if twin is None else {twin.cohort: twin}

    def tw(cohort: str) -> Twin:
        if cohort not in COHORTS:
            raise HTTPException(404, "unknown cohort")
        if cohort not in twins:
            twins[cohort] = Twin(cohort)
        return twins[cohort]

    @app.get("/api/cohorts")
    def cohorts():
        return [{"id": k, "label": v["label"], "treatments": v["treatments"], "note": v["note"]}
                for k, v in COHORTS.items() if v["root"] is None or (PROJECT_ROOT / v["root"]).exists()]

    @app.get("/api/meta")
    def meta(cohort: str = "physionet_b"):
        t = tw(cohort)
        return {"max_hour": t.max_hour, "threshold": t.thr, "patients": len(t.stays),
                "whatif": t.engine is not None, "note": t.meta_c["note"]}

    @app.get("/api/ward")
    def ward(clock: int = Query(24, ge=0), cohort: str = "physionet_b"):
        return tw(cohort).ward(clock)

    @app.get("/api/patients/{sid}")
    def patient(sid: int, clock: int = Query(24, ge=0), show_outcome: bool = False, cohort: str = "physionet_b"):
        return tw(cohort).patient(sid, clock, show_outcome)

    @app.get("/api/patients/{sid}/whatif")
    def whatif(sid: int, clock: int = Query(24, ge=0), cohort: str = "physionet_b",
               fluid_ml: float | None = Query(None, ge=0, le=2000), ne_dose: float | None = Query(None, ge=0, le=5)):
        return tw(cohort).whatif(sid, clock, fluid_ml, ne_dose)

    @app.get("/api/model-card")
    def model_card(cohort: str = "physionet_b"):
        card = tw(cohort).model_card()
        rep = PROJECT_ROOT / "data" / "whatif" / "whatif_v2_report.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            card["whatif"] = {
                "method": "Usual-care course learned from eICU demo + MIMIC-IV demo (gradient boosting, conformal "
                          "90% intervals); plan effects from published studies.",
                "effects": [{"outcome": o.upper(), "treatment": k, "effect": v["used_mean"], "sd": v["used_sd"],
                             "source": v["source"]} for o, e in r["effects"].items() for k, v in e.items()],
                "validation": r["A_factual"], "semi_synthetic": r["B_semi_synthetic"],
            }
        return card

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


if __name__ == "__main__":
    uvicorn.run(create_app(), host="127.0.0.1", port=8050)
