"""Phase 1 + 2 pipeline: source -> quality gate -> derivations -> data lakes -> labels."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from sdt import schema
from sdt.config import Config
from sdt.derive import hourly, interventions, layer1, layer2, layer3
from sdt.labels import sepsis3
from sdt.lake.store import LakeWriter
from sdt.quality import gate
from sdt.sources.mimic import MimicSource

log = logging.getLogger(__name__)

TIMELINE_STATS = ("", "__mask", "__age")


def urine_rows(out_ev: pd.DataFrame, stays: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    uo = cfg.parameters["urine_output"]
    ev = out_ev[out_ev["itemid"].isin({*uo["itemids"], *uo["negative_itemids"]})].copy()
    ev["value"] = np.where(ev["itemid"].isin(uo["negative_itemids"]), -ev["value"], ev["value"])
    ev["param"] = "urine_output"
    ev["unit"] = "mL"
    spec = {"urine_output": {"hard_limits": uo["hard_limits"], "itemids": {}, "unit": "mL"}}
    return gate.run_gate(ev, spec, stays)


def hourly_urine(lake_uo: pd.DataFrame, stays: pd.DataFrame) -> pd.Series:
    ok = lake_uo[lake_uo["flag"].isin(schema.USABLE_FLAGS)]
    intime = ok["stay_id"].map(stays.set_index("stay_id")["intime"])
    ok = ok.assign(hour=hourly.hour_index(ok["charttime"], intime))
    return ok.groupby(["stay_id", "hour"])["value"].sum().clip(lower=0).rename("urine_ml")


def run_batch(src: MimicSource, cfg: Config, stays: pd.DataFrame, batch: int, writer: LakeWriter) -> dict:
    ids = stays["stay_id"].tolist()
    prim_specs, sec_specs = cfg.param_specs("primary"), cfg.param_specs("secondary")

    # ---- Layer 1 and 2 measured values through the quality gate
    chart = gate.map_and_convert(src.chartevents(ids), prim_specs)
    chart = gate.run_gate(chart, prim_specs, stays)
    labs = gate.map_and_convert(src.labevents(stays), sec_specs)
    labs = gate.run_gate(labs, sec_specs, stays, pre_icu_hours={p: 24 for p in sec_specs})
    uo = urine_rows(src.outputevents(ids), stays, cfg)
    audit = pd.concat([chart, labs, uo], ignore_index=True)

    lake1 = gate.to_lake_rows(pd.concat([chart, uo], ignore_index=True), cfg.version)
    lake2 = gate.to_lake_rows(labs, cfg.version)

    # ---- causal hourly timeline (includes 24 pre-ICU hours for lab carry-forward)
    h = hourly.build_hourly(pd.concat([lake1, lake2]), stays, cfg.all_specs())
    h = h.join(hourly_urine(lake1[lake1["param"] == "urine_output"], stays))
    hadm = stays["hadm_id"].tolist()
    rx = src.prescriptions(hadm)
    iv = interventions.build_interventions(src.inputevents(ids), src.procedureevents(ids), rx, stays,
                                           cfg.interventions)
    h = h.join(iv, how="left")
    h[iv.columns] = h[iv.columns].fillna(0)

    # ---- derivations
    d1 = layer1.derive_layer1(h, cfg.parameters["derived"])
    d2 = layer2.derive_layer2(h, d1, lake2, stays, cfg.parameters["derived"])
    lake1 = pd.concat([lake1, writer.derived_rows(hourly.icu_hours(d1), layer1.LAYER1_META,
                                                  hourly.icu_hours(h)["hour_end"])], ignore_index=True)
    lake2 = pd.concat([lake2, writer.derived_rows(hourly.icu_hours(d2), layer2.LAYER2_META,
                                                  hourly.icu_hours(h)["hour_end"])], ignore_index=True)

    # ---- labels
    lab_cfg = cfg.settings["labels"]
    abx = interventions.classify_drugs(rx, cfg.interventions)
    abx = abx[abx["drug_class"] == "antibiotic"]
    cultures = src.cultures(hadm)
    susp = sepsis3.suspicion_times(abx, cultures, lab_cfg)
    onsets = sepsis3.sepsis_onsets(d2["sofa_label"], h["hour_end"], stays, susp, lab_cfg)
    hh = h.join(d1).join(d2)
    shock = sepsis3.shock_onset_hour(hourly.icu_hours(hh), onsets, lab_cfg)
    onsets = onsets.join(shock)
    icu_idx = hourly.icu_hours(h).index
    labels = sepsis3.hourly_labels(icu_idx, onsets, onsets["shock_hour"], lab_cfg["horizons_h"])

    # ---- Layer 3 static
    static = layer3.static_features(stays, src.diagnoses(stays["subject_id"].unique()), cultures,
                                    cfg.settings["environment"]["features"])
    # Outcomes are kept OUT of lake3 so they can never be picked up as baseline features.
    outcomes = onsets.join(stays.set_index("stay_id")[["hospital_expire_flag"]])

    # ---- model-ready timeline: value, mask, age per measured parameter + derived + interventions
    meas_cols = [f"{p}{s}" for p in cfg.all_specs() for s in TIMELINE_STATS] + ["urine_ml"]
    derived_cols = [c for c in pd.concat([d1, d2], axis=1).columns if not c.endswith("__source")]
    timeline = hourly.icu_hours(hh[["hour_end"] + meas_cols + derived_cols + list(iv.columns)]).join(labels)

    for name, df in [("lake1_primary", lake1), ("lake2_secondary", lake2), ("lake3_tertiary", static), ("outcomes", outcomes),
                     ("interventions", hourly.icu_hours(iv)), ("labels", labels),
                     ("timeline", timeline),
                     ("quality_removed", audit[audit["flag"].isin(schema.REMOVED_FLAGS)])]:
        writer.write(name, df, batch)
    return {"audit": audit, "outcomes": outcomes, "timeline": timeline}


def run(cfg: Config, batch_size: int = 2000, limit: int | None = None) -> dict:
    src = MimicSource(cfg)
    src.stage()
    itemid_report = src.verify_itemids()
    stays = src.icu_stays()
    stays = stays.merge(src.admission_weight(stays["stay_id"]), on="stay_id", how="left")
    if limit:
        stays = stays.head(limit)
    writer = LakeWriter(cfg.output_root, cfg.version)
    log.info("cohort: %d ICU stays", len(stays))

    results = [run_batch(src, cfg, stays.iloc[i:i + batch_size].reset_index(drop=True), b, writer)
               for b, i in enumerate(range(0, len(stays), batch_size))]
    audit = pd.concat([r["audit"] for r in results])
    outcomes = pd.concat([r["outcomes"] for r in results])
    timeline = pd.concat([r["timeline"] for r in results])

    rep_dir = cfg.output_root / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    itemid_report.to_csv(rep_dir / "itemid_verification.csv", index=False)
    gate.gate_report(audit).to_csv(rep_dir / "quality_gate.csv", index=False)
    coverage = missingness(timeline, cfg)
    coverage.to_csv(rep_dir / "hourly_coverage.csv", index=False)
    summary = cohort_summary(stays, outcomes, timeline, cfg)
    (rep_dir / "cohort_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def missingness(tl: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for p in cfg.all_specs():
        rows.append({"param": p, "pct_hours_measured": round(100 * tl[f"{p}__mask"].mean(), 2),
                     "pct_hours_available_after_carry": round(100 * tl[p].notna().mean(), 2),
                     "median_age_h": tl[f"{p}__age"].median()})
    return pd.DataFrame(rows).sort_values("pct_hours_measured", ascending=False)


def cohort_summary(stays, outcomes, tl, cfg) -> dict:
    s = {"icu_stays": int(len(stays)), "patients": int(stays["subject_id"].nunique()),
         "icu_hours": int(len(tl)),
         "median_age": float(stays["age"].median()),
         "female_pct": round(100 * (stays["gender"] == "F").mean(), 1),
         "hospital_mortality_pct": round(100 * stays["hospital_expire_flag"].mean(), 1),
         "sepsis3_pct_of_stays": round(100 * outcomes["sepsis3"].mean(), 1),
         "sepsis_present_at_icu_admission_pct": round(100 * outcomes["sepsis_at_admission"].mean(), 1),
         "septic_shock_pct_of_stays": round(100 * outcomes["shock_hour"].notna().mean(), 1)}
    for H in cfg.settings["labels"]["horizons_h"]:
        el = tl[tl["eligible_sepsis"] == 1]
        s[f"positive_hours_sepsis_{H}h_pct"] = round(100 * el[f"y_sepsis_{H}h"].mean(), 3)
    return s
