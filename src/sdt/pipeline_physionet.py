"""Phases 1-2 on PhysioNet/CinC 2019: gate -> derivations -> lakes -> labels.

Same lake layout and timeline format as the MIMIC-IV pipeline, so Phases 3-4
read either source. Parameters the challenge does not record (GCS, PaO2, urine
output, treatments, ...) are present as all-missing columns with mask = 0.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import yaml

from sdt import schema
from sdt.config import PROJECT_ROOT, Config
from sdt.derive import hourly, layer1, layer2
from sdt.labels import sepsis3
from sdt.lake.store import LakeWriter
from sdt.quality import gate
from sdt.sources import physionet2019 as pn

log = logging.getLogger(__name__)

# Inputs the shared derivations read; absent ones are added as NaN.
_EXPECTED_MEASURED = ["gcs_eye", "gcs_verbal", "gcs_motor", "o2_flow", "pao2", "neutrophils_pct",
                      "lymphocytes_pct", "sodium", "albumin", "inr", "pt", "d_dimer"]
_EXPECTED_IV = ["vaso_norepinephrine", "vaso_epinephrine", "vaso_dopamine", "vaso_dobutamine",
                "vaso_phenylephrine", "vaso_vasopressin", "vent_invasive", "vent_noninvasive", "urine_ml"]


def load_adapter() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "config" / "physionet2019.yaml").read_text())


def _complete_columns(h: pd.DataFrame) -> pd.DataFrame:
    add = {}
    for p in _EXPECTED_MEASURED:
        for suf in ("", "__min", "__max"):
            if f"{p}{suf}" not in h:
                add[f"{p}{suf}"] = np.nan
    for c in _EXPECTED_IV:
        if c not in h:
            add[c] = np.nan          # unknown, NOT zero: no treatment data in this source
    return pd.concat([h, pd.DataFrame(add, index=h.index)], axis=1) if add else h


def run_batch(raw: pd.DataFrame, stays: pd.DataFrame, cfg: Config, adapter: dict,
              prim: dict, sec: dict, writer: LakeWriter, batch: int) -> dict:
    specs = {**prim, **sec}
    meas = pn.long_measurements(raw, adapter["columns"])
    meas["unit"] = meas["param"].map({k: v["unit"] for k, v in specs.items()})
    meas.loc[(meas["param"] == "fio2") & (meas["value"] > 1), "value"] /= 100
    audit = gate.run_gate(meas, specs, stays)
    lake = gate.to_lake_rows(audit, cfg.version)
    lake1 = lake[lake["param"].isin(prim)]
    lake2 = lake[lake["param"].isin(sec)]

    h = hourly.build_hourly(lake, stays, specs)
    h = _complete_columns(h)
    dcfg = {**cfg.parameters["derived"], "assume_room_air": adapter["assume_room_air"]}
    d1 = layer1.derive_layer1(h, dcfg)
    d2 = layer2.derive_layer2(h, d1, lake2, stays, dcfg)
    he = hourly.icu_hours(h)["hour_end"]
    lake1 = pd.concat([lake1, writer.derived_rows(hourly.icu_hours(d1), layer1.LAYER1_META, he)], ignore_index=True)
    lake2 = pd.concat([lake2, writer.derived_rows(hourly.icu_hours(d2), layer2.LAYER2_META, he)], ignore_index=True)

    onsets = pn.onset_hours(raw, adapter["label_shift_h"]).reindex(stays["stay_id"])
    icu = hourly.icu_hours(h)
    labels = sepsis3.hourly_labels(icu.index, onsets, pd.Series(dtype=float),
                                   cfg.settings["labels"]["horizons_h"])
    labels = labels.drop(columns=[c for c in labels if "shock" in c])   # no treatment data -> no shock label

    st = stays.set_index("stay_id")
    static = pd.DataFrame({"age": st["age"], "female": (st["gender"] == "F").astype("int8"),
                           "unit": st["unit"], "hosp_adm_time_h": st["hosp_adm_time_h"],
                           "hospital_set": st["hospital_set"]})
    for e in cfg.settings["environment"]["features"]:
        static[e] = np.nan
        static[f"{e}__mask"] = 0

    meas_cols = [f"{p}{s}" for p in specs for s in ("", "__mask", "__age")]
    dcols = [c for c in pd.concat([d1, d2], axis=1).columns if not c.endswith("__source")]
    dd = hourly.icu_hours(d1.join(d2))[dcols]
    dd = dd.loc[:, dd.notna().any()]          # drop derivations this source can never feed
    timeline = icu[["hour_end"] + meas_cols].join(dd).join(labels)

    outcomes = onsets.copy()
    for name, df in [("lake1_primary", lake1), ("lake2_secondary", lake2), ("lake3_tertiary", static),
                     ("outcomes", outcomes), ("labels", labels), ("timeline", timeline),
                     ("quality_removed", audit[audit["flag"].isin(schema.REMOVED_FLAGS)])]:
        writer.write(name, df, batch)
    return {"audit": audit, "outcomes": outcomes, "timeline_rows": len(timeline),
            "elig": labels["eligible_sepsis"].sum(), "pos6": labels.loc[labels.eligible_sepsis == 1, "y_sepsis_6h"].sum()}


def run(cfg: Config, batch_size: int = 4000, limit: int | None = None) -> dict:
    adapter = load_adapter()
    root, out_root = pn.adapter_paths(adapter)
    cfg = Config({**cfg.settings, "output_root": str(out_root)}, cfg.parameters, cfg.interventions)
    raw = pn.load_raw(root, out_root / "staging" / "raw.parquet")
    stays = pn.stays_table(raw)
    if limit:
        stays = stays.head(limit)
    prim, sec = pn.specs_for(cfg.parameters, adapter)
    writer = LakeWriter(out_root, cfg.version)
    log.info("PhysioNet 2019: %d stays (%s)", len(stays), stays["hospital_set"].value_counts().to_dict())

    res = []
    for b, i in enumerate(range(0, len(stays), batch_size)):
        st = stays.iloc[i:i + batch_size].reset_index(drop=True)
        r = raw[raw["stay_id"].isin(st["stay_id"])]
        res.append(run_batch(r, st, cfg, adapter, prim, sec, writer, b))
        log.info("batch %d done", b)

    audit = pd.concat([r["audit"] for r in res])
    outcomes = pd.concat([r["outcomes"] for r in res])
    rep = out_root / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    gate.gate_report(audit).to_csv(rep / "quality_gate.csv", index=False)
    s = stays.set_index("stay_id")
    summary = {
        "icu_stays": len(stays), "by_set": s["hospital_set"].value_counts().to_dict(),
        "icu_hours": int(sum(r["timeline_rows"] for r in res)),
        "median_age": float(s["age"].median()), "female_pct": round(100 * (s["gender"] == "F").mean(), 1),
        "sepsis_pct_of_stays": round(100 * outcomes["sepsis3"].mean(), 2),
        "sepsis_at_admission_pct": round(100 * outcomes["sepsis_at_admission"].mean(), 2),
        "sepsis_pct_by_set": outcomes.join(s["hospital_set"]).groupby("hospital_set")["sepsis3"].mean().mul(100).round(2).to_dict(),
        "positive_eligible_hours_6h_pct": round(100 * sum(r["pos6"] for r in res) / sum(r["elig"] for r in res), 3),
    }
    (rep / "cohort_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
