"""eICU demo -> same gate, derivations and lakes as PhysioNet 2019, plus the treatment panel.

  python scripts/run_eicu_pipeline.py      # reads data/raw/eicu-demo, writes data/eicu_demo
"""
import json
import logging

import pandas as pd

from sdt.config import PROJECT_ROOT, Config, load_config
from sdt.lake.store import LakeWriter
from sdt.pipeline_physionet import load_adapter, run_batch
from sdt.sources import eicu
from sdt.sources import physionet2019 as pn

RAW = PROJECT_ROOT / "data" / "raw" / "eicu-demo"
OUT = PROJECT_ROOT / "data" / "eicu_demo"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    cfg = Config({**cfg.settings, "output_root": str(OUT)}, cfg.parameters, cfg.interventions)
    adapter = load_adapter()
    raw = eicu.hourly_frame(RAW)
    (OUT / "staging").mkdir(parents=True, exist_ok=True)
    raw.to_parquet(OUT / "staging" / "raw.parquet", index=False)
    stays = pn.stays_table(raw)
    prim, sec = pn.specs_for(cfg.parameters, adapter)
    writer = LakeWriter(OUT, cfg.version)
    res = run_batch(raw, stays, cfg, adapter, prim, sec, writer, 0)
    panel = eicu.treatment_panel(RAW)
    writer.write("interventions", panel, 0)
    treated = panel.groupby(level="stay_id").agg(ne=("ne_dose", "max"), fl=("fluid_ml", "max"))
    summary = {"icu_stays": len(stays), "icu_hours": res["timeline_rows"],
               "hospitals": int(pd.read_csv(RAW / "patient.csv.gz", usecols=["hospitalid"])["hospitalid"].nunique()),
               "stays_with_vasopressor": int((treated["ne"] > 0).sum()),
               "stays_with_iv_fluid": int((treated["fl"] > 0).sum()),
               "hours_with_vasopressor": int((panel["ne_dose"] > 0).sum()),
               "hours_with_fluid_ge_250ml": int((panel["fluid_ml"] >= 250).sum())}
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / "cohort_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
