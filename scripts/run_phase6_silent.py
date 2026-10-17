"""Phase 6: silent-mode replay on held-out patients + causality check.

Replays external set-B stays hour by hour through the trained model, logging
would-be alerts without showing them (shadow mode), and verifies that no
prediction uses future data.

  python scripts/run_phase6_silent.py [--stays 1000]
"""
import argparse
import json
import pickle

import numpy as np
import pandas as pd

from sdt.deploy.silent import ShadowRunner
from sdt.model.data import build_stays, read_lake
from sdt.pipeline_physionet import load_adapter
from sdt.sources import physionet2019 as pn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stays", type=int, default=1000)
    args = ap.parse_args()
    _, root = pn.adapter_paths(load_adapter())
    runner = ShadowRunner(root / "models", root / "reports" / "phase3_4_report.json")
    sc = runner.schema

    static = read_lake(root, "lake3_tertiary")
    static = static.join(pd.get_dummies(static["unit"], prefix="unit", dtype=float))
    static["hosp_adm_time_h"] = static["hosp_adm_time_h"].clip(-2000, 0)
    for c in sc.static:
        if c not in static:
            static[c] = 0.0
    ids = static[static["hospital_set"] == "B"].sample(args.stays, random_state=0).index
    tl = read_lake(root, "timeline")
    tl = tl[tl.index.get_level_values("stay_id").isin(ids)]
    stays = build_stays(tl, static.loc[ids], sc, group_col="hospital_set")

    worst = runner.causality_check(stays[:60])
    log = runner.replay(stays, root / "reports" / "phase6_silent_log.csv")
    outcomes = read_lake(root, "outcomes")
    onset = log["stay_id"].map(outcomes["onset_hour"])
    pre = log[onset.notna() & (onset >= 0) & (log["hour"] < onset) & (log["hour"] >= onset - 12)]
    flagged = pre.groupby("stay_id")["would_alert_6h"].any()
    elig = log[log["eligible"]]
    summary = {
        "mode": "silent: no alerts shown, predictions logged only",
        "stays_replayed": len(stays), "hours": int(len(log)),
        "causality_max_abs_risk_difference": worst,
        "causality_ok": worst < 1e-4,
        "threshold_6h": runner.threshold["6h"],
        "would_be_alerts_per_patient_day": float(elig["would_alert_6h"].sum() / (len(elig) / 24)),
        "septic_stays_in_sample": int(flagged.size),
        "septic_flagged_in_12h_before_onset_pct": float(100 * flagged.mean()) if flagged.size else None,
        "dominant_stream_share": log["dominant_stream"].value_counts(normalize=True).round(3).to_dict(),
    }
    (root / "reports" / "phase6_silent_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
