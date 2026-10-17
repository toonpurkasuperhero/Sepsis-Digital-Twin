"""Phase 5: what-if engine validation.

1. Semi-synthetic benchmark (true effects known): G-computation engine vs a naive
   predictive model, for "give a fluid bolus every hour" and "start a vasopressor"
   versus "no new treatment", 6 h ahead, on MAP.
2. Positivity guardrail: a plan almost never seen in similar patients must be refused.
3. MIMIC-IV demo plumbing run on real treatment data: NOT clinically valid (100 patients).

  python scripts/run_phase5_whatif.py
"""
import json

import numpy as np
import pandas as pd

from sdt.config import PROJECT_ROOT, load_config
from sdt.whatif.engine import EngineConfig, WhatIfEngine, naive_effect
from sdt.whatif.simulator import World, simulate, true_effect

OUT = PROJECT_ROOT / "data" / "whatif"


def benchmark(report: dict, transition: str) -> None:
    world, T, start, H = World(), 40, 24, 6
    train = simulate(4000, T, world, seed=1)
    cfg = EngineConfig(obs=["map", "lactate", "hr"], treatments=["fluid", "vaso"], n_boot=5, n_samples=40,
                       transition=transition)
    eng = WhatIfEngine(cfg).fit(train)
    n_test = 60
    test = simulate(n_test, T, world, seed=2)
    hist = test[test.hour < start]
    none = {"fluid": 0, "vaso": 0}
    rows = []
    for name, plan in {"fluid_every_hour": {"fluid": 1, "vaso": 0}, "start_vasopressor": {"fluid": 0, "vaso": 1}}.items():
        truth = true_effect(n_test, T, world, 2, plan, none, start, "map", H)
        est, lo, hi, refused = [], [], [], 0
        for sid in range(n_test):
            r = eng.query(hist[hist.stay_id == sid], plan, none, H)
            if not r["answered"]:
                refused += 1; est.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            last = r["trajectories"]["map"].iloc[-1]
            est.append(last["difference_mean"]); lo.append(last["difference_p05"]); hi.append(last["difference_p95"])
        est, lo, hi = map(np.array, (est, lo, hi))
        naive = naive_effect(train, cfg, "map", H, hist[hist.hour == start - 1], plan, none)
        ok = ~np.isnan(est)
        rows.append({
            "plan": name, "transition_model": transition, "true_mean_effect_mmHg": float(truth.mean()),
            "gcomp_mean_effect_mmHg": float(np.nanmean(est)),
            "naive_mean_effect_mmHg": float(naive.mean()),
            "gcomp_abs_bias": float(abs(np.nanmean(est) - truth[ok].mean())),
            "naive_abs_bias": float(abs(naive.mean() - truth.mean())),
            "gcomp_interval_coverage_90": float(np.mean((truth[ok] >= lo[ok]) & (truth[ok] <= hi[ok]))),
            "refused": refused, "patients": n_test})
        print(rows[-1], flush=True)
    report.setdefault("semi_synthetic_benchmark", []).extend(rows)

    # positivity: the healthiest test patient, asked to start a vasopressor
    last = hist[hist.hour == start - 1].sort_values("map", ascending=False)
    sid = int(last.iloc[0]["stay_id"])
    strict = WhatIfEngine(EngineConfig(**{**cfg.__dict__, "min_propensity": 0.02}))
    strict.models, strict.propensity, strict._columns = eng.models, eng.propensity, eng._columns
    r = strict.query(hist[hist.stay_id == sid], {"fluid": 0, "vaso": 1}, none, H)
    report["positivity_check"] = {"patient_map_mmHg": float(last.iloc[0]["map"]), "answered": r["answered"],
                                  "propensity": r["propensity"], "reason": r.get("reason")}
    print(report["positivity_check"], flush=True)


def mimic_plumbing(report: dict) -> None:
    root = load_config().output_root / "lakes" / "timeline"
    if not root.exists():
        report["mimic_demo"] = "skipped: run scripts/run_pipeline.py first"
        return
    tl = pd.read_parquet(root).reset_index()
    df = pd.DataFrame({"stay_id": tl["stay_id"], "hour": tl["hour"], "map": tl["map_any"],
                       "lactate": tl["lactate"], "hr": tl["heart_rate"],
                       "fluid": (tl["iv_fluid_ml"] >= 250).astype(float),       # >= 250 mL in the hour
                       "vaso": (tl["ne_equivalent"] > 0).astype(float)})
    df = df.dropna(subset=["map", "hr"])
    df["lactate"] = df.groupby("stay_id")["lactate"].ffill()
    df = df.dropna()
    cfg = EngineConfig(obs=["map", "lactate", "hr"], treatments=["fluid", "vaso"], n_boot=3, n_samples=30)
    eng = WhatIfEngine(cfg).fit(df)
    sid = df.groupby("stay_id")["vaso"].max().idxmax()
    h = df[df.stay_id == sid].head(24)
    on_vaso = float(h["vaso"].iloc[-1])            # keep the current vasopressor status; vary fluids only
    r = eng.query(h, {"fluid": 1, "vaso": on_vaso}, {"fluid": 0, "vaso": on_vaso}, 6)
    report["mimic_demo"] = {
        "status": "PLUMBING ONLY - 100-patient demo, not clinically valid",
        "stays_used": int(df.stay_id.nunique()), "example_stay": int(sid), "answered": r["answered"],
        "propensity": r["propensity"],
        "map_difference_6h_mmHg": None if not r["answered"] else
            r["trajectories"]["map"].iloc[-1][["difference_mean", "difference_p05", "difference_p95"]].to_dict(),
        "reason": r.get("reason")}
    print(report["mimic_demo"], flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    for tm in ("gbm", "linear"):
        benchmark(rep, tm)
    mimic_plumbing(rep)
    (OUT / "phase5_report.json").write_text(json.dumps(rep, indent=2, default=float))
