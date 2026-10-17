"""Train and validate the hybrid what-if engine (v2) on REAL treatment data.

Data: eICU-CRD demo (open, 186 hospitals) + MIMIC-IV demo (open) -> hourly panel of
MAP, HR, lactate, IV fluid volume and norepinephrine-equivalent dose.

Validation:
  A. Factual forecasting on held-out stays: MAE 1-6 h ahead vs carrying the last value
     forward, overall and in hours where treatment changed; conformal 90 % coverage.
  B. Semi-synthetic recovery (true effects known): DML vs a naive regression.
  C. Literature agreement: data-only effect estimates vs published effects.

  python scripts/run_whatif_v2.py
"""
import json
import pickle

import numpy as np
import pandas as pd

from sdt.config import PROJECT_ROOT
from sdt.whatif.hybrid import D_UNIT, F_UNIT, HybridEngine, _prep, dml_effects
from sdt.whatif.simulator import World, simulate

OUT = PROJECT_ROOT / "data" / "whatif"


def panel_from(root, ne_col, fluid_col, iv=None, prefix=""):
    tl = pd.read_parquet(root / "lakes" / "timeline")
    st = pd.read_parquet(root / "lakes" / "lake3_tertiary")
    df = pd.DataFrame({
        "map": tl["map"].where(tl["map__mask"] == 1),
        "hr": tl["heart_rate"].where(tl["heart_rate__mask"] == 1),
        "lactate": tl["lactate"]}, index=tl.index)
    src = iv if iv is not None else tl
    df["ne_dose"] = src[ne_col].reindex(df.index).fillna(0).clip(0, 5)
    df["fluid_ml"] = src[fluid_col].reindex(df.index).fillna(0).clip(0, 5000)
    df = df.reset_index()
    df["age"] = df["stay_id"].map(st["age"])
    df["stay_id"] = prefix + df["stay_id"].astype(str)
    return df


def build_panel() -> pd.DataFrame:
    e_root = PROJECT_ROOT / "data" / "eicu_demo"
    eicu = panel_from(e_root, "ne_dose", "fluid_ml", iv=pd.read_parquet(e_root / "lakes" / "interventions"), prefix="eicu-")
    parts = [eicu.assign(source="eICU demo")]
    m_root = PROJECT_ROOT / "data" / "processed"
    if (m_root / "lakes" / "timeline").exists():
        parts.append(panel_from(m_root, "ne_equivalent", "iv_fluid_ml", prefix="mimic-").assign(source="MIMIC-IV demo"))
    return pd.concat(parts, ignore_index=True)


def split(panel, seed=0):
    rng = np.random.default_rng(seed)
    treated = panel.groupby("stay_id").apply(lambda d: (d.ne_dose.max() > 0) or (d.fluid_ml.max() >= 250))
    part = {}
    for flag in (False, True):
        ids = treated[treated == flag].index.to_numpy(); rng.shuffle(ids)
        n = len(ids)
        part.update({i: "train" for i in ids[:int(.6 * n)]})
        part.update({i: "calib" for i in ids[int(.6 * n):int(.8 * n)]})
        part.update({i: "test" for i in ids[int(.8 * n):]})
    s = panel["stay_id"].map(part)
    return panel[s == "train"], panel[s == "calib"], panel[s == "test"]


def factual_validation(eng: HybridEngine, test: pd.DataFrame, n_origins=800, seed=1) -> dict:
    """Forecast the next hours of held-out stays given the treatments actually given, three ways:
    baseline course only, + published effects (what the engine uses), + data (DML) effects."""
    rng = np.random.default_rng(seed)
    test = _prep(test)
    H = eng.horizon
    rows = []
    for sid in rng.choice(test["stay_id"].unique(), min(n_origins, test["stay_id"].nunique()), replace=False):
        d = test[test["stay_id"] == sid].reset_index(drop=True)
        if len(d) < H + 4:
            continue
        t = int(rng.integers(3, len(d) - H))
        fut = d.iloc[t + 1:t + 1 + H]
        ne_now = float(d["ne_dose"].iloc[t])
        changed = bool((fut["fluid_ml"] >= 250).any() or (fut["ne_dose"] - ne_now).abs().gt(0.02).any())
        base = eng.baseline(d.iloc[:t + 1], H)
        for o in ("map", "hr"):
            last = d[o].iloc[:t + 1].dropna()
            if last.empty:
                continue
            pub, _ = eng.effect_path(o, ne_now, fut["ne_dose"].tolist(), fut["fluid_ml"].tolist())
            e = eng.effects[o]
            dne = (fut["ne_dose"].to_numpy() - ne_now) / D_UNIT
            fl = fut["fluid_ml"].to_numpy() / F_UNIT
            dat = e["ne"].data_mean * dne + e["fluid"].data_mean * fl
            for k in range(H):
                y = fut[o].iloc[k]
                if pd.notna(y):
                    rows.append({"outcome": o, "h": k + 1, "treatment_changed": changed,
                                 "err_published": abs(base[o][k] + pub[k] - y),
                                 "err_baseline_only": abs(base[o][k] - y),
                                 "err_data_effects": abs(base[o][k] + dat[k] - y),
                                 "err_carry_forward": abs(last.iloc[-1] - y),
                                 "covered": abs(base[o][k] - y) <= eng.resid_q90[o][k]})
    r = pd.DataFrame(rows)
    out = {}
    for (o, ch), g in r.groupby(["outcome", "treatment_changed"]):
        key = f"{o}_{'treatment_changed' if ch else 'no_change'}"
        out[key] = {"n": int(len(g)),
                    "mae_engine_published_effects": round(g["err_published"].mean(), 2),
                    "mae_baseline_only": round(g["err_baseline_only"].mean(), 2),
                    "mae_data_effects": round(g["err_data_effects"].mean(), 2),
                    "mae_carry_forward": round(g["err_carry_forward"].mean(), 2),
                    "coverage_90": round(g["covered"].mean(), 3),
                    "mae_by_hour_engine": g.groupby("h")["err_published"].mean().round(2).tolist()}
    return out


def semi_synthetic() -> dict:
    """Simulated world with known immediate effects: +3 mmHg per 500 mL fluid in the same hour,
    +6 mmHg per 0.1 mcg/kg/min norepinephrine while running; treatment is confounded by severity."""
    world = World()
    sim = simulate(3000, 40, world, seed=5)
    sim = sim.sort_values(["stay_id", "hour"])
    # the vasopressor state that acts on MAP in hour t is the one decided at the end of hour t-1
    sim["ne_dose"] = sim.groupby("stay_id")["vaso"].shift(1).fillna(0) * D_UNIT
    sim["fluid_ml"] = sim["fluid"] * F_UNIT
    sim = sim.assign(age=60.0)
    theta, cov = dml_effects(sim, "map", design="same_hour")
    theta_l, cov_l = dml_effects(sim, "map", design="lagged")
    # naive: next-hour MAP regressed on the treatments only (ignores why they were given)
    g = sim.groupby("stay_id")
    y = g["map"].shift(-1); F = g["fluid_ml"].shift(-1) / F_UNIT; D = (g["ne_dose"].shift(-1) - sim["ne_dose"]) / D_UNIT
    ok = y.notna() & F.notna() & D.notna()
    A = np.column_stack([np.ones(ok.sum()), F[ok], D[ok]])
    naive = np.linalg.lstsq(A, y[ok], rcond=None)[0][1:]
    return {"true": {"fluid_per_500ml": world.fluid_map_effect, "ne_per_0.1": world.vaso_map_effect},
            "dml_same_hour": {"fluid_per_500ml": float(theta[0]), "fluid_se": float(np.sqrt(cov[0, 0])),
                              "ne_per_0.1": float(theta[1]), "ne_se": float(np.sqrt(cov[1, 1]))},
            "dml_lagged": {"fluid_per_500ml": float(theta_l[0]), "ne_per_0.1": float(theta_l[1])},
            "naive": {"fluid_per_500ml": float(naive[0]), "ne_per_0.1": float(naive[1])}}


def examples(eng: HybridEngine, test: pd.DataFrame) -> list[dict]:
    test = _prep(test)
    out = []
    on_ne = test[test["ne_dose"] > 0.05]["stay_id"].unique()
    hypo = test[(test["map"] < 65) & (test["ne_dose"] == 0)]["stay_id"].unique()
    enough = test.groupby("stay_id")["map"].count()
    enough = enough[enough >= 8].index
    test = test[test["stay_id"].isin(enough)]
    on_ne = [s for s in on_ne if s in set(enough)]
    hypo = [s for s in hypo if s in set(enough)]
    healthy = test.groupby("stay_id")["map"].median().sort_values().index[-1:]
    cases = [("on vasopressor: +0.05 dose", on_ne[:1], lambda ne: (0.0, ne + 0.05)),
             ("hypotensive: 500 mL bolus", hypo[:1], lambda ne: (500.0, ne)),
             ("normotensive: start vasopressor 0.3", healthy, lambda ne: (0.0, 0.3))]
    for name, ids, plan in cases:
        if not len(ids):
            continue
        d = test[test["stay_id"] == ids[0]].reset_index(drop=True)
        cand = d.index[d["map"].notna() & (d.index >= 3)]
        if name.startswith("on vasopressor"):
            cand = cand[d.loc[cand, "ne_dose"] > 0.05]
        if not len(cand):
            continue
        t = int(cand[0])
        fl, ne = plan(float(d["ne_dose"].iloc[t]))
        r = eng.query(d.iloc[:t + 1], fl, ne)
        item = {"case": name, "stay": ids[0], "answered": r["answered"]}
        if r["answered"]:
            m = r["trajectories"]["map"]
            item["map_difference_mmHg_by_hour"] = m["difference"].round(1).tolist()
            item["map_plan_interval_h1"] = [round(m["plan_low"].iloc[0], 1), round(m["plan_high"].iloc[0], 1)]
        else:
            item["refusal"] = r["reasons"]
        out.append(item)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    tr, ca, te = split(panel)
    print({"stays": panel.stay_id.nunique(), "train": tr.stay_id.nunique(), "calib": ca.stay_id.nunique(),
           "test": te.stay_id.nunique(), "vasopressor_hours": int((panel.ne_dose > 0).sum()),
           "bolus_hours": int((panel.fluid_ml >= 250).sum())}, flush=True)
    eng = HybridEngine().fit(tr, ca)
    report = {
        "data": {"stays": int(panel.stay_id.nunique()), "by_source": panel.groupby("source").stay_id.nunique().to_dict(),
                 "vasopressor_hours": int((panel.ne_dose > 0).sum()), "bolus_hours_ge_250ml": int((panel.fluid_ml >= 250).sum())},
        "effects": {o: {k: v.as_dict() for k, v in e.items()} for o, e in eng.effects.items()},
        "conformal_q90": {o: v.round(2).tolist() for o, v in eng.resid_q90.items()},
        "support": eng.support,
    }
    print("A. factual validation", flush=True)
    report["A_factual"] = factual_validation(eng, te)
    print(json.dumps(report["A_factual"], indent=1), flush=True)
    print("B. semi-synthetic", flush=True)
    report["B_semi_synthetic"] = semi_synthetic()
    print(report["B_semi_synthetic"], flush=True)
    real_same = {o: dml_effects(_prep(tr), o, design="same_hour")[0] for o in ("map", "hr")}
    report["C_literature"] = {o: {k: {"data_lagged": round(v.data_mean, 2),
                                      "data_same_hour": round(float(real_same[o][0 if k == "fluid" else 1]), 2), "data_ci95": [round(v.data_mean - 1.96 * v.data_se, 2),
                                                                                    round(v.data_mean + 1.96 * v.data_se, 2)],
                                      "published": v.prior_mean, "source": v.prior_source}
                                  for k, v in e.items()} for o, e in eng.effects.items()}
    report["examples"] = examples(eng, te)
    print(json.dumps(report["examples"], indent=1, default=str), flush=True)
    (OUT / "whatif_v2_report.json").write_text(json.dumps(report, indent=2, default=str))
    with open(OUT / "engine_v2.pkl", "wb") as f:
        pickle.dump(eng, f)


if __name__ == "__main__":
    main()
