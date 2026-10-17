"""Phases 3-4 on the PhysioNet 2019 lakes.

Phase 3: causal imputer + mask-and-recover validation; encoder pre-training.
Phase 4: fusion training on set A (patient-level 70/15/15 split), isotonic
calibration and threshold on the validation split, evaluation on the internal
test split and EXTERNALLY on set B (a different hospital system); GBM baseline,
stream importance (gates + ablation), subgroups, imputation downstream test.

  python scripts/run_phase3_4.py [--epochs 20] [--quick]
"""
import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from sdt.config import PROJECT_ROOT, load_config
from sdt.model import baseline, evaluate, imputer, train
from sdt.model.data import Normalizer, build_stays, make_schema, read_lake
from sdt.model.fusion import Batcher, FusionNet
from sdt.pipeline_physionet import load_adapter
from sdt.sources import physionet2019 as pn

TASKS = ["6h", "12h"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="3000-stay subset, for a smoke test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config()
    adapter = load_adapter()
    _, root = pn.adapter_paths(adapter)
    out = root / "models"; out.mkdir(parents=True, exist_ok=True)
    rep = root / "reports"; rep.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ data
    tl = read_lake(root, "timeline")
    static = read_lake(root, "lake3_tertiary")
    outcomes = read_lake(root, "outcomes")
    if args.quick:
        keep = static.groupby("hospital_set").head(1500).index
        static, outcomes = static.loc[keep], outcomes.loc[keep]
        tl = tl[tl.index.get_level_values("stay_id").isin(keep)]
    static = static.join(pd.get_dummies(static["unit"], prefix="unit", dtype=float))
    static["hosp_adm_time_h"] = static["hosp_adm_time_h"].clip(-2000, 0)
    prim, sec = pn.specs_for(cfg.parameters, adapter)
    max_age = {k: v.get("max_age_h", 24) for k, v in {**prim, **sec}.items()}
    static_cols = ["age", "female", "hosp_adm_time_h"] + [c for c in static if c.startswith("unit_")]
    env = cfg.settings["environment"]["features"]
    sc = make_schema(tl, list(prim), list(sec), static_cols, env, max_age)
    log(f"schema: {len(sc.primary)} vitals, {len(sc.secondary)} labs, {len(sc.derived1)} L1-derived, "
        f"{len(sc.derived2)} L2-derived, {len(sc.static)} static, {len(sc.env)} env")
    stays = build_stays(tl, static, sc, group_col="hospital_set")
    del tl
    log(f"{len(stays)} stays built")

    # patient-level split of set A, stratified by sepsis; set B is external
    rng = np.random.default_rng(args.seed)
    A = [s for s in stays if s.group == "A"]; B = [s for s in stays if s.group == "B"]
    sep = outcomes["sepsis3"]
    split = {}
    for lab in (0, 1):
        ids = np.array([s.stay_id for s in A if sep.get(s.stay_id, 0) == lab]); rng.shuffle(ids)
        n = len(ids)
        split.update({i: "train" for i in ids[:int(.7 * n)]})
        split.update({i: "val" for i in ids[int(.7 * n):int(.85 * n)]})
        split.update({i: "test" for i in ids[int(.85 * n):]})
    tr = [s for s in A if split[s.stay_id] == "train"]
    va = [s for s in A if split[s.stay_id] == "val"]
    te = [s for s in A if split[s.stay_id] == "test"]
    log(f"split: train {len(tr)}, val {len(va)}, internal test {len(te)}, external (set B) {len(B)}")
    norm = Normalizer.fit(tr)
    max_age1 = np.array([sc.max_age[p] for p in sc.primary], np.float32)
    max_age2 = np.array([sc.max_age[p] for p in sc.secondary], np.float32)
    report = {"data": {"train": len(tr), "val": len(va), "internal_test": len(te), "external_test": len(B),
                       "schema": sc.__dict__}}

    # ------------------------------------------------------------ Phase 3: imputer
    log("Phase 3: training causal imputer")
    imp = imputer.train_imputer(tr, va, norm, max_age1, dev, log=log)
    imp_rep = imputer.mask_and_recover(imp, te, norm, max_age1, sc.primary, dev)
    imp_rep.to_csv(rep / "phase3_imputation.csv", index=False)
    g = imp_rep[imp_rep.method == "causal_gru"]
    piv = imp_rep.pivot_table(index=["pattern", "param"], columns="method", values="mae")
    report["phase3"] = {
        "coverage_90_median": float(g["coverage_90"].median()),
        "coverage_90_range": [float(g["coverage_90"].min()), float(g["coverage_90"].max())],
        "gru_beats_locf_share": float((piv["causal_gru"] < piv["locf"]).mean()),
        "mae_ratio_gru_over_locf_median": float((piv["causal_gru"] / piv["locf"]).median()),
    }
    torch.save(imp.state_dict(), out / "imputer.pt")
    log(f"Phase 3 imputation: {report['phase3']}")

    # ------------------------------------------------------------ Phase 4: fusion
    batcher = Batcher(norm, max_age1, dev)
    n_e1 = batcher.e1_features(tr[0]).shape[1]
    model = FusionNet(n_e1, len(sc.secondary), len(sc.derived2), len(sc.static), len(sc.env),
                      env_cap=cfg.settings["environment"]["max_contribution_logit"],
                      n_vitals=len(sc.primary)).to(dev)
    log("Phase 3: encoder pre-training (next-hour forecasting)")
    train.pretrain_encoders(model, tr, batcher, epochs=2 if args.quick else 3, log=log)
    y_tr = np.concatenate([s.y[s.elig == 1] for s in tr])
    pw = torch.tensor(np.sqrt((1 - y_tr.mean(0)) / y_tr.mean(0)), dtype=torch.float32, device=dev)
    log(f"Phase 4: fusion training (pos_weight {pw.tolist()})")
    hist = train.train_fusion(model, tr, va, batcher, pw, cfg.settings["environment"]["l2_penalty"],
                              epochs=3 if args.quick else args.epochs, log=log)
    report["training"] = hist
    torch.save(model.state_dict(), out / "fusion.pt")

    onsets = outcomes["onset_hour"].where(outcomes["onset_hour"] >= 0)
    sets = {"val": va, "internal_test": te, "external_test_setB": B}
    preds = {k: train.predict(model, v, batcher) for k, v in sets.items()}
    report["phase4"] = {}
    calib = {}
    for k_task, task in enumerate(TASKS):
        dv = evaluate.flatten(va, preds["val"][0], k_task)
        iso = evaluate.fit_isotonic(dv)
        thr_raw = evaluate.threshold_for_sensitivity(dv["y"].to_numpy(), iso.predict(dv["p"]), 0.8)
        calib[task] = iso
        for name in ("internal_test", "external_test_setB"):
            df = evaluate.flatten(sets[name], preds[name][0], k_task)
            report["phase4"][f"{name}_{task}"] = evaluate.full_report(df, iso, thr_raw, onsets,
                                                                     n_boot=50 if args.quick else 200)
            log(f"{name} {task}: AUROC {report['phase4'][f'{name}_{task}']['auroc']:.3f} "
                f"AUPRC {report['phase4'][f'{name}_{task}']['auprc']:.3f}")
    with open(out / "calibration.pkl", "wb") as f:
        pickle.dump(calib, f)

    # stream importance: mean gate weight + ablation (external set, 6 h)
    dfe = evaluate.flatten(B, preds["external_test_setB"][0], 0)
    base_auc = roc_auc_score(dfe["y"], dfe["p"])
    gates = np.concatenate([gt[s.elig == 1] for s, gt in zip(B, preds["external_test_setB"][1])])
    imp_rows = []
    for k, nm in enumerate(model.stream_names):
        pa, _ = train.predict(model, B, batcher, drop=nm)
        auc = roc_auc_score(dfe["y"], evaluate.flatten(B, pa, 0)["p"])
        imp_rows.append({"stream": nm, "mean_gate_weight": float(gates[:, k].mean()),
                         "auroc_drop_when_removed": float(base_auc - auc)})
    report["stream_importance_external_6h"] = imp_rows
    env_max = float(model.env_contribution(torch.ones(1, len(sc.env), device=dev) * 3).abs().max().detach())
    report["environment_branch"] = {"cap_logit": model.env_cap, "max_abs_contribution_at_3sd": env_max,
                                    "note": "inputs masked (not linkable in this dataset)"}

    # subgroups (external, 6 h)
    st = static
    age_band = pd.cut(st["age"], [0, 50, 65, 80, 200], labels=["<50", "50-64", "65-79", "80+"])
    sub = pd.concat([evaluate.subgroup_auroc(dfe, st["female"].map({0: "male", 1: "female"}), "sex"),
                     evaluate.subgroup_auroc(dfe, age_band.astype(str), "age"),
                     evaluate.subgroup_auroc(dfe, st["unit"], "unit")])
    sub.to_csv(rep / "phase4_subgroups.csv", index=False)

    # imputation downstream test (internal test, 6 h): hide 15 % of observed vitals
    rng2 = np.random.default_rng(7)
    hidden = [np.where(imputer.hide(s.x1, "mcar", rng2), np.nan, s.x1) for s in te]
    p_locf, _ = train.predict(model, te, batcher, x1_override=hidden)
    m_norm, _ = imputer.impute(imp, [norm("x1", h) for h in hidden], max_age1, dev)
    p_gru, _ = train.predict(model, te, batcher, x1_override=hidden, fills=m_norm)
    dft = evaluate.flatten(te, preds["internal_test"][0], 0)
    a_full = roc_auc_score(dft["y"], dft["p"])
    a_locf = roc_auc_score(dft["y"], evaluate.flatten(te, p_locf, 0)["p"])
    a_gru = roc_auc_score(dft["y"], evaluate.flatten(te, p_gru, 0)["p"])
    report["phase3"]["downstream_auroc_6h"] = {"all_observed": a_full, "15pct_hidden_carry_forward": a_locf,
                                               "15pct_hidden_gru_filled": a_gru,
                                               "delta_carry_forward": a_full - a_locf,
                                               "delta_gru": a_full - a_gru}

    # ------------------------------------------------------------ GBM baseline
    log("baseline: gradient boosting")
    Xtr, ytr, _ = baseline.hourly_matrix(tr, batcher, max_age2)
    gbm = {}
    for k_task, task in enumerate(TASKS):
        clf = baseline.train_gbm(Xtr, ytr[:, k_task], args.seed)
        Xv, yv, iv = baseline.hourly_matrix(va, batcher, max_age2)
        dv = pd.DataFrame({"stay_id": iv, "y": yv[:, k_task], "p": clf.predict_proba(Xv)[:, 1]})
        iso = evaluate.fit_isotonic(dv)
        thr = evaluate.threshold_for_sensitivity(dv["y"].to_numpy(), iso.predict(dv["p"]), 0.8)
        for name in ("internal_test", "external_test_setB"):
            Xs, ys, ids = baseline.hourly_matrix(sets[name], batcher, max_age2)
            hours = np.concatenate([np.arange(len(s.y))[s.elig == 1] for s in sets[name]])
            df = pd.DataFrame({"stay_id": ids, "hour": hours, "y": ys[:, k_task], "p": clf.predict_proba(Xs)[:, 1]})
            gbm[f"{name}_{task}"] = evaluate.full_report(df, iso, thr, onsets, n_boot=50 if args.quick else 200)
            log(f"GBM {name} {task}: AUROC {gbm[f'{name}_{task}']['auroc']:.3f} AUPRC {gbm[f'{name}_{task}']['auprc']:.3f}")
    report["baseline_gbm"] = gbm

    # ------------------------------------------------------------ phase gates
    ext = report["phase4"]["external_test_setB_6h"]
    report["gates"] = {
        "phase3_interval_coverage_near_90": 0.85 <= report["phase3"]["coverage_90_median"] <= 0.95,
        "phase3_downstream_auroc_change_lt_0.01": abs(report["phase3"]["downstream_auroc_6h"]["delta_carry_forward"]) < 0.01,
        "phase4_external_auroc": ext["auroc"],
        "phase4_external_calibration_slope": ext["calibration_slope"],
        "phase4_external_alerts_per_patient_day": ext["alerts_per_patient_day"],
        "phase4_note": "AUROC/alert targets must be agreed with clinicians; numbers reported, not auto-passed",
    }
    with open(out / "schema.pkl", "wb") as f:
        pickle.dump({"schema": sc, "norm": norm, "max_age1": max_age1, "max_age2": max_age2,
                     "n_e1": n_e1, "env_cap": model.env_cap}, f)
    (rep / "phase3_4_report.json").write_text(json.dumps(report, indent=2, default=float))
    log(f"done. gates: {json.dumps(report['gates'], default=float)}")


if __name__ == "__main__":
    main()
