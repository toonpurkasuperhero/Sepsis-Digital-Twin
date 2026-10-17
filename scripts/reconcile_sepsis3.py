"""Phase 1 gate: compare our Sepsis-3 labels with the MIT-LCP mimic-code table.

Export the reference table once (BigQuery `physionet-data.mimiciv_derived.sepsis3`,
or the Postgres build of mimic-code) to CSV with columns stay_id,
suspected_infection_time, sofa_time, sepsis3, then:

  python scripts/reconcile_sepsis3.py --reference sepsis3.csv [--output-root data/processed]
"""
import argparse
import json

import pandas as pd

from sdt.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output-root")
    args = ap.parse_args()
    cfg = load_config(output_root=args.output_root)

    ours = pd.read_parquet(cfg.output_root / "lakes" / "outcomes")
    ref = pd.read_csv(args.reference, parse_dates=["suspected_infection_time", "sofa_time"])
    ref = ref[ref["sepsis3"].astype(bool)].groupby("stay_id").first()

    both = pd.DataFrame(index=ours.index)
    both["ours"] = ours["sepsis3"].astype(bool)
    both["ref"] = both.index.isin(ref.index)
    tp = int((both.ours & both.ref).sum())
    agree = {
        "stays": len(both),
        "ours_positive_pct": round(100 * both.ours.mean(), 1),
        "reference_positive_pct": round(100 * both.ref.mean(), 1),
        "agreement_pct": round(100 * (both.ours == both.ref).mean(), 1),
        "cohens_kappa": round(_kappa(both.ours, both.ref), 3),
        "only_ours": int((both.ours & ~both.ref).sum()),
        "only_reference": int((~both.ours & both.ref).sum()),
        "both": tp,
    }
    common = ours.join(ref, how="inner")
    diff_h = (common["suspicion_time"] - common["suspected_infection_time"]).dt.total_seconds() / 3600
    agree["median_abs_suspicion_time_diff_h"] = float(diff_h.abs().median()) if len(diff_h) else None
    # Gate: agreement on positives within a few points, kappa >= 0.8 (set with the team).
    print(json.dumps(agree, indent=2))


def _kappa(a: pd.Series, b: pd.Series) -> float:
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


if __name__ == "__main__":
    main()
