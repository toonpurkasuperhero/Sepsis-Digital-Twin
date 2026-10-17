"""Runs Phases 1-2: stage MIMIC-IV, quality gate, derived parameters, data lakes, labels.

  python scripts/run_pipeline.py                         # MIMIC-IV demo (config default)
  python scripts/run_pipeline.py --mimic-root /data/mimiciv/3.1 --output-root data/full
"""
import argparse
import json
import logging

from sdt.config import load_config
from sdt.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic-root", help="folder containing MIMIC-IV hosp/ and icu/")
    ap.add_argument("--output-root", help="where staging, lakes and reports are written")
    ap.add_argument("--batch-size", type=int, default=2000, help="ICU stays per batch")
    ap.add_argument("--limit", type=int, help="process only the first N stays (smoke test)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(mimic_root=args.mimic_root, output_root=args.output_root)
    print(json.dumps(run(cfg, batch_size=args.batch_size, limit=args.limit), indent=2))


if __name__ == "__main__":
    main()
