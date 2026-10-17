"""Phases 1-2 on the open PhysioNet/CinC 2019 data (see config/physionet2019.yaml).

  python scripts/run_physionet_pipeline.py [--limit N]
"""
import argparse
import json
import logging

from sdt.config import load_config
from sdt.pipeline_physionet import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="first N stays only (smoke test)")
    ap.add_argument("--batch-size", type=int, default=4000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(json.dumps(run(load_config(), batch_size=args.batch_size, limit=args.limit), indent=2))
