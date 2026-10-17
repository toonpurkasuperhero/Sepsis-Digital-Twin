"""Phase 6: silent (shadow) deployment runner.

The model runs beside usual care and LOGS what it would have done; it never
shows an alert. The log is the raw material for the DECIDE-AI early clinical
evaluation (would-be alerts, lead time, alert burden, per-hour stream weights).

`causality_check` proves the live property the whole design depends on: the
risk at hour t is identical whether or not hours after t exist.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sdt.model.fusion import Batcher, FusionNet
from sdt.model.train import predict

TASKS = ["6h", "12h"]


class ShadowRunner:
    def __init__(self, model_dir: Path, report_path: Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        meta = pickle.loads((model_dir / "schema.pkl").read_bytes())
        self.schema, self.norm = meta["schema"], meta["norm"]
        sc = self.schema
        self.model = FusionNet(meta["n_e1"], len(sc.secondary), len(sc.derived2), len(sc.static),
                               len(sc.env), env_cap=meta["env_cap"], n_vitals=len(sc.primary)).to(self.device)
        self.model.load_state_dict(torch.load(model_dir / "fusion.pt", map_location=self.device))
        self.calib = pickle.loads((model_dir / "calibration.pkl").read_bytes())
        self.batcher = Batcher(self.norm, meta["max_age1"], self.device)
        rep = json.loads(report_path.read_text())
        self.threshold = {t: rep["phase4"][f"internal_test_{t}"]["threshold"] for t in TASKS}

    def risks(self, stays) -> list[np.ndarray]:
        probs, gates = predict(self.model, stays, self.batcher)
        out = []
        for p, g in zip(probs, gates):
            cal = np.stack([self.calib[t].predict(p[:, k]) for k, t in enumerate(TASKS)], 1)
            out.append((cal, g))
        return out

    def replay(self, stays, log_path: Path) -> pd.DataFrame:
        rows = []
        for s, (cal, g) in zip(stays, self.risks(stays)):
            for h in range(len(s.y)):
                rows.append({"stay_id": s.stay_id, "hour": h, "risk_6h": cal[h, 0], "risk_12h": cal[h, 1],
                             "would_alert_6h": bool(cal[h, 0] >= self.threshold["6h"]),
                             "dominant_stream": self.model.stream_names[int(np.argmax(g[h]))],
                             "eligible": bool(s.elig[h])})
        log = pd.DataFrame(rows)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(log_path, index=False)
        return log

    def causality_check(self, stays, n_cuts: int = 3, seed: int = 0) -> float:
        """Max |risk difference| at hour t between the full stay and the stay cut after t."""
        rng = np.random.default_rng(seed)
        full = predict(self.model, stays, self.batcher)[0]
        worst = 0.0
        for s, pf in zip(stays, full):
            if len(s.y) < 4:
                continue
            for t in rng.integers(1, len(s.y) - 1, n_cuts):
                cut = truncate(s, int(t))
                pc = predict(self.model, [cut], self.batcher)[0][0]
                worst = max(worst, float(np.abs(pc[t] - pf[t]).max()))
        return worst


def truncate(s, t: int):
    """The stay as it was known at the end of hour t."""
    keep = s.tok_t <= t
    return replace(s, x1=s.x1[:t + 1], d1=s.d1[:t + 1], x2=s.x2[:t + 1], d2=s.d2[:t + 1],
                   y=s.y[:t + 1], elig=s.elig[:t + 1],
                   tok_t=s.tok_t[keep], tok_p=s.tok_p[keep], tok_v=s.tok_v[keep])
