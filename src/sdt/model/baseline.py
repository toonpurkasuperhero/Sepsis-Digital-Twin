"""Gradient-boosting baseline on flat hourly features, for comparison with FusionNet.

Uses the same causal inputs: carried vitals with mask and age, carried labs with
mask and age, derived values, 6 h vital trends and static data. Environment
inputs are left out of the baseline (their effect is capped only in FusionNet).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from sdt.model.data import AGE_CAP_H, locf_features
from sdt.model.fusion import Batcher


def hourly_matrix(stays, batcher: Batcher, max_age2: np.ndarray, eligible_only: bool = True):
    rows, ys, ids = [], [], []
    for s in stays:
        e1 = batcher.e1_features(s)
        P1 = s.x1.shape[1]
        v1 = e1[:, :P1]
        trend = v1 - np.vstack([np.repeat(v1[:1], 6, 0), v1])[:len(v1)]
        x2n = batcher.norm("x2", s.x2)
        v2, m2, a2 = locf_features(x2n, max_age2)
        d2 = batcher.norm("d2", s.d2)
        st = np.nan_to_num(batcher.norm("static", s.static))
        X = np.concatenate([e1, trend, np.nan_to_num(v2, nan=np.nan), m2, np.minimum(a2, AGE_CAP_H),
                            d2, np.repeat(st[None], len(s.y), 0),
                            np.arange(len(s.y))[:, None].astype(np.float32)], axis=1)
        keep = s.elig == 1 if eligible_only else np.ones(len(s.y), bool)
        rows.append(X[keep]); ys.append(s.y[keep]); ids.append(np.full(keep.sum(), s.stay_id))
    return np.concatenate(rows).astype(np.float32), np.concatenate(ys), np.concatenate(ids)


def train_gbm(X, y, seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
                                          l2_regularization=1.0, class_weight="balanced",
                                          early_stopping=True, validation_fraction=0.1,
                                          random_state=seed).fit(X, y)
