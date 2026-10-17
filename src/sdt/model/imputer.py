"""Phase 3: forward-only (causal) imputer and its mask-and-recover validation.

CausalGRUImputer predicts every vital at hour t, with a Gaussian mean and
variance, from hours < t only, so it is usable in the live twin (a
bidirectional imputer such as BRITS would read the future).

Validation hides REAL observed values in three patterns (MCAR, contiguous
blocks, value-dependent MNAR-style), re-imputes them, and scores against the
hidden truth: MAE, RMSE and 90 % interval coverage. Imputed values are never
used as ground truth anywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from sdt.model.data import AGE_CAP_H, Normalizer, Stay, locf_features

Z90 = 1.6449


class CausalGRUImputer(nn.Module):
    def __init__(self, n_params: int, hidden: int = 96):
        super().__init__()
        self.gru = nn.GRU(3 * n_params, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2 * n_params)

    def forward(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """inp [B, T, 3P] = features of hours 0..T-1. Output at t predicts hour t
        from hours < t (inputs are shifted right by one hour)."""
        shifted = torch.cat([torch.zeros_like(inp[:, :1]), inp[:, :-1]], dim=1)
        h, _ = self.gru(shifted)
        mean, logvar = self.head(h).chunk(2, dim=-1)
        return mean, logvar.clamp(-6, 4)


def _features(x_obs_norm: np.ndarray, max_age: np.ndarray) -> np.ndarray:
    val, mask, age = locf_features(x_obs_norm, max_age)
    return np.concatenate([np.nan_to_num(val), mask, np.minimum(age, AGE_CAP_H) / AGE_CAP_H], axis=1)


def _batches(arrs: list[np.ndarray], targets: list[np.ndarray], size: int, device, shuffle: bool, rng):
    order = rng.permutation(len(arrs)) if shuffle else np.arange(len(arrs))
    for i in range(0, len(order), size):
        idx = order[i:i + size]
        T = max(len(arrs[j]) for j in idx)
        P = targets[idx[0]].shape[1]
        x = np.zeros((len(idx), T, arrs[idx[0]].shape[1]), np.float32)
        y = np.full((len(idx), T, P), np.nan, np.float32)
        for k, j in enumerate(idx):
            x[k, :len(arrs[j])] = arrs[j]
            y[k, :len(targets[j])] = targets[j]
        yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), idx


def train_imputer(train: list[Stay], val: list[Stay], norm: Normalizer, max_age: np.ndarray,
                  device: str = "cuda", epochs: int = 15, seed: int = 0, log=print) -> CausalGRUImputer:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xs = {k: [norm("x1", s.x1) for s in d] for k, d in (("tr", train), ("va", val))}
    feats = {k: [_features(x, max_age) for x in v] for k, v in xs.items()}
    model = CausalGRUImputer(train[0].x1.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        for x, y, _ in _batches(feats["tr"], xs["tr"], 128, device, True, rng):
            mean, logvar = model(x)
            m = ~torch.isnan(y)
            nll = 0.5 * (logvar + (torch.nan_to_num(y) - mean) ** 2 / logvar.exp())
            loss = (nll * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vl = _eval_nll(model, feats["va"], xs["va"], device)
        log(f"imputer epoch {ep}: val NLL {vl:.4f}")
        if vl < best - 1e-4:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _eval_nll(model, feats, xs, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for x, y, _ in _batches(feats, xs, 256, device, False, None):
        mean, logvar = model(x)
        m = ~torch.isnan(y)
        nll = 0.5 * (logvar + (torch.nan_to_num(y) - mean) ** 2 / logvar.exp())
        tot += (nll * m).sum().item(); n += m.sum().item()
    return tot / n


@torch.no_grad()
def impute(model, x_obs_norm: list[np.ndarray], max_age, device) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Causal predictions (mean, sd) for every hour of every stay, normalised units."""
    model.eval()
    feats = [_features(x, max_age) for x in x_obs_norm]
    means, sds = [None] * len(feats), [None] * len(feats)
    for x, _, idx in _batches(feats, x_obs_norm, 256, device, False, None):
        mean, logvar = model(x)
        for k, j in enumerate(idx):
            T = len(feats[j])
            means[j] = mean[k, :T].cpu().numpy()
            sds[j] = (0.5 * logvar[k, :T]).exp().cpu().numpy()
    return means, sds


# ------------------------------------------------------------ mask-and-recover

def hide(x: np.ndarray, pattern: str, rng, rate: float = 0.15, normal_band: tuple | None = None) -> np.ndarray:
    """Returns a boolean [T, P] array of OBSERVED entries to hide."""
    obs = ~np.isnan(x)
    if pattern == "mcar":
        return obs & (rng.random(x.shape) < rate)
    if pattern == "block":                      # sensor drop-outs of 2-8 h
        h = np.zeros_like(obs)
        T, P = x.shape
        for p in range(P):
            n_blocks = rng.binomial(max(T // 20, 1), rate * 2)
            for _ in range(n_blocks):
                s, L = rng.integers(0, T), rng.integers(2, 9)
                h[s:s + L, p] = True
        return obs & h
    if pattern == "mnar":                       # stable, in-range values are charted less
        lo, hi = normal_band
        inband = (x >= lo) & (x <= hi)
        p = np.where(inband, rate * 1.6, rate * 0.3)
        return obs & (rng.random(x.shape) < p)
    raise ValueError(pattern)


def mask_and_recover(model, stays: list[Stay], norm: Normalizer, max_age, names: list[str],
                     device: str, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mu, sd = norm.mean["x1"], norm.std["x1"]
    band = (-np.ones_like(mu), np.ones_like(mu))          # within 1 SD of the training mean
    rows = []
    for pattern in ("mcar", "block", "mnar"):
        hidden_sets, masked = [], []
        for s in stays:
            xn = norm("x1", s.x1)
            hmask = hide(xn, pattern, rng, normal_band=band)
            hidden_sets.append(hmask)
            masked.append(np.where(hmask, np.nan, xn))
        means, sds = impute(model, masked, max_age, device)
        truth, gru, gsd, locf, pidx = [], [], [], [], []
        for xm, hm, m, s_, st in zip(masked, hidden_sets, means, sds, stays):
            if not hm.any():
                continue
            carried, _, _ = locf_features(xm, np.full(xm.shape[1], np.inf))
            # LOCF at hour t must use hours < t: shift carried values down by one hour.
            prev = np.vstack([np.full((1, xm.shape[1]), np.nan), carried[:-1]])
            t, p = np.nonzero(hm)
            truth.append(norm("x1", st.x1)[t, p]); gru.append(m[t, p]); gsd.append(s_[t, p])
            locf.append(np.where(np.isnan(prev[t, p]), 0.0, prev[t, p])); pidx.append(p)
        p_all, T_, G_, S_, L_ = map(np.concatenate, (pidx, truth, gru, gsd, locf))
        for j, name in enumerate(names):
            k = p_all == j
            if k.sum() < 20:
                continue
            scale = sd[j]
            err = lambda pred: (pred[k] - T_[k]) * scale
            cover = np.mean(np.abs(T_[k] - G_[k]) <= Z90 * S_[k])
            for method, pred in (("causal_gru", G_), ("locf", L_), ("train_mean", np.zeros_like(T_))):
                e = err(pred)
                rows.append(pd.DataFrame([{
                    "pattern": pattern, "param": name, "method": method, "n_hidden": int(k.sum()),
                    "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
                    "coverage_90": float(cover) if method == "causal_gru" else np.nan}]))
    return pd.concat(rows, ignore_index=True)
