"""Training loops for encoder pre-training (Phase 3) and fusion fine-tuning (Phase 4)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

from sdt.model.fusion import Batcher, FusionNet


def _order_by_length(stays, size, rng=None):
    """Length-bucketed batches (less padding); shuffled across buckets when rng is given."""
    idx = np.argsort([len(s.y) for s in stays], kind="stable")
    chunks = [idx[i:i + size] for i in range(0, len(idx), size)]
    if rng is not None:
        rng.shuffle(chunks)
    return chunks


def _masked_mse(pred, target, valid):
    m = ~torch.isnan(target) & (valid.unsqueeze(-1) > 0)
    return ((pred - torch.nan_to_num(target)) ** 2 * m).sum() / m.sum().clamp(min=1)


def pretrain_encoders(model: FusionNet, train, batcher: Batcher, epochs: int = 3, lr: float = 1e-3,
                      seed: int = 0, log=print) -> None:
    """E1 forecasts next-hour vitals; E2 forecasts next-hour labs (plan, Step 1)."""
    rng = np.random.default_rng(seed)
    params = [*model.e1.parameters(), *model.e2.parameters(),
              *model.e1_forecast.parameters(), *model.e2_forecast.parameters()]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    for ep in range(epochs):
        tot = []
        for chunk in _order_by_length(train, 64, rng):
            b = batcher([train[i] for i in chunk])
            z1, z2, _ = model.streams(b)
            loss = _masked_mse(model.e1_forecast(z1), b["next1"], b["valid"]) + \
                   _masked_mse(model.e2_forecast(z2), b["next2"], b["valid"])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot.append(loss.item())
        log(f"pre-train epoch {ep}: forecast loss {np.mean(tot):.4f}")


def train_fusion(model: FusionNet, train, val, batcher: Batcher, pos_weight: torch.Tensor,
                 env_l2: float, epochs: int = 20, patience: int = 3, lr: float = 1e-3,
                 seed: int = 0, log=print) -> dict:
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        losses = []
        for chunk in _order_by_length(train, 64, rng):
            b = batcher([train[i] for i in chunk])
            logits, _ = model(b)
            w = (b["elig"] * b["valid"]).unsqueeze(-1)
            bce = F.binary_cross_entropy_with_logits(logits, b["y"], pos_weight=pos_weight, reduction="none")
            loss = (bce * w).sum() / w.sum().clamp(min=1) / 2 + env_l2 * model.env_penalty()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(loss.item())
        probs, _ = predict(model, val, batcher)
        y = np.concatenate([s.y[s.elig == 1, 0] for s in val])
        p = np.concatenate([pr[s.elig == 1, 0] for s, pr in zip(val, probs)])
        ap = average_precision_score(y, p)
        sched.step(ap)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_auprc_6h": float(ap)})
        log(f"epoch {ep}: train loss {np.mean(losses):.4f}  val AUPRC(6h) {ap:.4f}")
        if ap > best + 1e-4:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_val_auprc_6h": best, "history": history}


@torch.no_grad()
def predict(model: FusionNet, stays, batcher: Batcher, drop: str | None = None,
            x1_override=None, fills=None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    model.eval()
    probs, gates = [None] * len(stays), [None] * len(stays)
    for chunk in _order_by_length(stays, 128):
        b = batcher([stays[i] for i in chunk],
                    None if x1_override is None else [x1_override[i] for i in chunk],
                    None if fills is None else [fills[i] for i in chunk])
        logits, g = model(b, drop)
        pr, g = torch.sigmoid(logits).cpu().numpy(), g.cpu().numpy()
        for k, i in enumerate(chunk):
            L = len(stays[i].y)
            probs[i], gates[i] = pr[k, :L], g[k, :L]
    return probs, gates
