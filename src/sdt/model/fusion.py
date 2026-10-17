"""Phase 4: multi-stream fusion network (plan: "Fusion and outcome prediction").

  E1  causal 1D-CNN + LSTM over hourly vitals (carried value, mask, age) + Layer 1 derived
  E2  masked multi-head attention: each hour attends only to lab tokens observed by then
  E3  MLP over static baseline
  E4  interventions (only when the source has them)
  fusion: self-attention across the stream vectors of each hour + softmax gate;
          the gate weights are reported as per-hour stream importance
  ENV environment inputs act through a separate, hard-capped branch:
          contribution = cap * tanh(w . env), |contribution| <= cap (default 0.05 logit),
          with an L2 penalty on w, as decided for air quality and weather.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sdt.model.data import AGE_CAP_H, Normalizer, Stay, locf_features

MAX_TOKENS = 768


def time_encoding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding of hour values t [...] -> [..., dim]."""
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device) * (-math.log(1000.0) / half))
    ang = t.unsqueeze(-1).float() * freqs
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class VitalsEncoder(nn.Module):
    def __init__(self, n_in: int, d: int):
        super().__init__()
        self.conv = nn.Conv1d(n_in, d, kernel_size=3)
        self.lstm = nn.LSTM(d, d, batch_first=True)

    def forward(self, x):                          # x [B, T, n_in]
        h = F.pad(x.transpose(1, 2), (2, 0))       # left padding only: causal
        h = F.gelu(self.conv(h)).transpose(1, 2)
        return self.lstm(h)[0]


class LabEncoder(nn.Module):
    def __init__(self, n_params: int, n_derived: int, d: int, heads: int = 4):
        super().__init__()
        self.param_emb = nn.Embedding(n_params, d)
        self.val = nn.Linear(1, d)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.q_time = nn.Linear(d, d)
        self.q_derived = nn.Linear(2 * n_derived, d) if n_derived else None
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.heads, self.d = heads, d

    def forward(self, tok_t, tok_p, tok_v, tok_pad, d2, T):
        B, N = tok_t.shape
        k = self.param_emb(tok_p) + self.val(tok_v.unsqueeze(-1)) + time_encoding(tok_t, self.d)
        k = torch.cat([self.null.expand(B, 1, -1), k], dim=1)                    # [B, N+1, d]
        hours = torch.arange(T, device=tok_t.device)
        q = self.q_time(time_encoding(hours, self.d)).unsqueeze(0).expand(B, -1, -1)
        if self.q_derived is not None:
            q = q + self.q_derived(d2)
        blocked = (tok_t.unsqueeze(1) > hours.view(1, T, 1)) | tok_pad.unsqueeze(1)   # future or padding
        blocked = torch.cat([torch.zeros(B, T, 1, dtype=torch.bool, device=q.device), blocked], dim=2)
        mask = blocked.repeat_interleave(self.heads, dim=0)
        out, w = self.attn(q, k, k, attn_mask=mask, need_weights=False)
        return out


class FusionNet(nn.Module):
    def __init__(self, n_e1: int, n_lab: int, n_d2: int, n_static: int, n_env: int,
                 n_e4: int = 0, d: int = 64, env_cap: float = 0.05, n_targets: int = 2,
                 n_vitals: int = 0):
        super().__init__()
        self.e1 = VitalsEncoder(n_e1, d)
        self.e2 = LabEncoder(n_lab, n_d2, d)
        self.e3 = nn.Sequential(nn.Linear(n_static, d), nn.GELU(), nn.Linear(d, d))
        self.e4 = nn.Linear(n_e4, d) if n_e4 else None
        n_streams = 4 if n_e4 else 3
        self.stream_emb = nn.Parameter(torch.randn(n_streams, d) * 0.02)
        self.mix = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True)
        self.gate = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, n_targets))
        self.env = nn.Linear(n_env, 1, bias=False) if n_env else None
        self.env_cap = env_cap
        # pre-training heads (Phase 3 encoder pre-training)
        self.e1_forecast = nn.Linear(d, n_vitals) if n_vitals else None
        self.e2_forecast = nn.Linear(d, n_lab)
        self.stream_names = ["E1_vitals", "E2_labs", "E3_static"] + (["E4_interventions"] if n_e4 else [])

    def streams(self, b, drop: str | None = None):
        T = b["e1"].shape[1]
        z = [self.e1(b["e1"]), self.e2(b["tok_t"], b["tok_p"], b["tok_v"], b["tok_pad"], b["d2"], T),
             self.e3(b["static"]).unsqueeze(1).expand(-1, T, -1)]
        if self.e4 is not None:
            z.append(self.e4(b["e4"]))
        if drop is not None:                               # stream ablation at inference
            z[self.stream_names.index(drop)] = torch.zeros_like(z[0])
        return z

    def forward(self, b, drop: str | None = None):
        z = torch.stack(self.streams(b, drop), dim=2)      # [B, T, K, d]
        B, T, K, d = z.shape
        z = z + self.stream_emb
        m = self.mix(z.reshape(B * T, K, d))
        g = torch.softmax(self.gate(m).squeeze(-1), dim=-1)            # [B*T, K]
        fused = (g.unsqueeze(-1) * m).sum(1).reshape(B, T, d)
        logits = self.head(fused)
        if self.env is not None:
            logits = logits + self.env_contribution(b["env"]).unsqueeze(1)
        return logits, g.reshape(B, T, K)

    def env_contribution(self, env):
        return self.env_cap * torch.tanh(self.env(env))                # [B, 1], bounded by the cap

    def env_penalty(self):
        return (self.env.weight ** 2).sum() if self.env is not None else torch.tensor(0.0)


# ------------------------------------------------------------------ batching

class Batcher:
    """Builds padded tensors from Stay objects. `x1_override` lets evaluation
    feed vitals with values hidden (mask-and-recover downstream test)."""

    def __init__(self, norm: Normalizer, max_age1: np.ndarray, device: str):
        self.norm, self.max_age1, self.device = norm, max_age1, device

    def e1_features(self, s: Stay, x1=None, fill=None):
        xn = self.norm("x1", s.x1 if x1 is None else x1)
        val, mask, age = locf_features(xn, self.max_age1)
        if fill is not None:                       # imputed values where nothing is carried
            val = np.where(np.isnan(val), fill, val)
        d1 = self.norm("d1", s.d1)
        d1_ok = np.isfinite(s.d1)
        feats = np.concatenate([np.nan_to_num(val), mask, np.minimum(age, AGE_CAP_H) / AGE_CAP_H,
                                np.where(d1_ok, d1, 0.0), d1_ok.astype(np.float32)], axis=1)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)   # never feed inf to the network

    def __call__(self, stays: list[Stay], x1_override=None, fills=None):
        B, T = len(stays), max(len(s.y) for s in stays)
        e1 = [self.e1_features(s, None if x1_override is None else x1_override[i],
                               None if fills is None else fills[i]) for i, s in enumerate(stays)]
        n1 = e1[0].shape[1]
        N = max(1, min(MAX_TOKENS, max(len(s.tok_t) for s in stays)))
        P1, P2 = stays[0].x1.shape[1], stays[0].x2.shape[1]
        Q2 = stays[0].d2.shape[1]
        out = {k: np.zeros(shape, np.float32) for k, shape in {
            "e1": (B, T, n1), "d2": (B, T, 2 * Q2), "y": (B, T, 2), "elig": (B, T), "valid": (B, T),
            "next1": (B, T, P1), "next2": (B, T, P2), "tok_v": (B, N)}.items()}
        out["next1"][:] = np.nan; out["next2"][:] = np.nan
        tok_t = np.zeros((B, N), np.int64); tok_p = np.zeros((B, N), np.int64)
        tok_pad = np.ones((B, N), bool)
        mu2, sd2 = self.norm.mean["x2"], self.norm.std["x2"]
        for i, s in enumerate(stays):
            L = len(s.y)
            out["e1"][i, :L] = e1[i]
            d2n = self.norm("d2", s.d2)
            d2_ok = np.isfinite(s.d2)
            out["d2"][i, :L] = np.concatenate([np.where(d2_ok, d2n, 0.0), d2_ok.astype(np.float32)], 1)
            out["y"][i, :L] = s.y; out["elig"][i, :L] = s.elig; out["valid"][i, :L] = 1
            out["next1"][i, :L - 1] = self.norm("x1", s.x1)[1:]
            out["next2"][i, :L - 1] = self.norm("x2", s.x2)[1:]
            n = min(len(s.tok_t), N)
            if n:
                sl = slice(len(s.tok_t) - n, None)                # keep the most recent tokens
                tok_t[i, :n], tok_p[i, :n] = s.tok_t[sl], s.tok_p[sl]
                out["tok_v"][i, :n] = (s.tok_v[sl] - mu2[s.tok_p[sl]]) / sd2[s.tok_p[sl]]
                tok_pad[i, :n] = False
        static = np.stack([np.nan_to_num(self.norm("static", s.static)) for s in stays])
        env = np.stack([np.nan_to_num(s.env) for s in stays])
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self.device)
        b = {k: t(v) for k, v in out.items()}
        b.update(tok_t=t(tok_t, torch.long), tok_p=t(tok_p, torch.long), tok_pad=t(tok_pad, torch.bool),
                 static=t(static), env=t(env))
        return b
