"""Writes the data lakes as Parquet, one file per batch of stays.

Layout under <output_root>/lakes/:
  lake1_primary/       measured vitals + urine output + Layer 1 derived (long)
  lake2_secondary/     measured labs + Layer 2 derived (long)
  lake3_tertiary/      static baseline features (one row per stay)
  outcomes/            per-stay Sepsis-3 onset, shock hour, mortality (never a feature)
  interventions/       hourly intervention stream (wide)
  labels/              hourly Sepsis-3 / shock targets (wide)
  timeline/            model-ready hourly table: value, mask, age per parameter
  quality_removed/     rows the gate removed (to audit the gate itself)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sdt import schema


class LakeWriter:
    def __init__(self, root: Path, version: str):
        self.root = root / "lakes"
        self.version = version

    def write(self, name: str, df: pd.DataFrame, batch: int) -> None:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        # A named index (stay_id, hour) is data and is always kept, even when it
        # happens to be a contiguous range that pandas stores as a RangeIndex.
        keep_index = any(n is not None for n in df.index.names) or not isinstance(df.index, pd.RangeIndex)
        df.to_parquet(d / f"part-{batch:04d}.parquet", index=keep_index)

    def derived_rows(self, wide: pd.DataFrame, meta: dict, hour_end: pd.Series) -> pd.DataFrame:
        """Converts derived wide columns into lake rows with source, uncertainty and inputs."""
        parts = []
        for col, (unit, source, inputs) in meta.items():
            if col not in wide:
                continue
            v = wide[col]
            keep = v.notna()
            if not keep.any():
                continue
            src = wide[f"{col}__source"][keep] if source is None else source
            unc = wide[f"{col}__unc"][keep] if f"{col}__unc" in wide else np.nan
            parts.append(pd.DataFrame({
                "stay_id": v.index.get_level_values("stay_id")[keep],
                "charttime": hour_end[keep].to_numpy(),
                "param": col, "value": v[keep].to_numpy(), "unit": unit,
                "source": np.asarray(src), "uncertainty": np.asarray(unc, dtype=float) if np.ndim(unc) else unc,
                "flag": schema.OK, "qc_reason": "", "inputs": inputs, "itemid": pd.NA,
                "pipeline_version": self.version,
            }))
        if not parts:
            return pd.DataFrame(columns=schema.LAKE_COLUMNS)
        return pd.concat(parts, ignore_index=True)[schema.LAKE_COLUMNS]
