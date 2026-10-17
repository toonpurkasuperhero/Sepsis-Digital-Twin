"""PhysioNet/CinC Challenge 2019 source (open access).

Each patient file pNNNNNN.psv holds one row per ICU hour (ICULOS = 1, 2, ...).
Set A (p0xxxxx) and set B (p1xxxxx) come from different hospital systems, which
gives an external-validation split without credentialed data.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from sdt.config import _resolve

INTIME = pd.Timestamp("2000-01-01")   # the challenge has no dates; one synthetic origin per stay


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    df.insert(0, "stay_id", int(path.stem[1:]))
    return df


def load_raw(root: Path, cache: Path, workers: int = 16) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    files = sorted(Path(root).rglob("p*.psv"))
    if not files:
        raise FileNotFoundError(f"no .psv files under {root}")
    with ProcessPoolExecutor(workers) as ex:
        df = pd.concat(ex.map(_read, files, chunksize=500), ignore_index=True)
    df["hospital_set"] = np.where(df["stay_id"] >= 100000, "B", "A")
    df["hour"] = df["ICULOS"].astype(int) - 1
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def stays_table(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.groupby("stay_id")
    st = g.agg(los_h=("ICULOS", "max"), age=("Age", "first"), gender_code=("Gender", "first"),
               unit1=("Unit1", "max"), unit2=("Unit2", "max"), hosp_adm_time_h=("HospAdmTime", "first"),
               hospital_set=("hospital_set", "first")).reset_index()
    st["gender"] = np.where(st["gender_code"] == 1, "M", "F")
    st["unit"] = np.select([st["unit1"] == 1, st["unit2"] == 1], ["MICU", "SICU"], "unknown")
    st["intime"] = INTIME
    st["outtime"] = INTIME + pd.to_timedelta(st["los_h"], unit="h")
    st["hadm_id"] = st["stay_id"]
    st["subject_id"] = st["stay_id"]
    st["weight_kg"] = np.nan
    return st


def long_measurements(raw: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """One row per observed value; charttime is the middle of its ICU hour."""
    keep = [c for c in columns if c in raw]
    m = raw[["stay_id", "hour", *keep]].melt(["stay_id", "hour"], var_name="col", value_name="value")
    m = m.dropna(subset=["value"])
    m["param"] = m["col"].map(columns)
    m["charttime"] = INTIME + pd.to_timedelta(m["hour"] + 0.5, unit="h")
    m["itemid"] = pd.NA
    return m.drop(columns=["col", "hour"])


def onset_hours(raw: pd.DataFrame, shift_h: int) -> pd.DataFrame:
    """Undo the challenge's 6 h label shift. If the first positive row is the
    first ICU hour, onset lies somewhere in hours <= shift_h and is ambiguous;
    such stays are treated as septic at admission (excluded from onset task)."""
    pos = raw[raw["SepsisLabel"] == 1].groupby("stay_id")["hour"].min()
    first = raw.groupby("stay_id")["hour"].min()
    out = pd.DataFrame(index=first.index)
    out["first_positive_hour"] = pos
    out["onset_hour"] = pos + shift_h
    ambiguous = out["first_positive_hour"] == first
    out.loc[ambiguous, "onset_hour"] = -1.0
    out["sepsis3"] = pos.reindex(out.index).notna().astype("int8")
    out["sepsis_at_admission"] = (ambiguous & out["sepsis3"].astype(bool)).astype("int8")
    return out


def specs_for(cfg_params: dict, adapter: dict) -> tuple[dict, dict]:
    """Gate specs per SDT parameter for the challenge columns, split by layer."""
    prim, sec = {}, {}
    for param in adapter["columns"].values():
        if param in cfg_params["primary"]:
            prim[param] = {**cfg_params["primary"][param], "itemids": {}}
        elif param in cfg_params["secondary"]:
            sec[param] = {**cfg_params["secondary"][param], "itemids": {}}
        else:
            spec = dict(adapter["extra_params"][param])
            (prim if spec.pop("layer") == "primary" else sec)[param] = {**spec, "itemids": {}}
    for s in (*prim.values(), *sec.values()):
        s.pop("convert", None)
    return prim, sec


def adapter_paths(adapter: dict) -> tuple[Path, Path]:
    return _resolve(adapter["root"]), _resolve(adapter["output_root"])
