import numpy as np
import pandas as pd
import pytest

from sdt.config import load_config
from sdt.model.data import locf_features
from sdt.wearable.hrv import clean_rr, hrv_windows
from sdt.wearable.ingest import ingest, perfusion_derivations
from sdt.whatif.engine import EngineConfig, WhatIfEngine
from sdt.whatif.simulator import World, simulate

T0 = pd.Timestamp("2150-01-01")


def test_clean_rr_drops_ectopics_and_impossible():
    rr = np.array([800, 810, 400, 805, 2500, 815])
    assert clean_rr(rr).tolist() == [True, True, False, True, False, True]


def test_hrv_matches_hand_computation():
    rr = np.tile([800.0, 840.0], 60)                       # alternating: successive diff 40 ms
    beats = pd.DataFrame({"stay_id": 1, "time": T0 + pd.to_timedelta(np.cumsum(rr) / 1000, unit="s"),
                          "rr_ms": rr})
    w = hrv_windows(beats, window="10min")
    assert w["rmssd_ms"].iloc[0] == pytest.approx(40.0)
    assert w["sdnn_ms"].iloc[0] == pytest.approx(np.std(rr, ddof=1))
    assert w["pnn50_pct"].iloc[0] == 0.0
    assert w["flag"].iloc[0] == "ok"


def test_wearable_ingest_gates_values():
    cfg = load_config()
    stays = pd.DataFrame({"stay_id": [1], "intime": [T0], "outtime": [T0 + pd.Timedelta(hours=24)]})
    samples = pd.DataFrame({"stay_id": 1, "timestamp": [T0 + pd.Timedelta(minutes=m) for m in (5, 10, 15)],
                            "signal": ["gsr", "gsr", "perfusion_index"], "value": [2.0, 150.0, 0.8]})
    beats = pd.DataFrame(columns=["stay_id", "timestamp", "rr_ms"])
    lake, audit = ingest(samples, beats, stays, cfg.parameters, "test-device", "t")
    assert 150.0 not in lake["value"].tolist()              # above GSR hard limit: removed
    assert set(lake["source"]) == {"measured"} and lake["inputs"].iloc[0] == "device:test-device"
    d = perfusion_derivations(pd.Series([37.5]), pd.Series([29.0]), pd.Series([0.8]))
    assert d.iloc[0].tolist() == [8.5, 1.0, 1.0]


def test_locf_respects_max_age():
    x = np.array([[1.0], [np.nan], [np.nan], [np.nan]])
    val, mask, age = locf_features(x, np.array([2.0]))
    assert val[:, 0].tolist()[:3] == [1.0, 1.0, 1.0] and np.isnan(val[3, 0])
    assert mask[:, 0].tolist() == [1, 0, 0, 0] and age[:, 0].tolist() == [0, 1, 2, 3]


def test_whatif_refuses_unsupported_plan():
    df = simulate(800, 20, World(), seed=3)
    eng = WhatIfEngine(EngineConfig(obs=["map", "lactate", "hr"], treatments=["fluid", "vaso"],
                                    n_boot=1, n_samples=5)).fit(df)
    hist = df[df.hour < 10]
    healthiest = hist[hist.hour == 9].sort_values("map").iloc[-1]["stay_id"]
    r = eng.query(hist[hist.stay_id == healthiest], {"fluid": 0, "vaso": 1}, {"fluid": 0, "vaso": 0}, 3)
    assert r["answered"] is False and "Positivity" in r["reason"]
