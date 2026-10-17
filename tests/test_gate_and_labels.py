import numpy as np
import pandas as pd

from sdt import schema
from sdt.labels import sepsis3
from sdt.quality import gate

T0 = pd.Timestamp("2150-01-01 00:00")
STAYS = pd.DataFrame({"stay_id": [1], "hadm_id": [10], "intime": [T0], "outtime": [T0 + pd.Timedelta(hours=48)]})
SPECS = {"heart_rate": {"unit": "bpm", "itemids": {220045: "Heart Rate"}, "hard_limits": [0, 300],
                        "extreme": [30, 200], "max_jump": 80, "jump_window_min": 5}}


def rows(minutes, values, itemid=220045):
    return pd.DataFrame({"stay_id": 1, "itemid": itemid,
                         "charttime": [T0 + pd.Timedelta(minutes=m) for m in minutes], "value": values})


def test_gate_keeps_extremes_removes_impossible_and_flags_spikes():
    raw = rows([10, 11, 12, 13, 60, 61, 3000], [70, 190, 72, 72, 350, 25, 80])
    out = gate.run_gate(gate.map_and_convert(raw, SPECS), SPECS, STAYS)
    flags = dict(zip(out["value"], out["flag"]))
    assert flags[190] == schema.SUSPECT          # 70 -> 190 -> 72 within minutes: transient spike
    assert flags[350] == schema.IMPOSSIBLE       # above hard limit: removed
    assert flags[25] == schema.EXTREME           # bradycardia 25: real, kept
    assert flags[80] == schema.OUT_OF_WINDOW     # charted 50 h after ICU admission
    lake = gate.to_lake_rows(out, "test")
    assert 350 not in lake["value"].tolist() and 25 in lake["value"].tolist()


def test_gate_bp_consistency():
    specs = {"sbp": {"unit": "mmHg", "itemids": {220179: "x"}}, "dbp": {"unit": "mmHg", "itemids": {220180: "x"}},
             "map": {"unit": "mmHg", "itemids": {220181: "x"}}}
    raw = pd.concat([rows([5], [120], 220179), rows([5], [70], 220180), rows([5], [130], 220181)])
    out = gate.run_gate(gate.map_and_convert(raw, specs), specs, STAYS)
    assert out.set_index("param").loc["map", "flag"] == schema.SUSPECT


def test_suspicion_windows():
    cfg = {"abx_then_culture_h": 72, "culture_then_abx_h": 24}
    abx = pd.DataFrame({"hadm_id": [10, 10], "starttime": [T0, T0 + pd.Timedelta(hours=200)]})
    cultures = pd.DataFrame({"hadm_id": [10, 10],
                             "culture_time": [T0 + pd.Timedelta(hours=10), T0 + pd.Timedelta(hours=190)]})
    s = sepsis3.suspicion_times(abx, cultures, cfg).sort_values("suspicion_time")
    # abx then culture 10 h later -> abx time; culture then abx 10 h later -> culture time
    assert s["suspicion_time"].tolist() == [T0, T0 + pd.Timedelta(hours=190)]


def test_onset_is_min_of_suspicion_and_sofa_and_labels_are_future_only():
    cfg = {"sofa_threshold": 2, "baseline_sofa": 0, "sofa_window_before_h": 48, "sofa_window_after_h": 24}
    idx = pd.MultiIndex.from_product([[1], range(48)], names=["stay_id", "hour"])
    sofa = pd.Series(0.0, index=idx)
    sofa.loc[(1, 20):] = 3                                   # SOFA >= 2 from hour 20
    hour_end = pd.Series([T0 + pd.Timedelta(hours=h + 1) for h in range(48)], index=idx)
    susp = pd.DataFrame({"hadm_id": [10], "suspicion_time": [T0 + pd.Timedelta(hours=30)]})
    on = sepsis3.sepsis_onsets(sofa, hour_end, STAYS, susp, cfg)
    assert on.loc[1, "onset_hour"] == 20                     # SOFA time (hour 20) precedes suspicion (30 h)
    lab = sepsis3.hourly_labels(idx, on, pd.Series(dtype=float), [6])
    pos = lab.index.get_level_values("hour")[lab["y_sepsis_6h"] == 1].tolist()
    assert pos == [14, 15, 16, 17, 18, 19]
    assert lab.loc[(1, 20), "eligible_sepsis"] == 0
