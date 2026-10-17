import numpy as np
import pandas as pd
import pytest

from sdt.config import PROJECT_ROOT
from sdt.whatif.hybrid import FLUID_KERNEL, HybridEngine, dml_effects
from sdt.whatif.simulator import World, simulate


def _sim_panel(n=1500, seed=5):
    sim = simulate(n, 30, World(), seed=seed).sort_values(["stay_id", "hour"])
    sim["ne_dose"] = sim.groupby("stay_id")["vaso"].shift(1).fillna(0) * 0.1
    sim["fluid_ml"] = sim["fluid"] * 500
    return sim.assign(age=60.0)


def test_dml_recovers_known_effects_and_beats_naive_sign():
    theta, cov = dml_effects(_sim_panel(), "map", design="same_hour")
    assert theta[1] == pytest.approx(6.0, abs=1.0)        # vasopressor: true +6 per 0.1
    assert 1.0 < theta[0] < 4.0                             # fluid: true +3 per 500 mL (confounded naive is negative)


def test_effect_path_shapes():
    eng = HybridEngine()
    from sdt.whatif.hybrid import Effect, PRIORS
    eng.effects = {o: {k: Effect(0.0, 1.0, *PRIORS[o][k]) for k in ("fluid", "ne")} for o in ("map", "hr")}
    m, sd = eng.effect_path("map", 0.1, [0.1] * 6, [500, 0, 0, 0, 0, 0])
    assert np.allclose(m[:3], 5.0 * FLUID_KERNEL) and np.allclose(m[3:], 0)     # bolus decays
    m, _ = eng.effect_path("map", 0.1, [0.15] * 6, [0] * 6)
    assert np.allclose(m, 10.0)                                                   # dose effect persists
    assert (sd >= 0).all()


@pytest.mark.skipif(not (PROJECT_ROOT / "data" / "whatif" / "engine_v2.pkl").exists(), reason="run run_whatif_v2.py")
def test_trained_engine_guardrails():
    import pickle
    eng = pickle.loads((PROJECT_ROOT / "data" / "whatif" / "engine_v2.pkl").read_bytes())
    hist = pd.DataFrame({"stay_id": "x", "hour": range(6), "map": [70, 68, 66, 65, 64, 63.0],
                         "hr": [100.0] * 6, "lactate": [3.0] * 6, "ne_dose": [0.1] * 6, "fluid_ml": [0.0] * 6,
                         "age": 70.0})
    r = eng.query(hist, 0.0, 3.0)                          # far above any dose in the data
    assert r["answered"] is False and any("above the doses" in x for x in r["reasons"])
    r = eng.query(hist, 500.0, 0.1)
    assert r["answered"] is True
    t = r["trajectories"]["map"]
    assert t["difference"].iloc[0] == pytest.approx(5.0) and (t["plan_high"] > t["plan_low"]).all()
