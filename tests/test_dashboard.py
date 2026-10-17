"""Dashboard API smoke tests (skipped until Phase 3-4 models exist)."""
import pytest

from sdt.pipeline_physionet import load_adapter
from sdt.sources import physionet2019 as pn

_, ROOT = pn.adapter_paths(load_adapter())
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "fusion.pt").exists(), reason="train Phase 3-4 first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from sdt.dashboard.app import Twin, create_app
    return TestClient(create_app(Twin(n_stays=40, n_septic=10)))


def test_ward_and_patient_never_show_the_future(client):
    ward = client.get("/api/ward?clock=10").json()
    assert ward and all(r["hour"] <= 10 for r in ward)
    sid = ward[0]["stay_id"]
    p = client.get(f"/api/patients/{sid}?clock=10").json()
    assert len(p["risk"]["6h"]) == p["hour"] + 1 <= 11
    assert all(len(v["values"]) == p["hour"] + 1 for v in p["vitals"].values())
    assert "recorded_onset_hour" not in p                 # outcome hidden unless asked for


def test_whatif_is_disabled_and_model_card_present(client):
    assert client.get("/api/patients/1/whatif").json()["available"] is False
    card = client.get("/api/model-card").json()
    assert len(card["validation"]) == 2 and card["limitations"]
    assert client.get("/").status_code == 200


@pytest.mark.skipif(not (ROOT.parent / "eicu_demo" / "lakes" / "timeline").exists(), reason="run run_eicu_pipeline.py")
def test_eicu_cohort_whatif_answers_or_refuses_with_reasons():
    from fastapi.testclient import TestClient
    from sdt.dashboard.app import Twin, create_app
    c = TestClient(create_app(Twin("eicu", n_stays=40)))
    ward = c.get("/api/ward?clock=12&cohort=eicu").json()
    for r in ward:
        s = c.get(f"/api/patients/{r['stay_id']}/whatif?clock=12&cohort=eicu").json()
        if s.get("available"):
            q = c.get(f"/api/patients/{r['stay_id']}/whatif?clock=12&cohort=eicu&fluid_ml=500&ne_dose={s['ne_now']}").json()
            assert q["answered"] in (True, False)
            assert q["trajectories"]["map"]["difference"][0] == 5.0 if q["answered"] else q["reasons"]
            return
    pytest.fail("no patient with enough data for a what-if")
