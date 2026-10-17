"""Worked-example tests for every derived formula (Phase 2 gate: "formulas unit-tested")."""
import numpy as np
import pandas as pd
import pytest

from sdt.derive import layer1, layer2, layer3


def s(*vals):
    return pd.Series(vals, dtype=float)


# ------------------------------------------------------------------ Layer 1

def test_pao2_from_spo2_inverts_severinghaus():
    # Severinghaus: S = 1 / (23400 / (P^3 + 150 P) + 1). Round-trip at several PaO2.
    for p in (40, 60, 80):
        sat = 100 / (23400 / (p ** 3 + 150 * p) + 1)
        assert layer1.pao2_from_spo2([sat])[0] == pytest.approx(p, rel=1e-3)


def test_news2_bands_match_rcp_table():
    b = layer1._bands
    rr = b(s(8, 9, 11, 12, 20, 21, 24, 25), [8, 11, 20, 24], [3, 1, 0, 2, 3])
    assert rr.tolist() == [3, 1, 1, 0, 0, 2, 2, 3]
    hr = b(s(40, 41, 50, 51, 90, 91, 110, 111, 130, 131), [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
    assert hr.tolist() == [3, 1, 1, 0, 0, 1, 1, 2, 2, 3]
    temp = b(s(35.0, 35.1, 36.0, 36.1, 38.0, 38.1, 39.0, 39.1), [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])
    assert temp.tolist() == [3, 1, 1, 0, 0, 1, 1, 2]


def test_gcs_intubated_is_15_and_flagged():
    h = pd.DataFrame({"gcs_eye": [3, 2], "gcs_verbal": [4, 0], "gcs_motor": [6, 5]})
    out = layer1.gcs_total(h)
    assert out["gcs"].tolist() == [13, 15]
    assert out["gcs_unable"].tolist() == [0, 1]


def test_fio2_sources():
    h = pd.DataFrame({"fio2": [0.5, np.nan, np.nan, np.nan], "o2_flow": [np.nan, 2, 0, 15]})
    cfg = {"nasal_cannula_fio2_per_lpm": 0.04, "nasal_cannula_max_lpm": 6}
    out = layer1.fio2_effective(h, cfg)
    assert out["fio2_eff"].iloc[0] == 0.5
    assert out["fio2_eff"].iloc[1] == pytest.approx(0.29)
    assert out["fio2_eff"].iloc[2] == 0.21
    assert np.isnan(out["fio2_eff"].iloc[3])                 # high flow: unknown FiO2, not guessed
    assert out["fio2_eff__source"].tolist() == ["measured", "model_estimate", "assumed", ""]


# ------------------------------------------------------------------ Layer 2 (SOFA)

def test_sofa_resp_caps_at_2_without_ventilation():
    pf = s(450, 350, 250, 150, 80)
    vent = pd.Series([True] * 5)
    assert layer2.sofa_resp(pf, vent).tolist() == [0, 1, 2, 3, 4]
    assert layer2.sofa_resp(pf, ~vent).tolist() == [0, 1, 2, 2, 2]


def test_sofa_organ_cutoffs():
    assert layer2.sofa_coag(s(150, 149, 100, 99, 50, 49, 20, 19)).tolist() == [0, 1, 1, 2, 2, 3, 3, 4]
    assert layer2.sofa_liver(s(1.1, 1.2, 1.9, 2.0, 5.9, 6.0, 11.9, 12.0)).tolist() == [0, 1, 1, 2, 2, 3, 3, 4]
    assert layer2.sofa_cns(s(15, 14, 13, 12, 10, 9, 6, 5, 3)).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert layer2.sofa_renal(s(1.1, 1.2, 2.0, 3.5, 5.0), s(np.nan, np.nan, np.nan, np.nan, np.nan)).tolist() == [0, 1, 2, 3, 4]
    # urine output overrides a lower creatinine score
    assert layer2.sofa_renal(s(1.0, 1.0), s(400, 150)).tolist() == [3, 4]


def test_sofa_cardiovascular():
    iv = pd.DataFrame({"vaso_dopamine": [0, 0, 4, 10, 0, 0, 20], "vaso_dobutamine": [0, 0, 0, 0, 0, 0, 0],
                       "vaso_epinephrine": [0, 0, 0, 0, 0.05, 0.2, 0],
                       "vaso_norepinephrine": [0, 0, 0, 0, 0, 0, 0]}, dtype=float)
    mp = s(75, 65, 65, 65, 65, 65, 65)
    assert layer2.sofa_cv(mp, iv).tolist() == [0, 1, 2, 3, 3, 4, 4]


# ------------------------------------------------------------------ Layer 2 (other)

def test_egfr_ckd_epi_2021_reference_values():
    # Published CKD-EPI 2021 examples: 50-year-old, creatinine 1.0 mg/dL.
    assert layer2.egfr_ckd_epi_2021(s(1.0), np.array([50.0]), np.array([False]))[0] == pytest.approx(92, abs=1)
    assert layer2.egfr_ckd_epi_2021(s(1.0), np.array([50.0]), np.array([True]))[0] == pytest.approx(68, abs=1)


def test_sic_score():
    out = layer2.sic_score(s(90, 140, 200), s(1.5, 1.3, 1.0), s(3, 1, 0))
    assert out["sic_score"].tolist() == [6, 3, 0]           # 2+2+2, 1+1+1, 0
    assert out["sic_positive"].tolist() == [1, 0, 0]


def test_isth_dic_boundaries():
    cfg = {"dic_d_dimer_moderate_ngml": 1000, "dic_d_dimer_strong_ngml": 5000, "dic_normal_pt_s": 12.0}
    # platelets 50 -> 1, d-dimer 6000 -> 3, PT 15 (+3 s) -> 1, fibrinogen 90 -> 1  => 6
    assert layer2.isth_dic(s(50), s(6000), s(15), s(90), cfg).iloc[0] == 6
    # platelets 101 -> 0, d-dimer 500 -> 0, PT 14.9 -> 0, fibrinogen 200 -> 0
    assert layer2.isth_dic(s(101), s(500), s(14.9), s(200), cfg).iloc[0] == 0


# ------------------------------------------------------------------ Layer 3

def test_charlson_prefix_ranges_and_hierarchy():
    assert "I69" in layer3._PREFIXES["cerebrovascular"][1]
    assert "1958" in layer3._PREFIXES["malignancy"][0]
    dx = pd.DataFrame({
        "subject_id": [1, 1, 1], "hadm_id": [10, 10, 10],
        "icd_code": ["C50.9", "C78.0", "E11.22"], "icd_version": [10, 10, 10],
        "admittime": pd.to_datetime(["2100-01-01"] * 3)})
    flags = layer3.comorbidity_flags(dx)
    # metastatic (6) replaces malignancy (2); diabetes with complications (2)
    assert layer3.charlson_score(flags).iloc[0] == 8
