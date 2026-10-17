"""Layer 3: static baseline and risk indices (plan, "Layer 3" table).

Leakage note: ICD codes of the CURRENT admission are assigned at discharge, so
they describe the whole stay, including what happens after any prediction
hour. `charlson` (the model input) therefore uses PRIOR admissions only;
`charlson_incl_current` is kept for descriptive statistics and comparisons
with published cohorts, and must not be used as a model input.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Charlson comorbidity groups: ICD-9-CM and ICD-10 coding algorithms of Quan et
# al. 2005 (Med Care), as implemented in the MIT-LCP mimic-code `charlson`
# concept. Entries are code prefixes; "A-B" is an inclusive prefix range.
CHARLSON = {
    "mi": (1, ["410", "412"], ["I21", "I22", "I252"]),
    "chf": (1, ["428", "39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413",
                "40491", "40493", "4254-4259"],
            ["I43", "I50", "I099", "I110", "I130", "I132", "I255", "I420", "I425-I429", "P290"]),
    "pvd": (1, ["440", "441", "0930", "4373", "4471", "5571", "5579", "V434", "4431-4439"],
            ["I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551", "K558", "K559",
             "Z958", "Z959"]),
    "cerebrovascular": (1, ["430-438", "36234"], ["G45", "G46", "I60-I69", "H340"]),
    "dementia": (1, ["290", "2941", "3312"], ["F00-F03", "G30", "F051", "G311"]),
    "chronic_pulmonary": (1, ["490-505", "4168", "4169", "5064", "5081", "5088"],
                          ["J40-J47", "J60-J67", "I278", "I279", "J684", "J701", "J703"]),
    "rheumatic": (1, ["725", "4465", "7100-7104", "7140-7142", "7148"],
                  ["M05", "M06", "M32-M34", "M315", "M351", "M353", "M360"]),
    "peptic_ulcer": (1, ["531-534"], ["K25-K28"]),
    "mild_liver": (1, ["570", "571", "0706", "0709", "5733", "5734", "5738", "5739", "V427",
                       "07022", "07023", "07032", "07033", "07044", "07054"],
                   ["B18", "K73", "K74", "K700-K703", "K709", "K713-K715", "K717", "K760",
                    "K762-K764", "K768", "K769", "Z944"]),
    "diabetes": (1, ["2500-2503", "2508", "2509"],
                 [f"E1{d}{s}" for d in range(5) for s in "01689"]),
    "diabetes_cc": (2, ["2504-2507"], [f"E1{d}{s}" for d in range(5) for s in "23457"]),
    "paraplegia": (2, ["342", "343", "3341", "3440-3446", "3449"],
                   ["G81", "G82", "G041", "G114", "G801", "G802", "G830-G834", "G839"]),
    "renal": (2, ["582", "585", "586", "V56", "5830-5837", "40301", "40311", "40391", "40402",
                  "40403", "40412", "40413", "40492", "40493", "5880", "V420", "V451"],
              ["N18", "N19", "N052-N057", "N250", "I120", "I131", "N032-N037", "Z490-Z492",
               "Z940", "Z992"]),
    "malignancy": (2, ["140-172", "1740-1958", "200-208", "2386"],
                   ["C00-C26", "C30-C34", "C37-C41", "C43", "C45-C58", "C60-C76", "C81-C85",
                    "C88", "C90-C97"]),
    "severe_liver": (3, ["4560-4562", "5722-5728"],
                     ["I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729", "K765",
                      "K766", "K767"]),
    "metastatic": (6, ["196-199"], ["C77-C80"]),
    "aids": (6, ["042-044"], ["B20-B22", "B24"]),
}
# A severe form replaces its mild form in the score (Charlson hierarchy).
_HIERARCHY = [("mild_liver", "severe_liver"), ("diabetes", "diabetes_cc"), ("malignancy", "metastatic")]


def _expand(entry: str) -> list[str]:
    if "-" not in entry:
        return [entry]
    lo, hi = entry.split("-")
    m_lo, m_hi = re.match(r"([A-Z]*)(\d+)", lo), re.match(r"([A-Z]*)(\d+)", hi)
    letter, width = m_lo.group(1), len(m_lo.group(2))
    return [f"{letter}{n:0{width}d}" for n in range(int(m_lo.group(2)), int(m_hi.group(2)) + 1)]


_PREFIXES = {
    name: ({p for e in icd9 for p in _expand(e)}, {p for e in icd10 for p in _expand(e)})
    for name, (_, icd9, icd10) in CHARLSON.items()
}


def comorbidity_flags(dx: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject_id, hadm_id) with a 0/1 column per Charlson group."""
    dx = dx.assign(code=dx["icd_code"].str.upper().str.replace(".", "", regex=False).str.strip())
    out = dx[["subject_id", "hadm_id", "admittime"]].drop_duplicates().set_index(["subject_id", "hadm_id"])
    for name, (p9, p10) in _PREFIXES.items():
        hit9 = (dx["icd_version"] == 9) & dx["code"].map(lambda c: any(c.startswith(p) for p in p9))
        hit10 = (dx["icd_version"] == 10) & dx["code"].map(lambda c: any(c.startswith(p) for p in p10))
        flag = (hit9 | hit10).groupby([dx["subject_id"], dx["hadm_id"]]).max()
        out[name] = flag.reindex(out.index).fillna(False).astype("int8")
    return out.reset_index()


def charlson_score(flags: pd.DataFrame) -> pd.Series:
    f = flags.copy()
    for mild, severe in _HIERARCHY:
        f[mild] = np.where(f[severe] == 1, 0, f[mild])
    return sum(f[name] * weight for name, (weight, _, _) in CHARLSON.items())


def static_features(stays: pd.DataFrame, dx: pd.DataFrame, cultures: pd.DataFrame,
                    env_features: list[str]) -> pd.DataFrame:
    flags = comorbidity_flags(dx)
    groups = list(CHARLSON)

    # Prior admissions: any admission of the same patient that started before this one.
    prior = stays[["stay_id", "subject_id", "admittime"]].merge(
        flags, on="subject_id", suffixes=("", "_dx"))
    prior = prior[prior["admittime_dx"] < prior["admittime"]]
    prior_flags = prior.groupby("stay_id")[groups].max().reindex(stays["stay_id"]).fillna(0)
    current = stays[["stay_id", "hadm_id"]].merge(flags, on="hadm_id", how="left").set_index("stay_id")
    current_flags = (current[groups].fillna(0).reindex(stays["stay_id"]).fillna(0)
                     .combine(prior_flags, np.maximum))

    s = stays.set_index("stay_id")
    out = pd.DataFrame(index=s.index)
    out["age"] = s["age"]
    out["female"] = (s["gender"] == "F").astype("int8")
    out["admission_type"] = s["admission_type"].fillna("unknown")
    out["first_careunit"] = s["first_careunit"].fillna("unknown")
    out["weight_kg"] = s.get("weight_kg")
    out["has_prior_admission"] = s.index.isin(prior["stay_id"]).astype("int8")
    out["charlson"] = charlson_score(prior_flags).astype(float)
    out["charlson_incl_current"] = charlson_score(current_flags).astype(float)
    for g in ["renal", "mild_liver", "severe_liver", "diabetes", "diabetes_cc",
              "chronic_pulmonary", "malignancy", "metastatic", "aids", "chf"]:
        out[f"hx_{g}"] = prior_flags[g].astype("int8")
    out["frailty_cfs"] = "unknown"          # not recorded in MIMIC-IV: explicit unknown category
    out = out.join(infection_source(stays, cultures))
    for e in env_features:                  # not linkable in MIMIC-IV (dates shifted, no location)
        out[e] = np.nan
        out[f"{e}__mask"] = 0
    out["race"] = s["race"]                 # fairness subgroup analysis only; NOT a model input
    return out


_SOURCE_MAP = [("respiratory", r"SPUTUM|BRONCH|BAL\b|TRACHEAL|PLEURAL|LUNG|RESPIRATORY|INFLUENZA"),
               ("urinary", r"URINE"), ("blood", r"BLOOD CULTURE"),
               ("abdominal", r"PERITONEAL|BILE|ABSCESS|STOOL|FECES"),
               ("csf", r"CSF"), ("line", r"CATHETER TIP"), ("wound_skin", r"WOUND|TISSUE|SWAB")]
# Surveillance screens and serology say nothing about the infection source.
_NOT_A_SOURCE = r"MRSA SCREEN|STAPH AUREUS SWAB|SEROLOGY|IMMUNOLOGY|^BLOOD \(|VRE|C\. ?DIFF"


def infection_source(stays: pd.DataFrame, cultures: pd.DataFrame) -> pd.DataFrame:
    """Specimen type of the first culture drawn within 24 h of ICU admission
    (a time-zero proxy for the suspected source; refined by the label step)."""
    c = cultures.merge(stays[["hadm_id", "stay_id", "intime"]], on="hadm_id")
    c = c[(c["culture_time"] >= c["intime"] - pd.Timedelta(hours=24))
          & (c["culture_time"] <= c["intime"] + pd.Timedelta(hours=24))]
    c = c.sort_values("culture_time")
    def classify(desc: str) -> str:
        for name, pat in _SOURCE_MAP:
            if re.search(pat, str(desc).upper()):
                return name
        return "other"
    desc = c["spec_type_desc"].str.upper()
    c = c[~desc.str.contains(_NOT_A_SOURCE, regex=True, na=False)]
    nonblood = c[~c["spec_type_desc"].str.upper().str.contains("BLOOD CULTURE", na=False)]
    first = nonblood.groupby("stay_id")["spec_type_desc"].first().map(classify)
    any_blood = c.groupby("stay_id")["spec_type_desc"].apply(
        lambda s: s.str.upper().str.contains("BLOOD CULTURE", na=False).any())
    out = pd.DataFrame(index=stays["stay_id"])
    out["infection_source"] = first.reindex(out.index).fillna(
        any_blood.reindex(out.index).map({True: "blood_only"})).fillna("none_sampled")
    return out
