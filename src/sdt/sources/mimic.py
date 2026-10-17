"""MIMIC-IV access layer.

Large event tables are staged once into Parquet, keeping only the itemids the
twin uses (chartevents alone is ~300 M rows in the full release). Every later
step reads the staged Parquet through DuckDB, one batch of ICU stays at a time.
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from sdt.config import Config

log = logging.getLogger(__name__)

# Tables staged whole (small) or filtered by itemid (large).
_WHOLE = {
    "icustays": "icu", "d_items": "icu", "patients": "hosp", "admissions": "hosp",
    "d_labitems": "hosp", "diagnoses_icd": "hosp", "microbiologyevents": "hosp",
    "prescriptions": "hosp",
}
_FILTERED = {"chartevents": "icu", "labevents": "hosp", "outputevents": "icu",
             "inputevents": "icu", "procedureevents": "icu"}


class ItemIdMismatch(RuntimeError):
    pass


class MimicSource:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.mimic_root
        self.stage_dir = cfg.output_root / "staging"
        self.con = duckdb.connect()

    # ------------------------------------------------------------------ staging
    def _csv(self, table: str, module: str) -> Path:
        path = self.root / module / f"{table}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"MIMIC-IV table not found: {path}")
        return path

    def staged(self, table: str) -> Path:
        return self.stage_dir / f"{table}.parquet"

    def itemids_for(self, table: str) -> list[int]:
        cfg = self.cfg
        if table == "chartevents":
            ids = [i for s in cfg.param_specs("primary").values() for i in s["itemids"]]
            ids.append(226512)  # admission weight (kg), used for urine output per kg
            return sorted(set(ids))
        if table == "labevents":
            return sorted({i for s in cfg.param_specs("secondary").values() for i in s["itemids"]})
        if table == "outputevents":
            uo = cfg.parameters["urine_output"]
            return sorted({*uo["itemids"], *uo["negative_itemids"]})
        if table == "inputevents":
            iv = cfg.interventions
            return sorted({v["itemid"] for v in iv["vasopressors"].values()} | set(iv["iv_fluids"]["itemids"]))
        if table == "procedureevents":
            return sorted(v["itemid"] for v in cfg.interventions["ventilation"].values())
        raise KeyError(table)

    def stage(self, force: bool = False) -> None:
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        for table, module in {**_WHOLE, **_FILTERED}.items():
            out = self.staged(table)
            if out.exists() and not force:
                continue
            src = self._csv(table, module).as_posix()
            where = ""
            if table in _FILTERED:
                where = f"WHERE itemid IN ({','.join(map(str, self.itemids_for(table)))})"
            log.info("staging %s", table)
            self.con.execute(
                f"COPY (SELECT * FROM read_csv('{src}', header=true, all_varchar=false, "
                f"sample_size=-1) {where}) TO '{out.as_posix()}' (FORMAT parquet)"
            )

    # ------------------------------------------------------------- validation
    def verify_itemids(self) -> pd.DataFrame:
        """Checks each configured itemid's label against MIMIC's dictionaries."""
        expected: list[tuple[int, str, str]] = []
        for layer in ("primary", "secondary"):
            for name, spec in self.cfg.param_specs(layer).items():
                expected += [(i, lbl, name) for i, lbl in spec["itemids"].items()]
        uo = self.cfg.parameters["urine_output"]
        expected += [(i, l, "urine_output") for i, l in {**uo["itemids"], **uo["negative_itemids"]}.items()]
        iv = self.cfg.interventions
        expected += [(v["itemid"], v["label"], f"vasopressor:{k}") for k, v in iv["vasopressors"].items()]
        expected += [(i, l, "iv_fluid") for i, l in iv["iv_fluids"]["itemids"].items()]
        expected += [(v["itemid"], v["label"], f"ventilation:{k}") for k, v in iv["ventilation"].items()]

        labels = self.query(
            f"SELECT itemid, label FROM '{self.staged('d_items')}' UNION ALL "
            f"SELECT itemid, label FROM '{self.staged('d_labitems')}'"
        ).drop_duplicates("itemid").set_index("itemid")["label"]

        rows = []
        for itemid, want, param in expected:
            got = labels.get(itemid)
            rows.append({"itemid": itemid, "param": param, "expected": want, "found": got,
                         "match": got is not None and got.strip().lower() == want.strip().lower()})
        report = pd.DataFrame(rows)
        bad = report[~report["match"]]
        if not bad.empty:
            raise ItemIdMismatch("itemid labels do not match MIMIC dictionaries:\n" + bad.to_string(index=False))
        return report

    # ---------------------------------------------------------------- queries
    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    def icu_stays(self) -> pd.DataFrame:
        c = self.cfg.settings["cohort"]
        df = self.query(f"""
            SELECT s.subject_id, s.hadm_id, s.stay_id, s.first_careunit, s.intime, s.outtime,
                   date_diff('minute', s.intime, s.outtime) / 60.0 AS los_h,
                   p.gender, p.anchor_age + (year(a.admittime) - p.anchor_year) AS age,
                   a.admittime, a.dischtime, a.deathtime, a.admission_type, a.race,
                   a.hospital_expire_flag
            FROM '{self.staged('icustays')}' s
            JOIN '{self.staged('patients')}' p USING (subject_id)
            JOIN '{self.staged('admissions')}' a USING (subject_id, hadm_id)
            ORDER BY s.subject_id, s.intime
        """)
        df = df[(df["age"] >= c["min_age"]) & (df["los_h"] >= c["min_icu_los_hours"])]
        if c["first_stay_only"]:
            df = df.groupby("subject_id", as_index=False).first()
        return df.reset_index(drop=True)

    def _stay_filter(self, stay_ids) -> str:
        return ",".join(str(int(s)) for s in stay_ids)

    def chartevents(self, stay_ids) -> pd.DataFrame:
        return self.query(f"""
            SELECT stay_id, charttime, itemid, valuenum AS value, valueuom AS unit
            FROM '{self.staged('chartevents')}'
            WHERE stay_id IN ({self._stay_filter(stay_ids)}) AND valuenum IS NOT NULL
        """)

    def labevents(self, stays: pd.DataFrame, pre_icu_hours: int = 24) -> pd.DataFrame:
        """Labs joined to stays by subject and time: from 24 h before ICU admission
        (needed for baselines and carry-forward into hour 0) until ICU discharge."""
        self.con.register("stays_tmp", stays[["subject_id", "stay_id", "intime", "outtime"]])
        try:
            return self.query(f"""
                SELECT s.stay_id, l.charttime, l.itemid, l.valuenum AS value,
                       l.valueuom AS unit, l.specimen_id
                FROM '{self.staged('labevents')}' l
                JOIN stays_tmp s ON l.subject_id = s.subject_id
                 AND l.charttime >= s.intime - INTERVAL {pre_icu_hours} HOUR
                 AND l.charttime <= s.outtime
                WHERE l.valuenum IS NOT NULL
            """)
        finally:
            self.con.unregister("stays_tmp")

    def outputevents(self, stay_ids) -> pd.DataFrame:
        return self.query(f"""
            SELECT stay_id, charttime, itemid, value
            FROM '{self.staged('outputevents')}'
            WHERE stay_id IN ({self._stay_filter(stay_ids)}) AND value IS NOT NULL
        """)

    def inputevents(self, stay_ids) -> pd.DataFrame:
        return self.query(f"""
            SELECT stay_id, starttime, endtime, itemid, amount, amountuom, rate, rateuom,
                   patientweight, statusdescription
            FROM '{self.staged('inputevents')}'
            WHERE stay_id IN ({self._stay_filter(stay_ids)})
        """)

    def procedureevents(self, stay_ids) -> pd.DataFrame:
        return self.query(f"""
            SELECT stay_id, starttime, endtime, itemid
            FROM '{self.staged('procedureevents')}'
            WHERE stay_id IN ({self._stay_filter(stay_ids)})
        """)

    def admission_weight(self, stay_ids) -> pd.DataFrame:
        return self.query(f"""
            SELECT stay_id, arg_min(valuenum, charttime) AS weight_kg
            FROM '{self.staged('chartevents')}'
            WHERE itemid = 226512 AND stay_id IN ({self._stay_filter(stay_ids)})
              AND valuenum BETWEEN 20 AND 400
            GROUP BY stay_id
        """)

    def prescriptions(self, hadm_ids) -> pd.DataFrame:
        ids = ",".join(str(int(h)) for h in hadm_ids)
        return self.query(f"""
            SELECT subject_id, hadm_id, starttime, stoptime, lower(drug) AS drug, route
            FROM '{self.staged('prescriptions')}'
            WHERE hadm_id IN ({ids}) AND starttime IS NOT NULL
        """)

    def cultures(self, hadm_ids) -> pd.DataFrame:
        ids = ",".join(str(int(h)) for h in hadm_ids)
        return self.query(f"""
            SELECT DISTINCT subject_id, hadm_id, micro_specimen_id,
                   coalesce(charttime, CAST(chartdate AS TIMESTAMP)) AS culture_time,
                   charttime IS NULL AS date_only, spec_type_desc,
                   max(CASE WHEN org_name IS NOT NULL AND org_name <> '' THEN 1 ELSE 0 END)
                       OVER (PARTITION BY micro_specimen_id) AS positive
            FROM '{self.staged('microbiologyevents')}'
            WHERE hadm_id IN ({ids})
        """)

    def diagnoses(self, subject_ids) -> pd.DataFrame:
        ids = ",".join(str(int(s)) for s in subject_ids)
        return self.query(f"""
            SELECT d.subject_id, d.hadm_id, d.icd_code, d.icd_version, a.admittime
            FROM '{self.staged('diagnoses_icd')}' d
            JOIN '{self.staged('admissions')}' a USING (subject_id, hadm_id)
            WHERE d.subject_id IN ({ids})
        """)
