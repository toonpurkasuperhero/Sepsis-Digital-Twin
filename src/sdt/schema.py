"""Shared vocabulary for every data lake row.

Each value in a lake carries where it came from (`source`), how sure we are
(`uncertainty`), what the quality gate thought of it (`flag`) and what it was
computed from (`inputs`). This is the fix for "predicted values mixed with
measured ones" in the plan review.
"""
from __future__ import annotations

# source: how the value came to exist
MEASURED = "measured"
DERIVED_FORMULA = "derived_formula"      # deterministic published formula / score
MODEL_ESTIMATE = "model_estimate"        # statistical estimate; uncertainty required
ASSUMED = "assumed"                      # documented default (e.g. room air FiO2)

# flag: quality-gate verdict
OK = "ok"
EXTREME = "extreme"          # real but rare physiology: kept, used everywhere
SUSPECT = "suspect"          # kept in the lake, excluded from derivations
IMPOSSIBLE = "impossible"    # outside hard limits: removed from the lake
DUPLICATE = "duplicate"      # removed
OUT_OF_WINDOW = "out_of_window"  # outside the ICU stay window: removed

REMOVED_FLAGS = {IMPOSSIBLE, DUPLICATE, OUT_OF_WINDOW}
USABLE_FLAGS = {OK, EXTREME}

LAKE_COLUMNS = [
    "stay_id", "charttime", "param", "value", "unit", "source", "uncertainty",
    "flag", "qc_reason", "inputs", "itemid", "pipeline_version",
]
