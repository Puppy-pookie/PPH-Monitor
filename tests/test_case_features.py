import numpy as np
import pandas as pd

from pph_pipeline.case_features import (
    FEATURE_RELEVANCE, SUMMARY_FEATURES, summarize_all, summarize_case_detail,
)
from pph_pipeline.config import Config


def _windows(minutes=120, base_min=10, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, (base_min + minutes) * 60, 30.0)
    n = len(t)
    return pd.DataFrame({
        "t_start": t, "ppg_ac": 1 + rng.normal(0, 0.02, n), "ppg_pi": 2 + rng.normal(0, 0.02, n),
        "ppg_hr": 75 + rng.normal(0, 1, n), "hr_best": 75 + rng.normal(0, 1, n),
        "sbp": 115 + rng.normal(0, 2, n), "shock_index": 0.65 + rng.normal(0, 0.01, n),
    })


def test_every_summary_feature_is_tagged_for_device_relevance():
    assert set(SUMMARY_FEATURES) <= set(FEATURE_RELEVANCE)
    assert set(FEATURE_RELEVANCE.values()) <= {"ppg_only", "needs_bp", "monitor_hr"}


def test_analysis_window_caps_duration():
    w = _windows(minutes=120)
    cfg_all, cfg_60 = Config(), Config(analysis_minutes=60)
    full, _ = summarize_case_detail(w, cfg_all, op_start=600.0, op_end=600.0 + 120 * 60)
    hour, _ = summarize_case_detail(w, cfg_60, op_start=600.0, op_end=600.0 + 120 * 60)
    assert full["duration_min"] > 100 and 55 <= hour["duration_min"] <= 61


def test_exclusion_reasons_are_reported():
    short = _windows(minutes=5)
    feats, reason = summarize_case_detail(short, Config(), op_start=600.0, op_end=900.0)
    assert feats is None and reason.startswith("surgery_too_short")
    no_base = _windows(base_min=1)
    feats, reason = summarize_case_detail(no_base, Config(), op_start=60.0, op_end=7000.0)
    assert feats is None and reason.startswith("baseline_too_short")


def test_summarize_all_returns_table_and_exclusions_with_extras():
    cases = pd.DataFrame({"caseid": [1, 2], "label": [1, 0], "opstart": [600.0, 600.0], "opend": [7800.0, 900.0]})
    table, excl = summarize_all(cases, {1: _windows(), 2: _windows(minutes=5)}, Config(),
                                extras={1: {"bp_type": "arterial"}})
    assert list(table["caseid"]) == [1] and table.loc[0, "bp_type"] == "arterial"
    assert list(excl["caseid"]) == [2] and "surgery_too_short" in excl.loc[0, "reason"]
