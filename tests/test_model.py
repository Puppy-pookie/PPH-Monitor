import numpy as np
import pandas as pd

from pph_pipeline.config import Config
from pph_pipeline.model import MODEL_SETS, PPG_ONLY, cv_evaluate, evaluate, univariate_auc

FAST = dict(cv_repeats=4, perm_repeats=2, n_permutations=20, n_bootstrap=80)


def _table(n=40, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([1, 0] * (n // 2))
    df = pd.DataFrame({"caseid": np.arange(n), "label": y})
    for f in PPG_ONLY + ["si_rise_max", "sbp_drop_max_pct"]:
        df[f] = rng.normal(0, 1, n) + (1.5 * y if signal else 0)
    df["age"] = rng.integers(25, 80, n)
    df["asa"] = rng.integers(1, 4, n)
    df["emop"] = 0.0
    df["duration_min"] = rng.normal(120, 30, n)
    df["intraop_phe"] = rng.exponential(100, n)
    df["sex"] = "F"
    return df


def test_cv_reports_ci_and_permutation_p_for_real_signal():
    res = cv_evaluate(_table(signal=True), PPG_ONLY, Config(**FAST))
    assert res["status"] == "ok"
    assert 0.5 < res["auc_ci_low"] < res["auc_mean"] < res["auc_ci_high"] <= 1.0
    assert res["p_permutation"] <= 0.1
    assert res["repeats"] == 4 and res["folds"] == 5


def test_no_signal_is_not_called_significant_and_ci_spans_chance():
    res = cv_evaluate(_table(signal=False, seed=3), PPG_ONLY, Config(**FAST))
    assert res["p_permutation"] > 0.05
    assert res["auc_ci_low"] < 0.5 < res["auc_ci_high"] + 0.2


def test_evaluate_runs_all_pre_specified_models_and_notes():
    res = evaluate(_table(signal=False, seed=5), Config(**FAST))
    assert res["status"] == "ok" and set(res["models"]) == set(MODEL_SETS)
    assert any("not distinguishable from chance" in n for n in res["notes"])
    assert any("Fewer than 30 cases" in n for n in res["notes"])


def test_univariate_table_has_ci_and_device_relevance():
    uni = univariate_auc(_table(), PPG_ONLY + ["age"], Config(**FAST))
    assert {"auc_ci_low", "auc_ci_high", "device_relevance"} <= set(uni.columns)
    assert set(uni.loc[uni.feature.isin(PPG_ONLY), "device_relevance"]) == {"ppg_only"}
    assert uni.loc[uni.feature == "age", "device_relevance"].iloc[0] == "covariate"


def test_too_few_cases_is_reported_not_crashing():
    df = _table(n=6)
    res = evaluate(df, Config(**FAST))
    assert res["status"] == "insufficient_data"


def test_invasive_bp_note_appears_when_arterial_used():
    df = _table(signal=True)
    df["bp_invasive"] = True
    res = evaluate(df, Config(**FAST))
    assert any("arterial lines" in n for n in res["notes"])
