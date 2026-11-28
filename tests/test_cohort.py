import numpy as np
import pandas as pd

from pph_pipeline.cohort import build_manifest, candidate_cases, class_quota, make_label
from pph_pipeline.config import Config


def _cases(n=200):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "caseid": np.arange(1, n + 1),
        "department": rng.choice(["Gynecology", "General surgery"], n),
        "intraop_ebl": rng.choice([50, 200, 700, 1200, np.nan], n),
        "intraop_rbc": rng.choice([0, 0, 2, np.nan], n),
        "opstart": 600.0, "opend": 3000.0,
    })


def test_label_ebl_threshold_and_nan():
    cfg = Config(ebl_threshold_ml=500)
    lab = make_label(_cases(), cfg)
    assert set(lab.dropna().unique()) <= {0.0, 1.0}
    assert lab.isna().sum() > 0


def test_candidates_filter_department_and_balance():
    cfg = Config(department="Gynecology", max_cases=10)
    cands = candidate_cases(_cases(), cfg)
    assert cands["label"].isin([0, 1]).all()
    manifest = build_manifest(cands, cfg)
    q = class_quota(cfg)
    assert (manifest.label == 1).sum() <= q[1] * 3
    assert (manifest.label == 0).sum() <= q[0] * 3


def test_unknown_label_mode_raises():
    cfg = Config(label_mode="nope")
    try:
        make_label(_cases(), cfg)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_operation_filter_matches_text_in_operation_name():
    cases = _cases()
    cases["opname"] = np.where(cases["caseid"] % 4 == 0, "Cesarean section", "Hysterectomy")
    cfg = Config(operation="cesar")
    cands = candidate_cases(cases, cfg)
    assert len(cands) > 0 and set(cands["caseid"] % 4) == {0}
    assert len(candidate_cases(cases, Config())) > len(cands)
