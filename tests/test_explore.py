import numpy as np
import pandas as pd

from pph_pipeline.config import Config
from pph_pipeline.pipeline import explore_report


def _cases(n=120):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "caseid": np.arange(1, n + 1),
        "department": rng.choice(["Gynecology", "General surgery", "Urology"], n),
        "intraop_ebl": rng.choice([50, 200, 400, 700, 1200], n),
        "intraop_rbc": rng.choice([0, 0, 2], n),
        "aline1": rng.choice(["x", None], n),
        "opstart": 600.0, "opend": 3000.0,
    })


def test_explore_report_lists_departments_and_counts():
    cfg = Config(max_cases=10, department="Gyn")
    text = explore_report(_cases(), cfg)
    assert "Clinical table: 120 cases" in text
    for name in ("Gynecology", "General surgery", "Urology"):
        assert name in text
    assert "EBL >=  500 mL" in text and "current --ebl-threshold" in text
    assert "usable candidates: label=1" in text
    assert "asks for 5 higher-loss and 5 lower-loss" in text
