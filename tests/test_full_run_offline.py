"""Exercises select -> download -> summarize -> evaluate with a fake VitalDB (no network)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pph_pipeline import pipeline
from pph_pipeline.config import Config
from pph_pipeline.synthetic import make_synthetic_case

N_CASES = 60


def _fake_clinical(cfg, force=False):
    rows = []
    for cid in range(1, N_CASES + 1):
        high = cid % 2 == 0
        rows.append({"caseid": cid, "department": "Gynecology",
                     "intraop_ebl": 900.0 if high else 100.0, "intraop_rbc": 0.0,
                     "opstart": 600.0, "opend": 2400.0, "age": 50.0, "sex": "F", "asa": 2.0,
                     "emop": 0.0, "intraop_phe": 0.0, "intraop_eph": 0.0, "intraop_epi": 0.0})
    return pd.DataFrame(rows)


CALLS = []


def _fake_loader(caseid, cfg):
    CALLS.append(caseid)
    if caseid % 3 == 0:  # simulate cases with no PPG track
        return None
    rng = np.random.default_rng(caseid)
    ppg, hr, sbp = make_synthetic_case(caseid % 2 == 0, cfg, rng)
    nan = np.full_like(sbp, np.nan)
    return {"ppg": ppg, "ppg_fs": cfg.target_fs, "hr": hr, "numeric_interval_s": cfg.numeric_interval_s,
            "sbp": {cfg.sbp_tracks[0]: sbp, cfg.sbp_tracks[1]: nan, cfg.sbp_tracks[2]: nan}}


def test_full_offline_run(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_clinical_table", _fake_clinical)
    monkeypatch.setattr(pipeline, "load_case_signals", _fake_loader)
    cfg = Config(data_dir=tmp_path / "data", output_dir=tmp_path / "out", max_cases=12, department="Gynecology")

    pipeline.step_select(cfg)
    status = pipeline.step_download(cfg)
    assert "no_ppg_track" in set(status["status"])          # those cases are skipped, not fatal
    assert len(list(cfg.features_dir.glob("windows_*.csv.gz"))) == 12

    # second run must not touch the network again: good cases are cached, bad ones remembered
    CALLS.clear()
    status2 = pipeline.step_download(cfg)
    assert CALLS == []
    assert set(status2["status"]) <= {"cached", "no_ppg_track"}

    table, windows = pipeline.step_summarize(cfg)
    assert len(table) == 12 and set(table["label"]) == {0, 1}
    run_dir = pipeline.step_evaluate(cfg, table, windows, mode="vitaldb")
    res = json.loads((Path(run_dir) / "results.json").read_text(encoding="utf-8"))
    assert res["mode"] == "vitaldb"
    assert res["results"]["status"] == "ok"
