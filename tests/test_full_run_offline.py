"""Exercises select -> download -> summarize -> evaluate with a fake VitalDB (no network)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pph_pipeline import pipeline
from pph_pipeline.config import Config
from pph_pipeline.synthetic import make_cuff_track, make_synthetic_case

N_CASES = 60
FAST = dict(cv_repeats=3, perm_repeats=2, n_permutations=10, n_bootstrap=50)
CALLS = []


def _fake_clinical(cfg, force=False):
    rows = []
    for cid in range(1, N_CASES + 1):
        high = cid % 2 == 0
        rows.append({"caseid": cid, "department": "Gynecology", "opname": "Cesarean section" if high else "Myomectomy",
                     "intraop_ebl": 900.0 if high else 100.0, "intraop_rbc": 0.0,
                     "opstart": 600.0, "opend": 2400.0, "age": 50.0, "sex": "F", "asa": 2.0,
                     "emop": 0.0, "intraop_phe": 0.0, "intraop_eph": 0.0, "intraop_epi": 0.0})
    return pd.DataFrame(rows)


def _fake_loader(caseid, cfg):
    CALLS.append(caseid)
    if caseid % 3 == 0:  # simulate cases with no PPG track
        return None
    rng = np.random.default_rng(caseid)
    ppg, hr, sbp = make_synthetic_case(caseid % 2 == 0, cfg, rng)
    art, fem, cuff = cfg.sbp_tracks
    return {"ppg": ppg, "ppg_fs": cfg.target_fs, "hr": hr, "numeric_interval_s": cfg.numeric_interval_s,
            "sbp": {art: sbp, fem: np.full_like(sbp, np.nan), cuff: make_cuff_track(sbp, rng)}}


def _cfg(tmp_path, **kw):
    return Config(data_dir=tmp_path / "data", output_dir=tmp_path / "out", max_cases=12,
                  department="Gynecology", **FAST, **kw)


def _setup(monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(pipeline, "load_clinical_table", _fake_clinical)
    monkeypatch.setattr(pipeline, "load_case_signals", _fake_loader)


def test_full_offline_run_and_outputs(tmp_path, monkeypatch):
    _setup(monkeypatch)
    cfg = _cfg(tmp_path)
    pipeline.step_select(cfg)
    status = pipeline.step_download(cfg)
    assert "no_ppg_track" in set(status["status"])          # those cases are skipped, not fatal
    assert len(list(cfg.features_dir.glob("windows_*.csv.gz"))) == 12
    assert len(list(cfg.features_dir.glob("case_*_meta.json"))) == 12

    table, windows, excl = pipeline.step_summarize(cfg)
    assert len(table) == 12 and set(table["label"]) == {0, 1}
    assert set(table["bp_type"]) == {"arterial"} and table["bp_coverage"].min() > 0.9
    run_dir = Path(pipeline.step_evaluate(cfg, table, windows, mode="vitaldb", exclusions=excl))
    res = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert res["mode"] == "vitaldb" and res["results"]["status"] == "ok"
    for name in ("report.md", "environment.json", "exclusions.csv", "qc_summary.csv", "case_features.csv"):
        assert (run_dir / name).exists(), name
    assert "Run summary" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_cached_cases_keep_their_blood_pressure_source(tmp_path, monkeypatch):
    """Regression for the v0.2 bug: cached cases lost their BP source metadata."""
    _setup(monkeypatch)
    cfg = _cfg(tmp_path)
    pipeline.step_select(cfg)
    first = pipeline.step_download(cfg)
    CALLS.clear()
    second = pipeline.step_download(cfg)
    assert CALLS == []                                       # nothing downloaded again
    used = second[second["status"] == "cached"]
    assert len(used) == 12 and used["bp_source"].notna().all()
    assert set(used["bp_type"]) == {"arterial"}
    assert set(first[first["status"] == "downloaded"]["bp_source"]) == set(used["bp_source"])


def test_switching_bp_source_needs_no_redownload(tmp_path, monkeypatch):
    _setup(monkeypatch)
    pipeline.step_select(_cfg(tmp_path))
    pipeline.step_download(_cfg(tmp_path))
    CALLS.clear()
    cuff_cfg = _cfg(tmp_path, bp_source="cuff")
    status = pipeline.step_download(cuff_cfg)
    table, _, _ = pipeline.step_summarize(cuff_cfg)
    assert CALLS == [] and set(status["bp_type"].dropna()) == {"cuff"}
    assert set(table["bp_type"]) == {"cuff"} and not table["bp_invasive"].any()


def test_stale_cache_is_rebuilt_when_quality_settings_change(tmp_path, monkeypatch):
    _setup(monkeypatch)
    cfg = _cfg(tmp_path)
    pipeline.step_select(cfg)
    pipeline.step_download(cfg)
    CALLS.clear()
    pipeline.step_download(_cfg(tmp_path, strict_qc=True))   # strict QC changes which windows exist
    assert len(CALLS) > 0
    # an old-style cache (windows file but no metadata) is also rebuilt
    cid = next(iter(set(range(1, N_CASES + 1)) - {c for c in CALLS if c % 3 == 0}))
    pipeline._meta_path(cfg, cid).unlink(missing_ok=True)
    assert pipeline._load_case(cfg, cid) is None


def test_operation_filter_reaches_the_manifest(tmp_path, monkeypatch):
    _setup(monkeypatch)
    cfg = _cfg(tmp_path, operation="Cesarean")
    manifest = pipeline.step_select(cfg)
    assert set(manifest["label"]) == {1}                      # only the cesarean cases are 'high loss' here
    assert (manifest["opname"] == "Cesarean section").all()
