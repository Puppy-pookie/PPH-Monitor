import numpy as np
import pandas as pd

from pph_pipeline.bp import apply_bp_mode, build_bp_columns, cuff_forward_fill
from pph_pipeline.config import Config

ART, FEM, CUFF = "Solar8000/ART_SBP", "Solar8000/FEM_SBP", "Solar8000/NIBP_SBP"
N = 1800  # 30 minutes at 1 s


def _cuff_every(seconds, value=110.0):
    v = np.full(N, np.nan)
    v[::seconds] = value
    return v


def _windows(cfg, art_cov=True, cuff_cov=True, meta_source=None):
    tracks = {}
    if art_cov:
        tracks[ART] = np.full(N, 120.0)
    if cuff_cov:
        tracks[CUFF] = _cuff_every(300)
    a, c, meta = build_bp_columns(tracks, 1.0, cfg)
    w = pd.DataFrame({"t_start": np.arange(0, N, 30.0)})
    w = w.join(a.reset_index(drop=True)).join(c.reset_index(drop=True))
    w["hr"], w["ppg_hr"] = 80.0, 80.0
    return w, meta


def test_cuff_carry_forward_is_limited():
    v = _cuff_every(300)
    short = cuff_forward_fill(v, 1.0, 120.0)
    assert np.isfinite(short).mean() < 0.5      # about 120 of every 300 seconds
    long = cuff_forward_fill(v, 1.0, 600.0)
    assert np.isfinite(long).mean() > 0.95


def test_coverage_reflects_carry_forward_limit():
    _, _, meta_short = build_bp_columns({CUFF: _cuff_every(300)}, 1.0, Config(cuff_carry_s=120))
    _, _, meta_long = build_bp_columns({CUFF: _cuff_every(300)}, 1.0, Config(cuff_carry_s=600))
    assert meta_short["cuff_coverage"] <= 0.55 and meta_long["cuff_coverage"] > 0.9
    assert meta_long["cuff_raw_readings"] == len(range(0, N, 300))


def test_arterial_mode_never_substitutes_cuff():
    cfg = Config(bp_source="arterial")
    w, meta = _windows(cfg, art_cov=False, cuff_cov=True)
    w2, info = apply_bp_mode(w, meta, cfg)
    assert info["bp_type"] == "none" and info["bp_source"] is None
    assert w2["sbp"].isna().all() and w2["shock_index"].isna().all()


def test_cuff_mode_never_substitutes_arterial():
    cfg = Config(bp_source="cuff")
    w, meta = _windows(cfg, art_cov=True, cuff_cov=False)
    w2, info = apply_bp_mode(w, meta, cfg)
    assert info["bp_type"] == "none" and w2["sbp"].isna().all()


def test_either_prefers_arterial_and_records_type():
    cfg = Config(bp_source="either")
    w, meta = _windows(cfg)
    _, info = apply_bp_mode(w, meta, cfg)
    assert info["bp_type"] == "arterial" and info["bp_source"] == ART and info["bp_invasive"] is True
    w, meta = _windows(cfg, art_cov=False)
    w2, info = apply_bp_mode(w, meta, cfg)
    assert info["bp_type"] == "cuff" and info["bp_source"] == CUFF and info["bp_invasive"] is False
    assert info["bp_carry_forward_s"] == cfg.cuff_carry_s
    assert abs(w2["shock_index"].dropna().iloc[0] - 80.0 / 110.0) < 1e-9


def test_source_coverage_and_missing_counts_are_recorded():
    cfg = Config(bp_source="cuff", cuff_carry_s=120)
    w, meta = _windows(cfg, art_cov=False)
    _, info = apply_bp_mode(w, meta, cfg)
    assert info["bp_valid_windows"] + info["bp_missing_windows"] == len(w)
    assert info["bp_missing_windows"] > 0 and 0 < info["bp_coverage"] < 1


def test_bad_mode_raises():
    w, meta = _windows(Config())
    try:
        apply_bp_mode(w, meta, Config(bp_source="nope"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
