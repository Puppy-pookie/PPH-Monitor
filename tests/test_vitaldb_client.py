"""Tests the VitalDB loader against a real in-memory VitalFile (no network)."""
import numpy as np
import vitaldb

from pph_pipeline.config import Config
from pph_pipeline.vitaldb_client import load_case_signals

CFG = Config()


def _fake_file(with_ppg=True, seconds=120):
    vf = vitaldb.VitalFile()
    vf.dtstart, vf.dtend = 0.0, float(seconds)
    if with_ppg:
        n = int(seconds * 500)  # raw 500 Hz like SNUADC
        vf.add_track(CFG.ppg_track, [{"dt": 0.0, "val": np.sin(np.arange(n) / 500 * 2 * np.pi * 1.2).astype("float32")}], srate=500)
    hr = [{"dt": float(t), "val": 70.0} for t in range(0, seconds, 2)]
    sbp = [{"dt": float(t), "val": 115.0} for t in range(0, seconds, 2)]
    vf.add_track(CFG.hr_track, hr)
    vf.add_track(CFG.sbp_tracks[0], sbp)
    return vf


def test_load_case_signals_shapes():
    sig = load_case_signals(1, CFG, opener=lambda cid, tracks: _fake_file())
    assert sig is not None
    assert abs(len(sig["ppg"]) - 120 * CFG.target_fs) <= 2
    assert len(sig["hr"]) == 120
    assert set(sig["sbp"]) == set(CFG.sbp_tracks)
    assert np.isnan(sig["sbp"][CFG.sbp_tracks[2]]).all()  # missing track -> NaN, not a crash


def test_missing_ppg_returns_none():
    assert load_case_signals(2, CFG, opener=lambda cid, tracks: _fake_file(with_ppg=False)) is None
