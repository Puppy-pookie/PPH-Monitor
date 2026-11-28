import numpy as np

from pph_pipeline.config import Config
from pph_pipeline.signal_features import (
    assess_ppg_window, clip_fraction, extract_ppg_windows, extract_ppg_windows_qc,
    ppg_window_features, rmssd_adjacent,
)
from pph_pipeline.synthetic import synth_ppg

CFG = Config()
FS = CFG.target_fs


def _window(hr=72.0, ac=1.0, seconds=30, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    return synth_ppg(np.full(n, hr), np.full(n, ac), FS, rng)


def test_heart_rate_recovered():
    feats = ppg_window_features(_window(hr=72), FS, CFG)
    assert feats is not None
    assert abs(feats["ppg_hr"] - 72) < 2


def test_amplitude_scales_with_signal():
    a = ppg_window_features(_window(ac=1.0), FS, CFG)["ppg_ac"]
    b = ppg_window_features(_window(ac=0.5), FS, CFG)["ppg_ac"]
    assert 0.4 < b / a < 0.6


def test_rejects_flat_and_nan_windows_with_reasons():
    assert assess_ppg_window(np.full(3000, 5.0), FS, CFG) == (None, "flat")
    x = _window()
    x[:1500] = np.nan  # 50% missing
    assert assess_ppg_window(x, FS, CFG) == (None, "missing")


def test_clipped_window_is_rejected_with_reason():
    x = _window()
    x[x > np.percentile(x, 80)] = np.percentile(x, 80)  # flat-topped, as a saturated sensor would give
    assert clip_fraction(x, CFG.clip_run) > CFG.max_clip_frac
    assert assess_ppg_window(x, FS, CFG) == (None, "clipped")


def test_noisy_window_is_rejected_not_averaged_in():
    rng = np.random.default_rng(2)
    x = _window() + rng.normal(0, 0.5, int(30 * FS))
    feats, reason = assess_ppg_window(x, FS, CFG)
    assert feats is None and reason in {"low_sqi", "irregular", "few_beats"}


def test_clean_window_has_quality_metrics_and_no_flags():
    feats, reason = assess_ppg_window(_window(), FS, CFG)
    assert reason is None
    assert feats["ppg_beat_conf"] > 0.95
    assert feats["qc_flags"] == ""
    assert feats["ppg_clip_frac"] == 0.0


def test_drift_is_flagged_and_strict_qc_rejects_it():
    t = np.arange(int(30 * FS)) / FS
    x = _window() + 2.0 * np.sin(2 * np.pi * 0.2 * t)  # slow swing about twice the pulse amplitude
    feats, reason = assess_ppg_window(x, FS, CFG)
    assert reason is None and "drift" in feats["qc_flags"]
    assert assess_ppg_window(x, FS, Config(strict_qc=True)) == (None, "drift")


def test_motion_flag_uses_beat_confidence_threshold():
    cfg = Config(min_beat_conf=0.9999)  # threshold above even a clean window's score
    feats, reason = assess_ppg_window(_window(), FS, cfg)
    assert reason is None and "motion" in feats["qc_flags"]
    assert assess_ppg_window(_window(), FS, Config(min_beat_conf=0.9999, strict_qc=True)) == (None, "motion")


def test_rmssd_uses_adjacent_beats_only():
    ibi = np.array([0.8, 0.9, 2.5, 0.7, 0.7, 0.7, 0.7])
    ok = np.array([True, True, False, True, True, True, True])
    # adjacent valid pairs: (0.8,0.9)=0.1 and three zero differences -> sqrt(0.01/4) = 50 ms.
    # The old method differenced across the removed beat (0.9 -> 0.7) and gave about 100 ms.
    assert abs(rmssd_adjacent(ibi, ok) - 50.0) < 1e-6


def test_extract_windows_counts_and_rejection_log():
    good = [_window(seed=i) for i in range(3)]
    bad = np.full(int(30 * FS), 5.0)  # flat window
    x = np.concatenate(good + [bad])
    accepted, rejects, n = extract_ppg_windows_qc(x, FS, CFG)
    assert n == 4 and len(accepted) == 3
    assert list(rejects["reason"]) == ["flat"] and list(rejects["t_start"]) == [90.0]
    assert list(extract_ppg_windows(x, FS, CFG)["t_start"]) == [0.0, 30.0, 60.0]
