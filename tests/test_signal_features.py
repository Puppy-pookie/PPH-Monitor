import numpy as np

from pph_pipeline.config import Config
from pph_pipeline.signal_features import extract_ppg_windows, ppg_window_features
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


def test_rejects_flat_and_nan_windows():
    assert ppg_window_features(np.full(3000, 5.0), FS, CFG) is None
    x = _window()
    x[:1500] = np.nan  # 50% missing
    assert ppg_window_features(x, FS, CFG) is None


def test_extract_windows_counts():
    x = np.concatenate([_window(seed=i) for i in range(4)])  # 120 s
    df = extract_ppg_windows(x, FS, CFG)
    assert len(df) == 4
    assert list(df["t_start"]) == [0.0, 30.0, 60.0, 90.0]
