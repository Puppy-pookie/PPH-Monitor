"""Window-level features from the PPG waveform and numeric vital signs.

Everything here is pure numpy/pandas/scipy so it can be unit-tested offline.

Important: VitalDB PPG (SNUADC/PLETH) is in arbitrary units. Absolute values
are not comparable between patients; the pipeline therefore uses changes
relative to each patient's own baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

from .config import Config

PPG_COLUMNS = [
    "ppg_ac", "ppg_dc", "ppg_pi", "ppg_rise_ms",
    "ppg_hr", "ppg_ibi_rmssd_ms", "ppg_sqi", "n_beats",
]


def ppg_window_features(x, fs: float, cfg: Config) -> dict | None:
    """Beat-level PPG features for one window. Returns None if the window is unusable."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    nan = np.isnan(x)
    if nan.all() or nan.mean() > cfg.max_nan_frac:
        return None
    if nan.any():  # fill short gaps by linear interpolation
        idx = np.arange(x.size)
        x = np.interp(idx, idx[~nan], x[~nan])
    if np.ptp(x) < 1e-9:  # flat line / disconnected sensor
        return None

    dc = float(np.mean(x))
    sos = butter(3, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x)
    sd = float(np.std(xf))
    if sd <= 0:
        return None

    peaks, _ = find_peaks(xf, distance=max(1, int(0.33 * fs)), prominence=0.25 * sd)
    if len(peaks) < cfg.min_beats:
        return None

    amps, rises = [], []
    prev = None
    for p in peaks:
        lo = prev if prev is not None else max(0, p - int(0.8 * fs))
        seg = xf[lo:p]
        if seg.size >= 2:
            valley = lo + int(np.argmin(seg))
            amps.append(xf[p] - xf[valley])
            rises.append((p - valley) / fs)
        prev = p
    if len(amps) < 3:
        return None

    ibi = np.diff(peaks) / fs
    ok = (ibi > 0.33) & (ibi < 1.5)
    if ok.sum() < 3:
        return None
    med = float(np.median(ibi[ok]))
    ok &= np.abs(ibi - med) / med < 0.3
    if ok.sum() < 3:
        return None
    good = ibi[ok]

    amps = np.asarray(amps)
    mean_amp = float(np.mean(amps))
    cv = float(np.std(amps) / mean_amp) if mean_amp > 0 else 1.0
    sqi = float(ok.mean() * (1.0 - min(cv, 1.0)))
    if sqi < cfg.min_sqi:
        return None

    ac = float(np.median(amps))
    return {
        "ppg_ac": ac,
        "ppg_dc": dc,
        "ppg_pi": ac / abs(dc) * 100.0 if abs(dc) > 1e-9 else np.nan,
        "ppg_rise_ms": float(np.median(rises)) * 1000.0,
        "ppg_hr": 60.0 / float(np.median(good)),
        "ppg_ibi_rmssd_ms": float(np.sqrt(np.mean(np.diff(good) ** 2)) * 1000.0),
        "ppg_sqi": sqi,
        "n_beats": int(len(peaks)),
    }


def extract_ppg_windows(x, fs: float, cfg: Config) -> pd.DataFrame:
    """Slice a PPG array (sampled at fs Hz, starting at t=0) into fixed windows."""
    x = np.asarray(x, dtype=float)
    spw = int(round(cfg.window_s * fs))
    rows = []
    for i in range(x.size // spw):
        feats = ppg_window_features(x[i * spw:(i + 1) * spw], fs, cfg)
        if feats is not None:
            feats["t_start"] = i * cfg.window_s
            rows.append(feats)
    if not rows:
        return pd.DataFrame(columns=["t_start"] + PPG_COLUMNS)
    return pd.DataFrame(rows)[["t_start"] + PPG_COLUMNS]


def numeric_windows(v, interval_s: float, cfg: Config, lo: float, hi: float) -> pd.Series:
    """Median of a numeric vital sign (e.g. HR) per window; implausible values dropped."""
    v = np.asarray(v, dtype=float).copy()
    v[(v < lo) | (v > hi)] = np.nan
    per_window = int(round(cfg.window_s / interval_s))
    if per_window < 1 or v.size < per_window:
        return pd.Series(dtype=float)
    n_win = v.size // per_window
    s = pd.Series(v[: n_win * per_window]).groupby(np.arange(n_win * per_window) // per_window).median()
    s.index = s.index.to_numpy() * cfg.window_s
    s.index.name = "t_start"
    return s


def pick_sbp(candidates: dict[str, pd.Series], min_coverage: float = 0.3) -> tuple[pd.Series, str | None]:
    """Pick one SBP source per case: first in priority order with enough coverage,
    else whichever has the most data. Mixing sources inside a case is avoided."""
    best_name, best_cov = None, 0.0
    for name, s in candidates.items():
        cov = float(s.notna().mean()) if len(s) else 0.0
        if cov >= min_coverage:
            return s, name
        if cov > best_cov:
            best_name, best_cov = name, cov
    if best_name is None:
        return pd.Series(dtype=float), None
    return candidates[best_name], best_name


def build_case_windows(ppg, ppg_fs, hr, sbp_by_track, numeric_interval_s, cfg: Config):
    """Combine PPG and numeric features into one row per window.

    Returns (DataFrame, sbp_source_name_or_None).
    Columns: t_start, ppg_*, hr, sbp, hr_best, shock_index.
    """
    ppg_w = extract_ppg_windows(ppg, ppg_fs, cfg).set_index("t_start")
    hr_w = numeric_windows(hr, numeric_interval_s, cfg, 20, 220).rename("hr")
    sbp_cands = {
        name: numeric_windows(v, numeric_interval_s, cfg, 40, 260)
        for name, v in sbp_by_track.items()
    }
    sbp_w, sbp_src = pick_sbp(sbp_cands)
    sbp_w = sbp_w.rename("sbp")

    df = pd.concat([ppg_w, hr_w, sbp_w], axis=1).sort_index()
    df.index.name = "t_start"
    for col in PPG_COLUMNS + ["hr", "sbp"]:
        if col not in df.columns:
            df[col] = np.nan
    df["hr_best"] = df["hr"].fillna(df["ppg_hr"])
    df["shock_index"] = df["hr_best"] / df["sbp"]
    return df.reset_index(), sbp_src
