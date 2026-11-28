"""Window-level features from the PPG waveform and numeric vital signs.

Everything here is pure numpy/pandas/scipy so it can be unit-tested offline.

Important: VitalDB PPG (SNUADC/PLETH) is in arbitrary units. Absolute values are not
comparable between patients; the pipeline therefore uses changes relative to each
patient's own baseline.

Every rejected window gets a recorded reason (see REJECT_REASONS), and accepted windows
carry quality flags, so a drop in signal can be told apart from a sensor problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

from .bp import build_bp_columns
from .config import Config

PPG_COLUMNS = [
    "ppg_ac", "ppg_dc", "ppg_pi", "ppg_rise_ms", "ppg_hr", "ppg_ibi_rmssd_ms",
    "ppg_sqi", "ppg_beat_conf", "ppg_drift_ratio", "ppg_clip_frac", "ppg_amp_cv",
    "n_beats", "qc_flags",
]

REJECT_REASONS = ["missing", "flat", "clipped", "few_beats", "irregular", "low_sqi", "motion", "drift"]


def clip_fraction(x: np.ndarray, run: int = 5) -> float:
    """Share of samples that sit in flat runs at the window maximum or minimum (saturation)."""
    if x.size == 0:
        return 0.0
    mask = (x == x.max()) | (x == x.min())
    if not mask.any():
        return 0.0
    padded = np.concatenate(([0], mask.astype(int), [0]))
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    lengths = ends - starts
    return float(lengths[lengths >= run].sum() / x.size)


def beat_confidence(xf: np.ndarray, peaks: np.ndarray, fs: float) -> float:
    """Mean correlation of each beat with the window's median beat shape (1 = clean, low = artifact)."""
    pre, post = int(0.3 * fs), int(0.5 * fs)
    segs = [xf[p - pre:p + post] for p in peaks if p - pre >= 0 and p + post <= len(xf)]
    if len(segs) < 5:
        return float("nan")
    s = np.vstack(segs)
    s = (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-12)
    tmpl = np.median(s, axis=0)
    tmpl = (tmpl - tmpl.mean()) / (tmpl.std() + 1e-12)
    return float(np.mean((s * tmpl).mean(axis=1)))


def rmssd_adjacent(ibi: np.ndarray, ok: np.ndarray) -> float:
    """RMSSD (ms) using only differences between ADJACENT valid beats.

    v0.2 removed bad intervals first and then differenced, which could compare beats
    that were not neighbours. Here the original beat order is kept.
    """
    if ibi.size < 2:
        return float("nan")
    diffs = np.diff(ibi)
    pair_ok = ok[:-1] & ok[1:]
    if pair_ok.sum() < 3:
        return float("nan")
    return float(np.sqrt(np.mean(diffs[pair_ok] ** 2)) * 1000.0)


def assess_ppg_window(x, fs: float, cfg: Config):
    """Quality-control and feature extraction for one window.

    Returns (features, None) if usable, or (None, reason) if rejected.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None, "missing"
    nan = np.isnan(x)
    if nan.all() or nan.mean() > cfg.max_nan_frac:
        return None, "missing"
    if nan.any():  # fill short gaps by linear interpolation
        idx = np.arange(x.size)
        x = np.interp(idx, idx[~nan], x[~nan])
    if np.ptp(x) < 1e-9:  # flat line / disconnected sensor
        return None, "flat"
    if clip_fraction(x, cfg.clip_run) > cfg.max_clip_frac:
        return None, "clipped"
    clip = clip_fraction(x, cfg.clip_run)

    dc = float(np.mean(x))
    sos = butter(3, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x)
    sd = float(np.std(xf))
    if sd <= 0:
        return None, "flat"

    peaks, _ = find_peaks(xf, distance=max(1, int(0.33 * fs)), prominence=0.25 * sd)
    if len(peaks) < cfg.min_beats:
        return None, "few_beats"

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
        return None, "few_beats"

    ibi = np.diff(peaks) / fs
    ok = (ibi > 0.33) & (ibi < 1.5)
    if ok.sum() < 3:
        return None, "irregular"
    med = float(np.median(ibi[ok]))
    ok &= np.abs(ibi - med) / med < 0.3
    if ok.sum() < 3:
        return None, "irregular"
    good = ibi[ok]

    amps = np.asarray(amps)
    mean_amp = float(np.mean(amps))
    cv = float(np.std(amps) / mean_amp) if mean_amp > 0 else 1.0
    sqi = float(ok.mean() * (1.0 - min(cv, 1.0)))
    if sqi < cfg.min_sqi:
        return None, "low_sqi"

    ac = float(np.median(amps))
    conf = beat_confidence(xf, peaks, fs)
    low = sosfiltfilt(butter(2, 0.5, btype="low", fs=fs, output="sos"), x)
    drift = float(np.ptp(low) / ac) if ac > 0 else float("nan")

    flags = []
    if np.isfinite(conf) and conf < cfg.min_beat_conf:
        flags.append("motion")
    if np.isfinite(drift) and drift > cfg.max_drift_ratio:
        flags.append("drift")
    if cfg.strict_qc:
        if "motion" in flags:
            return None, "motion"
        if "drift" in flags:
            return None, "drift"

    return {
        "ppg_ac": ac,
        "ppg_dc": dc,
        "ppg_pi": ac / abs(dc) * 100.0 if abs(dc) > 1e-9 else np.nan,
        "ppg_rise_ms": float(np.median(rises)) * 1000.0,
        "ppg_hr": 60.0 / float(np.median(good)),
        "ppg_ibi_rmssd_ms": rmssd_adjacent(ibi, ok),
        "ppg_sqi": sqi,
        "ppg_beat_conf": conf,
        "ppg_drift_ratio": drift,
        "ppg_clip_frac": clip,
        "ppg_amp_cv": cv,
        "n_beats": int(len(peaks)),
        "qc_flags": ";".join(flags),
    }, None


def ppg_window_features(x, fs: float, cfg: Config) -> dict | None:
    """Backward-compatible wrapper: features for one window, or None if rejected."""
    feats, _ = assess_ppg_window(x, fs, cfg)
    return feats


def extract_ppg_windows_qc(x, fs: float, cfg: Config):
    """Slice a PPG array (sampled at fs Hz, starting at t=0) into fixed windows.

    Returns (accepted_windows DataFrame, rejected DataFrame[t_start, reason], n_windows).
    """
    x = np.asarray(x, dtype=float)
    spw = int(round(cfg.window_s * fs))
    n_windows = x.size // spw
    rows, rejects = [], []
    for i in range(n_windows):
        feats, reason = assess_ppg_window(x[i * spw:(i + 1) * spw], fs, cfg)
        t0 = i * cfg.window_s
        if feats is None:
            rejects.append({"t_start": t0, "reason": reason})
        else:
            feats["t_start"] = t0
            rows.append(feats)
    cols = ["t_start"] + PPG_COLUMNS
    accepted = pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)
    return accepted, pd.DataFrame(rejects, columns=["t_start", "reason"]), n_windows


def extract_ppg_windows(x, fs: float, cfg: Config) -> pd.DataFrame:
    """Backward-compatible wrapper: accepted windows only."""
    return extract_ppg_windows_qc(x, fs, cfg)[0]


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


@dataclass
class CaseWindows:
    """Everything computed for one case before a blood-pressure mode is chosen."""
    windows: pd.DataFrame            # one row per 30 s window (PPG features, hr, sbp_art, sbp_cuff)
    bp_meta: dict                    # which BP tracks exist and how well they cover the record
    qc: dict                         # PPG quality summary (counts by rejection reason, flags)
    rejects: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["t_start", "reason"]))


def qc_summary(accepted: pd.DataFrame, rejects: pd.DataFrame, n_windows: int) -> dict:
    out = {
        "qc_windows_total": int(n_windows),
        "qc_windows_accepted": int(len(accepted)),
        "qc_accept_frac": float(len(accepted) / n_windows) if n_windows else float("nan"),
    }
    counts = rejects["reason"].value_counts().to_dict() if len(rejects) else {}
    for r in REJECT_REASONS:
        out[f"qc_reject_{r}"] = int(counts.get(r, 0))
    flags = accepted["qc_flags"].fillna("") if len(accepted) else pd.Series(dtype=str)
    out["qc_flag_motion"] = int(flags.str.contains("motion").sum()) if len(flags) else 0
    out["qc_flag_drift"] = int(flags.str.contains("drift").sum()) if len(flags) else 0
    out["qc_beat_conf_median"] = float(accepted["ppg_beat_conf"].median()) if len(accepted) else float("nan")
    out["qc_sqi_median"] = float(accepted["ppg_sqi"].median()) if len(accepted) else float("nan")
    return out


def build_case_windows(ppg, ppg_fs, hr, sbp_by_track, numeric_interval_s, cfg: Config) -> CaseWindows:
    """Combine PPG and numeric features into one row per window.

    Blood pressure is stored in two separate columns (sbp_art, sbp_cuff) and the choice
    between them is made later (see bp.apply_bp_mode), so switching mode never needs a
    re-download and an invasive reading is never silently used in a non-invasive analysis.
    """
    accepted, rejects, n_windows = extract_ppg_windows_qc(ppg, ppg_fs, cfg)
    ppg_w = accepted.set_index("t_start")
    hr_w = numeric_windows(hr, numeric_interval_s, cfg, 20, 220).rename("hr")
    sbp_art, sbp_cuff, bp_meta = build_bp_columns(sbp_by_track, numeric_interval_s, cfg)

    df = pd.concat([ppg_w, hr_w, sbp_art, sbp_cuff], axis=1).sort_index()
    df.index.name = "t_start"
    for col in PPG_COLUMNS + ["hr", "sbp_art", "sbp_cuff"]:
        if col not in df.columns:
            df[col] = np.nan
    return CaseWindows(df.reset_index(), bp_meta, qc_summary(accepted, rejects, n_windows), rejects)
