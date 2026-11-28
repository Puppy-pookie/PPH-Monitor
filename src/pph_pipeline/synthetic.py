"""Synthetic cohort for testing the pipeline offline.

THIS IS NOT REAL DATA. 'Bleeding' cases get a gradual PPG-amplitude fall, heart-rate
rise and a late blood-pressure fall, so a working pipeline should find signal.
Use it to confirm your install works and to unit-test feature extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .signal_features import build_case_windows

SBP_TRACK = "Solar8000/ART_SBP"
CUFF_TRACK = "Solar8000/NIBP_SBP"


def _pulse_shape(phase: np.ndarray) -> np.ndarray:
    p = np.mod(phase, 1.0)
    s = np.exp(-((p - 0.25) / 0.08) ** 2) + 0.35 * np.exp(-((p - 0.55) / 0.12) ** 2)
    return s


def synth_ppg(hr_bpm: np.ndarray, ac: np.ndarray, fs: float, rng: np.random.Generator,
              dc: float = 100.0, noise: float = 0.02) -> np.ndarray:
    """PPG-like waveform: per-sample heart rate (bpm) and pulse amplitude arrays."""
    phase = np.cumsum(hr_bpm / 60.0 / fs)
    grid = _pulse_shape(np.linspace(0, 1, 1000, endpoint=False))
    norm = grid.max() - grid.min()
    shape = (_pulse_shape(phase) - grid.min()) / norm
    t = np.arange(hr_bpm.size) / fs
    wander = 0.05 * ac * np.sin(2 * np.pi * 0.25 * t)  # respiratory baseline wander
    return dc + ac * shape + wander + rng.normal(0, noise * ac, hr_bpm.size)


def make_synthetic_case(is_bleeder: bool, cfg: Config, rng: np.random.Generator,
                        op_start: float = 600.0, op_len: float = 1800.0):
    fs = cfg.target_fs
    total = op_start + op_len + 100.0
    n = int(total * fs)
    t = np.arange(n) / fs

    hr0, sbp0, ac0 = rng.normal(75, 8), rng.normal(115, 10), rng.uniform(0.8, 2.0)
    drift = np.cumsum(rng.normal(0, 0.0004, n))
    f = np.zeros(n)
    if is_bleeder:
        onset = op_start + rng.uniform(300, 900)
        sev = rng.uniform(0.5, 1.0)
        f = np.clip((t - onset) / 900.0, 0, 1) * sev
    hr = hr0 * (1 + 0.30 * f + 0.5 * drift)
    ac = ac0 * (1 - 0.55 * f + drift)
    sbp = sbp0 * (1 - 0.30 * f ** 3 + 0.5 * drift)

    ppg = synth_ppg(hr, ac, fs, rng)
    step = int(fs * cfg.numeric_interval_s)
    hr_num = hr[::step] + rng.normal(0, 1.0, hr[::step].size)
    sbp_num = sbp[::step] + rng.normal(0, 2.0, sbp[::step].size)
    return ppg, hr_num, sbp_num


def make_cuff_track(sbp_num: np.ndarray, rng: np.random.Generator, every_s: int = 180) -> np.ndarray:
    """Intermittent cuff readings: one value every few minutes, NaN in between."""
    cuff = np.full_like(sbp_num, np.nan, dtype=float)
    cuff[::every_s] = sbp_num[::every_s] + rng.normal(0, 3.0, cuff[::every_s].size)
    return cuff


def make_synthetic_cohort(cfg: Config, n_cases: int = 24, seed: int | None = None):
    """Returns (cases_df, case_windows_by_id) where each value is a CaseWindows object."""
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    rows, windows = [], {}
    for i in range(n_cases):
        is_bleeder = i % 2 == 0
        ppg, hr, sbp = make_synthetic_case(is_bleeder, cfg, rng)
        cuff = make_cuff_track(sbp, rng)
        built = build_case_windows(ppg, cfg.target_fs, hr, {SBP_TRACK: sbp, CUFF_TRACK: cuff},
                                   cfg.numeric_interval_s, cfg)
        cid = i + 1
        windows[cid] = built
        rows.append({
            "caseid": cid,
            "label": int(is_bleeder),
            "opstart": 600.0,
            "opend": 2400.0,
            "age": float(rng.integers(25, 80)),
            "sex": "F" if rng.random() < 0.5 else "M",
            "asa": float(rng.integers(1, 4)),
            "emop": 0.0,
            "intraop_ebl": float(rng.uniform(600, 1500) if is_bleeder else rng.uniform(50, 300)),
            "intraop_phe": float(rng.exponential(100)),
            "intraop_eph": float(rng.exponential(5)),
            "intraop_epi": 0.0,
        })
    return pd.DataFrame(rows), windows
