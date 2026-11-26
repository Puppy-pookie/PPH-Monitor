"""Collapse window-level features into one baseline-normalised row per case.

Why case-level? VitalDB stores *total* estimated blood loss per surgery, not when
the bleeding happened, so v0.1 can only ask: "do cases with more blood loss show
a bigger haemodynamic-compensation signature?" Bleed timing (via lab Hb draws)
is the planned next step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

SIGNAL_COLS = ["ppg_ac", "ppg_pi", "hr_best", "sbp", "shock_index"]

SUMMARY_FEATURES = [
    "si_rise_max",        # biggest rise in shock index vs baseline
    "si_ge_0p9_frac",     # fraction of surgery with shock index >= 0.9
    "ac_drop_max_pct",    # biggest fall in PPG pulse amplitude vs baseline (%)
    "pi_drop_max_pct",    # biggest fall in perfusion index vs baseline (%)
    "hr_rise_max_pct",    # biggest HR rise vs baseline (%)
    "sbp_drop_max_pct",   # biggest SBP fall vs baseline (%)
    "ac_trend_pct_per_h", # linear trend of normalised PPG amplitude (% of baseline per hour)
    "si_trend_per_h",     # linear trend of shock index (units per hour)
]


def _pos_median(series: pd.Series) -> float:
    val = series.median()
    return float(val) if pd.notna(val) else np.nan


def _drop_pct(roll: pd.Series, base: float) -> float:
    if not np.isfinite(base) or base <= 0 or roll.notna().sum() == 0:
        return np.nan
    return float((1.0 - roll.min() / base) * 100.0)


def _trend_per_hour(t_s: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(t_s) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    slope = np.polyfit(t_s[ok], y[ok], 1)[0]  # units per second
    return float(slope * 3600.0)


def summarize_case(windows: pd.DataFrame, cfg: Config, op_start=None, op_end=None) -> dict | None:
    """Return a dict of case-level features, or None if the case has too little usable data."""
    w = windows.sort_values("t_start").reset_index(drop=True)
    if w.empty:
        return None

    if op_start is not None and np.isfinite(op_start):
        base = w[(w.t_start >= op_start - cfg.baseline_s) & (w.t_start < op_start)]
        body = w[w.t_start >= op_start]
        if op_end is not None and np.isfinite(op_end):
            body = body[body.t_start <= op_end]
    else:  # fallback if surgery times are missing: first baseline_s of recording
        t0 = float(w.t_start.min())
        base = w[w.t_start < t0 + cfg.baseline_s]
        body = w[w.t_start >= t0 + cfg.baseline_s]

    if base["ppg_ac"].notna().sum() < cfg.min_baseline_windows:
        return None
    if body["ppg_ac"].notna().sum() < cfg.min_body_windows:
        return None

    b = {c: _pos_median(base[c]) for c in SIGNAL_COLS}
    roll = body[SIGNAL_COLS].rolling(cfg.rolling_windows, min_periods=3).median()
    t = body["t_start"].to_numpy(dtype=float)

    si_ok = body["shock_index"].notna()
    out = {
        "base_ac": b["ppg_ac"],
        "base_si": b["shock_index"],
        "base_hr": b["hr_best"],
        "base_sbp": b["sbp"],
        "si_rise_max": float(roll["shock_index"].max() - b["shock_index"])
        if np.isfinite(b["shock_index"]) and roll["shock_index"].notna().any() else np.nan,
        "si_ge_0p9_frac": float((body.loc[si_ok, "shock_index"] >= 0.9).mean()) if si_ok.any() else np.nan,
        "ac_drop_max_pct": _drop_pct(roll["ppg_ac"], b["ppg_ac"]),
        "pi_drop_max_pct": _drop_pct(roll["ppg_pi"], b["ppg_pi"]),
        "hr_rise_max_pct": float((roll["hr_best"].max() / b["hr_best"] - 1.0) * 100.0)
        if np.isfinite(b["hr_best"]) and b["hr_best"] > 0 and roll["hr_best"].notna().any() else np.nan,
        "sbp_drop_max_pct": _drop_pct(roll["sbp"], b["sbp"]),
        "ac_trend_pct_per_h": _trend_per_hour(t, (body["ppg_ac"] / b["ppg_ac"] * 100.0).to_numpy())
        if b["ppg_ac"] > 0 else np.nan,
        "si_trend_per_h": _trend_per_hour(t, body["shock_index"].to_numpy(dtype=float)),
        "duration_min": float((t.max() - t.min() + cfg.window_s) / 60.0),
        "n_body_windows": int(len(body)),
        "sbp_coverage": float(body["sbp"].notna().mean()),
    }
    return out


def summarize_all(cases: pd.DataFrame, windows_by_case: dict, cfg: Config) -> pd.DataFrame:
    """Build the case-level table. `cases` needs: caseid, label, opstart, opend (+ optional covariates)."""
    rows = []
    for _, row in cases.iterrows():
        cid = int(row["caseid"])
        w = windows_by_case.get(cid)
        if w is None:
            continue
        feats = summarize_case(w, cfg, row.get("opstart", np.nan), row.get("opend", np.nan))
        if feats is None:
            continue
        rec = row.to_dict()
        rec.update(feats)
        rows.append(rec)
    return pd.DataFrame(rows)
