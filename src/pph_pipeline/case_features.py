"""Collapse window-level features into one baseline-normalised row per case.

Why case-level? VitalDB stores *total* estimated blood loss per surgery, not when
the bleeding happened, so this version can only ask: "do cases with more blood loss
show a bigger haemodynamic-compensation signature?" Time-aware analysis and bleed
timing are planned next (see CHANGELOG / roadmap).

Every feature is tagged with what a NON-INVASIVE device would need to measure it
(FEATURE_RELEVANCE), because the end goal is a device, not a hospital-monitor model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

SIGNAL_COLS = ["ppg_ac", "ppg_pi", "ppg_hr", "hr_best", "sbp", "shock_index"]

SUMMARY_FEATURES = [
    "si_rise_max",          # biggest rise in shock index vs baseline
    "si_ge_0p9_frac",       # fraction of surgery with shock index >= 0.9
    "ac_drop_max_pct",      # biggest fall in PPG pulse amplitude vs baseline (%)
    "pi_drop_max_pct",      # biggest fall in perfusion index vs baseline (%)
    "hr_rise_max_pct",      # biggest rise in monitor heart rate vs baseline (%)
    "ppg_hr_rise_max_pct",  # biggest rise in PPG-derived pulse rate vs baseline (%)
    "sbp_drop_max_pct",     # biggest SBP fall vs baseline (%)
    "ac_trend_pct_per_h",   # linear trend of normalised PPG amplitude (% of baseline per hour)
    "si_trend_per_h",       # linear trend of shock index (units per hour)
]

# What a non-invasive device would need to measure each feature.
#   ppg_only    : an optical pulse sensor alone
#   needs_bp    : also needs blood pressure (non-invasive only if from a cuff)
#   monitor_hr  : heart rate from the hospital monitor (ECG); a device would use PPG pulse rate instead
FEATURE_RELEVANCE = {
    "ac_drop_max_pct": "ppg_only",
    "pi_drop_max_pct": "ppg_only",
    "ac_trend_pct_per_h": "ppg_only",
    "ppg_hr_rise_max_pct": "ppg_only",
    "hr_rise_max_pct": "monitor_hr",
    "si_rise_max": "needs_bp",
    "si_ge_0p9_frac": "needs_bp",
    "si_trend_per_h": "needs_bp",
    "sbp_drop_max_pct": "needs_bp",
}


def _pos_median(series: pd.Series) -> float:
    val = series.median()
    return float(val) if pd.notna(val) else np.nan


def _drop_pct(roll: pd.Series, base: float) -> float:
    if not np.isfinite(base) or base <= 0 or roll.notna().sum() == 0:
        return np.nan
    return float((1.0 - roll.min() / base) * 100.0)


def _rise_pct(roll: pd.Series, base: float) -> float:
    if not np.isfinite(base) or base <= 0 or roll.notna().sum() == 0:
        return np.nan
    return float((roll.max() / base - 1.0) * 100.0)


def _trend_per_hour(t_s: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(t_s) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    slope = np.polyfit(t_s[ok], y[ok], 1)[0]  # units per second
    return float(slope * 3600.0)


def summarize_case_detail(windows: pd.DataFrame, cfg: Config, op_start=None, op_end=None):
    """Return (features dict, None) or (None, reason string)."""
    w = windows.sort_values("t_start").reset_index(drop=True)
    if w.empty:
        return None, "no_windows"

    if op_start is not None and np.isfinite(op_start):
        t0 = float(op_start)
        base = w[(w.t_start >= t0 - cfg.baseline_s) & (w.t_start < t0)]
        body = w[w.t_start >= t0]
        if op_end is not None and np.isfinite(op_end):
            body = body[body.t_start <= op_end]
    else:  # fallback if surgery times are missing: first baseline_s of recording
        t_first = float(w.t_start.min())
        t0 = t_first + cfg.baseline_s
        base = w[w.t_start < t0]
        body = w[w.t_start >= t0]

    if cfg.analysis_minutes:  # fixed analysis window so long surgeries are not favoured
        body = body[body.t_start < t0 + cfg.analysis_minutes * 60.0]

    n_base = int(base["ppg_ac"].notna().sum())
    n_body = int(body["ppg_ac"].notna().sum())
    if n_base < cfg.min_baseline_windows:
        return None, f"baseline_too_short (usable PPG windows {n_base}, need {cfg.min_baseline_windows})"
    if n_body < cfg.min_body_windows:
        return None, f"surgery_too_short (usable PPG windows {n_body}, need {cfg.min_body_windows})"

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
        "hr_rise_max_pct": _rise_pct(roll["hr_best"], b["hr_best"]),
        "ppg_hr_rise_max_pct": _rise_pct(roll["ppg_hr"], b["ppg_hr"]),
        "sbp_drop_max_pct": _drop_pct(roll["sbp"], b["sbp"]),
        "ac_trend_pct_per_h": _trend_per_hour(t, (body["ppg_ac"] / b["ppg_ac"] * 100.0).to_numpy())
        if b["ppg_ac"] > 0 else np.nan,
        "si_trend_per_h": _trend_per_hour(t, body["shock_index"].to_numpy(dtype=float)),
        "duration_min": float((t.max() - t.min() + cfg.window_s) / 60.0),
        "n_body_windows": int(len(body)),
        "sbp_coverage": float(body["sbp"].notna().mean()),
    }
    return out, None


def summarize_case(windows: pd.DataFrame, cfg: Config, op_start=None, op_end=None) -> dict | None:
    """Backward-compatible wrapper: the feature dict, or None if the case is unusable."""
    feats, _ = summarize_case_detail(windows, cfg, op_start, op_end)
    return feats


def summarize_all(cases: pd.DataFrame, windows_by_case: dict, cfg: Config, extras: dict | None = None):
    """Build the case-level table and an exclusions table.

    `cases` needs: caseid, label, opstart, opend (+ optional covariates).
    `extras` maps caseid -> dict of extra columns (blood-pressure info, QC summary).
    Returns (table DataFrame, exclusions DataFrame[caseid, label, reason]).
    """
    rows, excluded = [], []
    for _, row in cases.iterrows():
        cid = int(row["caseid"])
        w = windows_by_case.get(cid)
        if w is None:
            continue
        feats, reason = summarize_case_detail(w, cfg, row.get("opstart", np.nan), row.get("opend", np.nan))
        if feats is None:
            excluded.append({"caseid": cid, "label": int(row["label"]), "reason": reason})
            continue
        rec = row.to_dict()
        rec.update(feats)
        if extras and cid in extras:
            rec.update(extras[cid])
        rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(excluded, columns=["caseid", "label", "reason"])
