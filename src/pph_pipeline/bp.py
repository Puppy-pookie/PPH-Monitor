"""Blood-pressure source handling.

The end goal is a NON-INVASIVE device, so every case records exactly which pressure
source was used and how complete it was:

  arterial : invasive arterial line, continuous. Great for research, unavailable to a device.
  cuff     : non-invasive cuff, one reading every few minutes (carried forward for a limited time).
  none     : no usable pressure; shock index is then missing, never guessed.

Both columns are stored in the cached window data; the mode (cfg.bp_source) picks one
at analysis time. "either" uses arterial when it is well covered, otherwise cuff, and
the type actually used is recorded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def cuff_forward_fill(v, interval_s: float, carry_s: float) -> np.ndarray:
    """Carry each cuff reading forward for at most carry_s seconds."""
    limit = max(1, int(round(carry_s / interval_s)))
    return pd.Series(np.asarray(v, dtype=float)).ffill(limit=limit).to_numpy()


def _coverage(s: pd.Series) -> float:
    return float(s.notna().mean()) if len(s) else 0.0


def _pick(candidates: dict, min_cov: float):
    """First candidate with enough coverage; otherwise the best one; otherwise none."""
    best, best_cov = None, 0.0
    for name, s in candidates.items():
        cov = _coverage(s)
        if cov >= min_cov:
            return name, s, cov
        if cov > best_cov:
            best, best_cov = name, cov
    if best is None:
        return None, pd.Series(dtype=float), 0.0
    return best, candidates[best], best_cov


def build_bp_columns(sbp_by_track: dict, interval_s: float, cfg: Config):
    """Return (sbp_art Series, sbp_cuff Series, meta dict) on the 30-second window grid."""
    from .signal_features import numeric_windows  # local import avoids a circular import

    art = {t: numeric_windows(v, interval_s, cfg, 40, 260)
           for t, v in sbp_by_track.items() if t in cfg.arterial_sbp_tracks}
    cuff, raw_counts = {}, {}
    for t, v in sbp_by_track.items():
        if t in cfg.cuff_sbp_tracks:
            raw = np.asarray(v, dtype=float)
            raw_counts[t] = int(((raw >= 40) & (raw <= 260)).sum())
            cuff[t] = numeric_windows(cuff_forward_fill(raw, interval_s, cfg.cuff_carry_s),
                                      interval_s, cfg, 40, 260)

    a_name, a_s, a_cov = _pick(art, cfg.min_bp_coverage)
    c_name, c_s, c_cov = _pick(cuff, 1e-9)
    meta = {
        "arterial_source": a_name,
        "arterial_coverage": a_cov,
        "arterial_low_coverage": bool(a_name is not None and a_cov < cfg.min_bp_coverage),
        "cuff_source": c_name,
        "cuff_coverage": c_cov,
        "cuff_raw_readings": raw_counts.get(c_name, 0) if c_name else 0,
        "cuff_carry_s": cfg.cuff_carry_s,
    }
    return a_s.rename("sbp_art"), c_s.rename("sbp_cuff"), meta


def apply_bp_mode(windows: pd.DataFrame, bp_meta: dict, cfg: Config):
    """Choose the pressure source for this analysis and compute shock index.

    Returns (windows with sbp, hr_best, shock_index columns, info dict).
    """
    w = windows.copy()
    mode = cfg.bp_source
    if mode not in ("arterial", "cuff", "either"):
        raise ValueError(f"bp_source must be 'arterial', 'cuff' or 'either', not {mode!r}")

    art_ok = bp_meta.get("arterial_source") is not None and "sbp_art" in w
    art_good = art_ok and bp_meta.get("arterial_coverage", 0.0) >= cfg.min_bp_coverage
    cuff_ok = bp_meta.get("cuff_source") is not None and "sbp_cuff" in w

    if mode == "arterial":
        use = "arterial" if art_ok else None
    elif mode == "cuff":
        use = "cuff" if cuff_ok else None
    else:
        use = "arterial" if art_good else ("cuff" if cuff_ok else None)

    if use == "arterial":
        w["sbp"] = w["sbp_art"]
        source, cov = bp_meta["arterial_source"], bp_meta["arterial_coverage"]
    elif use == "cuff":
        w["sbp"] = w["sbp_cuff"]
        source, cov = bp_meta["cuff_source"], bp_meta["cuff_coverage"]
    else:
        w["sbp"] = np.nan
        source, cov = None, 0.0

    w["hr_best"] = w["hr"].fillna(w["ppg_hr"])
    w["shock_index"] = w["hr_best"] / w["sbp"]
    info = {
        "bp_mode": mode,
        "bp_type": use or "none",
        "bp_source": source,
        "bp_coverage": float(cov),
        "bp_valid_windows": int(w["sbp"].notna().sum()),
        "bp_missing_windows": int(w["sbp"].isna().sum()),
        "bp_invasive": use == "arterial",
        "bp_low_coverage": bool(use == "arterial" and bp_meta.get("arterial_low_coverage", False)),
        "bp_cuff_raw_readings": int(bp_meta.get("cuff_raw_readings", 0)) if use == "cuff" else 0,
        "bp_carry_forward_s": float(cfg.cuff_carry_s) if use == "cuff" else 0.0,
    }
    return w, info
