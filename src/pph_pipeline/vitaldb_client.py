"""Thin wrapper around the official ``vitaldb`` package.

Two things are fetched:
  * the clinical table (one row per surgery: labels, drugs, surgery start/end times)
  * one case's signals (PPG waveform, heart rate, systolic BP)

The ``vitaldb`` package is imported lazily so the rest of the pipeline (and the
offline demo and tests) works without network access.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)


def load_clinical_table(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Download (and cache) VitalDB's clinical table of all 6,388 cases."""
    cfg.ensure_dirs()
    cache = cfg.raw_dir / "cases.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    url = f"{cfg.api_url}/cases"
    log.info("Downloading clinical table from %s", url)
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    df.to_csv(cache, index=False)
    return df


def _open_case(caseid: int, track_names: list[str]):
    """Open one case from the public VitalDB dataset (downloads the .vital file)."""
    import vitaldb  # lazy import

    return vitaldb.VitalFile(int(caseid), track_names)


def load_case_signals(caseid: int, cfg: Config, opener=None) -> dict | None:
    """Return the signals needed for one case, or None if the PPG track is missing.

    Result keys: ppg (array @ cfg.target_fs), ppg_fs, hr (array @ 1 interval),
    sbp (dict track -> array), numeric_interval_s.
    ``opener`` lets tests inject a fake VitalFile.
    """
    opener = opener or _open_case
    tracks = [cfg.ppg_track, cfg.hr_track, *cfg.sbp_tracks]
    vf = opener(caseid, tracks)

    available = set(vf.get_track_names())
    if cfg.ppg_track not in available:
        log.info("case %s: no PPG track (%s)", caseid, cfg.ppg_track)
        return None

    ppg = vf.to_numpy([cfg.ppg_track], 1.0 / cfg.target_fs)[:, 0]
    if ppg.size == 0 or np.isnan(ppg).mean() > 0.9:
        return None

    interval = cfg.numeric_interval_s
    num_tracks = [cfg.hr_track, *cfg.sbp_tracks]
    num = vf.to_numpy(num_tracks, interval)
    arrays = {name: num[:, i] for i, name in enumerate(num_tracks)}

    return {
        "ppg": ppg.astype(float),
        "ppg_fs": cfg.target_fs,
        "hr": arrays[cfg.hr_track],
        "sbp": {name: arrays[name] for name in cfg.sbp_tracks},
        "numeric_interval_s": interval,
    }
