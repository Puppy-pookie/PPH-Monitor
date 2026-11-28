"""All tunable settings live here so every run is reproducible and loggable."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

# Bump when the window-level feature computation changes, so stale caches are rebuilt.
FEATURES_VERSION = 3


@dataclass
class Config:
    # --- VitalDB source -------------------------------------------------
    api_url: str = "https://api.vitaldb.net"

    # --- folders (relative to where you run the command) ---------------
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    # --- VitalDB track names -------------------------------------------
    ppg_track: str = "SNUADC/PLETH"
    hr_track: str = "Solar8000/HR"
    # Invasive (arterial line) systolic pressure. FEM_* is the femoral-line name.
    arterial_sbp_tracks: tuple = ("Solar8000/ART_SBP", "Solar8000/FEM_SBP")
    # Non-invasive (cuff) systolic pressure: one reading every few minutes.
    cuff_sbp_tracks: tuple = ("Solar8000/NIBP_SBP",)

    # --- cohort selection ----------------------------------------------
    max_cases: int = 20            # total usable cases wanted (split between classes)
    department: str | None = None  # e.g. "Gyn"; None = all departments
    operation: str | None = None   # e.g. "Cesarean": text matched in the operation name
    label_mode: str = "ebl"        # "ebl" (estimated blood loss) or "any_rbc"
    ebl_threshold_ml: float = 500.0
    require_aline: bool = False    # keep only cases with an arterial line
    seed: int = 42

    # --- blood pressure handling ---------------------------------------
    # "arterial" = invasive only, "cuff" = non-invasive only, "either" = arterial if
    # well covered, otherwise cuff. The choice is always recorded per case.
    bp_source: str = "arterial"
    cuff_carry_s: float = 600.0    # a cuff reading is carried forward at most this long
    min_bp_coverage: float = 0.3   # minimum share of windows with a valid reading

    # --- windowing and PPG quality control ------------------------------
    window_s: float = 30.0
    target_fs: float = 100.0       # PPG is resampled to this rate on load
    numeric_interval_s: float = 1.0
    min_beats: int = 15
    min_sqi: float = 0.5
    max_nan_frac: float = 0.2
    max_clip_frac: float = 0.02    # share of samples sitting on flat clipped runs
    clip_run: int = 5              # a flat run this long (samples) counts as clipping
    min_beat_conf: float = 0.8     # below this, a window is flagged as motion/artifact
    max_drift_ratio: float = 3.0   # slow baseline swing relative to pulse amplitude
    strict_qc: bool = False        # True: also reject (not just flag) motion and drift windows

    # --- case-level summary --------------------------------------------
    baseline_s: float = 600.0      # baseline = 10 min before surgery start
    rolling_windows: int = 5       # median smoothing (5 x 30 s = 2.5 min)
    min_baseline_windows: int = 5
    min_body_windows: int = 20
    analysis_minutes: float | None = None  # None = whole surgery; e.g. 60 = first hour only

    # --- evaluation -----------------------------------------------------
    cv_folds: int = 5
    cv_repeats: int = 20           # repeated stratified cross-validation
    perm_repeats: int = 5          # CV repeats used inside each permutation test
    n_permutations: int = 200
    n_bootstrap: int = 1000

    # --- helpers --------------------------------------------------------
    @property
    def sbp_tracks(self) -> tuple:
        return tuple(self.arterial_sbp_tracks) + tuple(self.cuff_sbp_tracks)

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_dir) / "raw"

    @property
    def manifest_dir(self) -> Path:
        return Path(self.data_dir) / "manifests"

    @property
    def features_dir(self) -> Path:
        return Path(self.data_dir) / "features"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.manifest_dir, self.features_dir, Path(self.output_dir)):
            d.mkdir(parents=True, exist_ok=True)

    def feature_fingerprint(self) -> dict:
        """Settings that change the cached window features. If any differ, the cache is rebuilt."""
        return {
            "features_version": FEATURES_VERSION,
            "window_s": self.window_s, "target_fs": self.target_fs,
            "numeric_interval_s": self.numeric_interval_s,
            "min_beats": self.min_beats, "min_sqi": self.min_sqi, "max_nan_frac": self.max_nan_frac,
            "max_clip_frac": self.max_clip_frac, "clip_run": self.clip_run,
            "min_beat_conf": self.min_beat_conf, "max_drift_ratio": self.max_drift_ratio,
            "strict_qc": self.strict_qc,
            "ppg_track": self.ppg_track, "hr_track": self.hr_track,
            "arterial_sbp_tracks": list(self.arterial_sbp_tracks),
            "cuff_sbp_tracks": list(self.cuff_sbp_tracks),
            "cuff_carry_s": self.cuff_carry_s,
        }

    def to_dict(self) -> dict:
        out = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            out[key] = value
        return out
