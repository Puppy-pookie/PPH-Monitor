"""All tunable settings live here so every run is reproducible and loggable."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


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
    # Priority order. FEM_* is what VitalDB renames ART_* to for femoral lines.
    sbp_tracks: tuple = ("Solar8000/ART_SBP", "Solar8000/FEM_SBP", "Solar8000/NIBP_SBP")

    # --- cohort selection ----------------------------------------------
    max_cases: int = 20            # total valid cases wanted (split between classes)
    department: str | None = None  # e.g. "Gynecology"; None = all departments
    label_mode: str = "ebl"        # "ebl" (estimated blood loss) or "any_rbc"
    ebl_threshold_ml: float = 500.0
    require_aline: bool = False    # keep only cases with an arterial line
    seed: int = 42

    # --- windowing ------------------------------------------------------
    window_s: float = 30.0
    target_fs: float = 100.0       # PPG is resampled to this rate on load
    numeric_interval_s: float = 1.0
    min_beats: int = 15
    min_sqi: float = 0.5
    max_nan_frac: float = 0.2

    # --- case-level summary --------------------------------------------
    baseline_s: float = 600.0      # baseline = 10 min before surgery start
    rolling_windows: int = 5       # median smoothing (5 x 30 s = 2.5 min)
    min_baseline_windows: int = 5
    min_body_windows: int = 20

    # --- evaluation -----------------------------------------------------
    cv_folds: int = 5

    # --- helpers --------------------------------------------------------
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

    def to_dict(self) -> dict:
        out = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            out[key] = value
        return out
