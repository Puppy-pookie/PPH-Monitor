"""Pipeline steps: select -> download -> summarize -> evaluate."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .case_features import summarize_all, summarize_case
from .cohort import build_manifest, candidate_cases, class_quota
from .config import Config
from .model import evaluate
from .plotting import plot_feature_boxplots, plot_trajectories
from .signal_features import build_case_windows
from .synthetic import make_synthetic_cohort
from .version import __production_date__, __version__
from .vitaldb_client import load_case_signals, load_clinical_table

log = logging.getLogger(__name__)

CANDIDATES_FILE = "candidates.csv"
STATUS_FILE = "download_status.csv"
CASE_FEATURES_FILE = "case_features.csv"


def _windows_path(cfg: Config, caseid: int) -> Path:
    return cfg.features_dir / f"windows_{int(caseid)}.csv.gz"


# ---------------------------------------------------------------- select
def step_select(cfg: Config, force: bool = False) -> pd.DataFrame:
    cfg.ensure_dirs()
    cases = load_clinical_table(cfg, force=force)
    cands = candidate_cases(cases, cfg)
    manifest = build_manifest(cands, cfg)
    manifest.to_csv(cfg.manifest_dir / CANDIDATES_FILE, index=False)
    log.info("Candidates: %d cases (label=1: %d, label=0: %d)", len(manifest),
             int((manifest.label == 1).sum()), int((manifest.label == 0).sum()))
    return manifest


# -------------------------------------------------------------- download
def step_download(cfg: Config, force: bool = False, loader=None) -> pd.DataFrame:
    """Download each candidate case, compute window features, keep only cases that
    yield a usable case-level summary, until the per-class quota is met."""
    loader = loader or load_case_signals
    cfg.ensure_dirs()
    manifest = pd.read_csv(cfg.manifest_dir / CANDIDATES_FILE)
    quota = class_quota(cfg)
    have = {0: 0, 1: 0}
    status = []

    # Remember cases that were already found unusable so reruns do not re-download them.
    known_bad: dict[int, str] = {}
    status_path = cfg.manifest_dir / STATUS_FILE
    if status_path.exists() and not force:
        prev = pd.read_csv(status_path)
        for _, r in prev.iterrows():
            if r["status"] in ("no_ppg_track", "too_little_usable_data"):
                known_bad[int(r["caseid"])] = r["status"]

    try:
        for label in (1, 0):
            sub = manifest[manifest["label"] == label]
            for _, row in sub.iterrows():
                if have[label] >= quota[label]:
                    break
                cid = int(row["caseid"])
                if cid in known_bad:
                    status.append({"caseid": cid, "label": label, "status": known_bad[cid]})
                    continue
                path = _windows_path(cfg, cid)
                src = None
                if path.exists() and not force:
                    windows = pd.read_csv(path)
                    state = "cached"
                else:
                    try:
                        sig = loader(cid, cfg)
                    except Exception as exc:  # network error, corrupt file, ...
                        log.warning("case %s failed: %s", cid, exc)
                        status.append({"caseid": cid, "label": label, "status": f"error: {exc}"})
                        continue
                    if sig is None:
                        status.append({"caseid": cid, "label": label, "status": "no_ppg_track"})
                        continue
                    windows, src = build_case_windows(
                        sig["ppg"], sig["ppg_fs"], sig["hr"], sig["sbp"],
                        sig["numeric_interval_s"], cfg)
                    state = "downloaded"
                feats = summarize_case(windows, cfg, row.get("opstart", np.nan), row.get("opend", np.nan))
                if feats is None:
                    status.append({"caseid": cid, "label": label, "status": "too_little_usable_data"})
                    continue
                if state == "downloaded":
                    windows.to_csv(path, index=False)
                have[label] += 1
                status.append({"caseid": cid, "label": label, "status": state, "sbp_source": src})
                log.info("label=%d: %d/%d  case %d (%s)", label, have[label], quota[label], cid, state)
    finally:
        out = pd.DataFrame(status)
        out.to_csv(cfg.manifest_dir / STATUS_FILE, index=False)
    for label in (0, 1):
        if have[label] < quota[label]:
            log.warning("Only %d/%d usable cases found for label=%d (try --max-cases larger or a different filter)",
                        have[label], quota[label], label)
    return out


# ------------------------------------------------------------- summarize
def load_windows(cfg: Config, caseids) -> dict:
    out = {}
    for cid in caseids:
        p = _windows_path(cfg, cid)
        if p.exists():
            out[int(cid)] = pd.read_csv(p)
    return out


def step_summarize(cfg: Config) -> tuple[pd.DataFrame, dict]:
    cfg.ensure_dirs()
    manifest = pd.read_csv(cfg.manifest_dir / CANDIDATES_FILE)
    windows = load_windows(cfg, manifest["caseid"])
    table = summarize_all(manifest, windows, cfg)
    # respect the per-class quota, in manifest order, for reproducibility
    quota = class_quota(cfg)
    keep = []
    for label, q in quota.items():
        keep.append(table[table["label"] == label].head(q))
    table = pd.concat(keep).reset_index(drop=True) if keep else table
    table.to_csv(cfg.manifest_dir / CASE_FEATURES_FILE, index=False)
    log.info("Case-level table: %d cases", len(table))
    return table, {int(c): windows[int(c)] for c in table["caseid"]}


# -------------------------------------------------------------- evaluate
def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records"))
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def step_evaluate(cfg: Config, table: pd.DataFrame, windows_by_case: dict,
                  mode: str = "vitaldb") -> Path:
    """Run the model, write results.json + csv + figures into a new run folder."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / f"{stamp}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    synthetic = mode == "synthetic"
    note = "[SYNTHETIC DATA - pipeline test only]" if synthetic else ""

    res = evaluate(table, cfg)
    table.to_csv(run_dir / "case_features.csv", index=False)
    uni = res.get("univariate", pd.DataFrame())
    if isinstance(uni, pd.DataFrame) and not uni.empty:
        uni.to_csv(run_dir / "univariate_auc.csv", index=False)
        plot_feature_boxplots(table, uni, run_dir / "fig_feature_boxplots.png", note)
    if len(table):
        plot_trajectories(table, windows_by_case, cfg, run_dir / "fig_trajectories.png", note)

    payload = {
        "pipeline_version": __version__,
        "production_date": __production_date__,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "data_note": "SYNTHETIC - not real patient data" if synthetic else
                     "VitalDB surgical cohort (not obstetric); label is a proxy for blood loss",
        "config": cfg.to_dict(),
        "results": {k: v for k, v in res.items() if k != "univariate"},
        "univariate_top": uni.head(8) if isinstance(uni, pd.DataFrame) else None,
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    log.info("Results written to %s", run_dir)
    return run_dir


# ------------------------------------------------------------------ demo
def run_demo(cfg: Config, n_cases: int = 24) -> Path:
    """End-to-end run on synthetic data. No network needed."""
    log.info("Generating %d synthetic cases (offline demo)...", n_cases)
    cases, windows = make_synthetic_cohort(cfg, n_cases=n_cases)
    table = summarize_all(cases, windows, cfg)
    log.info("Case-level table: %d cases", len(table))
    return step_evaluate(cfg, table, windows, mode="synthetic")


# ------------------------------------------------------------ full run
def run_all(cfg: Config, force: bool = False) -> Path:
    step_select(cfg, force=False)
    step_download(cfg, force=force)
    table, windows = step_summarize(cfg)
    return step_evaluate(cfg, table, windows, mode="vitaldb")
