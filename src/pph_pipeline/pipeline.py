"""Pipeline steps: select -> download -> summarize -> evaluate."""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .bp import apply_bp_mode
from .case_features import summarize_all, summarize_case_detail
from .cohort import build_manifest, candidate_cases, class_quota
from .config import Config
from .model import MODEL_SETS, evaluate
from .plotting import plot_feature_boxplots, plot_trajectories
from .signal_features import CaseWindows, build_case_windows
from .synthetic import make_synthetic_cohort
from .version import __production_date__, __version__
from .vitaldb_client import load_case_signals, load_clinical_table

log = logging.getLogger(__name__)

CANDIDATES_FILE = "candidates.csv"
STATUS_FILE = "download_status.csv"
CASE_FEATURES_FILE = "case_features.csv"
SUMMARY_EXCLUSIONS_FILE = "summarize_exclusions.csv"


# ------------------------------------------------------------ case cache
def _windows_path(cfg: Config, caseid: int) -> Path:
    return cfg.features_dir / f"windows_{int(caseid)}.csv.gz"


def _meta_path(cfg: Config, caseid: int) -> Path:
    return cfg.features_dir / f"case_{int(caseid)}_meta.json"


def _rejects_path(cfg: Config, caseid: int) -> Path:
    return cfg.features_dir / f"rejects_{int(caseid)}.csv.gz"


def _save_case(cfg: Config, caseid: int, built: CaseWindows) -> None:
    built.windows.to_csv(_windows_path(cfg, caseid), index=False)
    built.rejects.to_csv(_rejects_path(cfg, caseid), index=False)
    meta = {
        "pipeline_version": __version__,
        "fingerprint": cfg.feature_fingerprint(),
        "bp_meta": built.bp_meta,
        "qc": built.qc,
    }
    with open(_meta_path(cfg, caseid), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def _load_case(cfg: Config, caseid: int) -> CaseWindows | None:
    """Load a cached case. Returns None if it is missing or was built with an older
    version or different feature settings (so it must be recomputed)."""
    wp, mp = _windows_path(cfg, caseid), _meta_path(cfg, caseid)
    if not (wp.exists() and mp.exists()):
        return None
    with open(mp, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("fingerprint") != json.loads(json.dumps(cfg.feature_fingerprint())):
        return None
    rp = _rejects_path(cfg, caseid)
    rejects = pd.read_csv(rp) if rp.exists() else pd.DataFrame(columns=["t_start", "reason"])
    return CaseWindows(pd.read_csv(wp), meta["bp_meta"], meta["qc"], rejects)


# --------------------------------------------------------------- explore
def explore_report(cases: pd.DataFrame, cfg: Config) -> str:
    """Plain-text summary of what is in the clinical table, so options can be chosen with numbers."""
    df = cases.copy()
    df.columns = [c.lower() for c in df.columns]
    n = len(df)
    ebl = pd.to_numeric(df["intraop_ebl"], errors="coerce") if "intraop_ebl" in df else pd.Series(dtype=float)
    rbc = pd.to_numeric(df["intraop_rbc"], errors="coerce") if "intraop_rbc" in df else pd.Series(dtype=float)
    has_aline = df["aline1"].notna() if "aline1" in df else pd.Series(False, index=df.index)

    def _breakdown(col: str, top: int | None = None) -> str:
        key = df[col].astype(str)
        tab = pd.DataFrame({
            "cases": key.value_counts(),
            f"EBL>={int(cfg.ebl_threshold_ml)}": (ebl >= cfg.ebl_threshold_ml).groupby(key).sum(),
            "any_RBC": (rbc > 0).groupby(key).sum(),
            "aline": has_aline.groupby(key).sum(),
        }).fillna(0).astype(int).sort_values("cases", ascending=False)
        return (tab.head(top) if top else tab).to_string()

    out = [f"Clinical table: {n} cases", ""]
    out.append("1) Label definitions (what counts as a 'higher blood loss' case)")
    out.append(f"   Estimated blood loss recorded: {int(ebl.notna().sum())} cases")
    for thr in (300, 500, 1000):
        k = int((ebl >= thr).sum())
        mark = "  <- current --ebl-threshold" if abs(thr - cfg.ebl_threshold_ml) < 1e-9 else ""
        out.append(f"   EBL >= {thr:>4} mL: {k:>5} cases{mark}")
    out.append(f"   Any red-cell transfusion (--label-mode any_rbc): {int((rbc > 0).sum())} of {int(rbc.notna().sum())} recorded")
    out.append(f"   Arterial line present (--require-aline): {int(has_aline.sum())} cases")
    out.append("")
    out.append("2) Departments (use any part of a name with --department, not case sensitive)")
    out.append(_breakdown("department") if "department" in df else "   (no department column found)")
    out.append("")
    out.append("3) Most common operations (use any part of a name with --operation)")
    if "opname" in df:
        out.append(_breakdown("opname", top=15))
        if "subjectid" in df:
            multi = int((df["subjectid"].value_counts() > 1).sum())
            out.append(f"   Patients with more than one surgery: {multi} (relevant for grouped cross-validation later)")
    else:
        out.append("   (no opname column found)")
    out.append("")
    out.append("4) What your current settings would give")
    cands = candidate_cases(df, cfg)
    out.append(f"   label_mode={cfg.label_mode}, department={cfg.department!r}, operation={cfg.operation!r}, "
               f"require_aline={cfg.require_aline}")
    out.append(f"   usable candidates: label=1 -> {int((cands.label == 1).sum())}, label=0 -> {int((cands.label == 0).sum())}")
    quota = class_quota(cfg)
    out.append(f"   --max-cases {cfg.max_cases} asks for {quota[1]} higher-loss and {quota[0]} lower-loss cases")
    out.append("")
    out.append("Rule of thumb: aim for at least 30 cases per group (more is better). With fewer, the confidence intervals are very wide.")
    out.append("Example: python -m pph_pipeline run --max-cases 40 --operation Cesarean --ebl-threshold 500")
    return "\n".join(out)


def step_explore(cfg: Config) -> str:
    cfg.ensure_dirs()
    cases = load_clinical_table(cfg)
    report = explore_report(cases, cfg)
    print(report)
    return report


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
    """Download each candidate case, compute window features, keep only cases that yield a
    usable case-level summary, until the per-class quota is met.

    Window features do not depend on the blood-pressure mode or analysis window, so
    changing those options never needs a re-download. Changing a feature or quality
    setting (or upgrading the pipeline) rebuilds the cache automatically.
    """
    loader = loader or load_case_signals
    cfg.ensure_dirs()
    manifest = pd.read_csv(cfg.manifest_dir / CANDIDATES_FILE)
    quota = class_quota(cfg)
    have = {0: 0, 1: 0}
    status = []

    # Remember cases that have no PPG track at all, so reruns do not re-download them.
    known_bad: set[int] = set()
    status_path = cfg.manifest_dir / STATUS_FILE
    if status_path.exists() and not force:
        prev = pd.read_csv(status_path)
        known_bad = {int(r.caseid) for r in prev.itertuples() if r.status == "no_ppg_track"}

    try:
        for label in (1, 0):
            sub = manifest[manifest["label"] == label]
            for _, row in sub.iterrows():
                if have[label] >= quota[label]:
                    break
                cid = int(row["caseid"])
                base = {"caseid": cid, "label": label}
                if cid in known_bad:
                    status.append({**base, "status": "no_ppg_track", "detail": "remembered from an earlier run"})
                    continue

                case = None if force else _load_case(cfg, cid)
                if case is not None:
                    state = "cached"
                else:
                    if not force and _windows_path(cfg, cid).exists():
                        log.info("case %s: rebuilding cache (older version or different feature settings)", cid)
                    try:
                        sig = loader(cid, cfg)
                    except Exception as exc:  # network error, corrupt file, ...
                        log.warning("case %s failed: %s", cid, exc)
                        status.append({**base, "status": "error", "detail": str(exc)})
                        continue
                    if sig is None:
                        status.append({**base, "status": "no_ppg_track", "detail": ""})
                        continue
                    case = build_case_windows(sig["ppg"], sig["ppg_fs"], sig["hr"], sig["sbp"],
                                              sig["numeric_interval_s"], cfg)
                    _save_case(cfg, cid, case)
                    state = "downloaded"

                w2, bp_info = apply_bp_mode(case.windows, case.bp_meta, cfg)
                feats, reason = summarize_case_detail(w2, cfg, row.get("opstart", np.nan), row.get("opend", np.nan))
                rec = {**base, **bp_info, **case.qc}
                if feats is None:
                    status.append({**rec, "status": "too_little_usable_data", "detail": reason})
                    continue
                have[label] += 1
                status.append({**rec, "status": state, "detail": ""})
                log.info("label=%d: %d/%d  case %d (%s, BP: %s)", label, have[label], quota[label], cid,
                         state, bp_info["bp_type"])
    finally:
        out = pd.DataFrame(status)
        out.to_csv(cfg.manifest_dir / STATUS_FILE, index=False)
    for label in (0, 1):
        if have[label] < quota[label]:
            log.warning("Only %d/%d usable cases found for label=%d (try a larger --max-cases or a different filter)",
                        have[label], quota[label], label)
    return out


# ------------------------------------------------------------- summarize
def step_summarize(cfg: Config):
    """Returns (case table, prepared windows by case, exclusions DataFrame)."""
    cfg.ensure_dirs()
    manifest = pd.read_csv(cfg.manifest_dir / CANDIDATES_FILE)
    prepared, extras = {}, {}
    n_stale = 0
    for cid in manifest["caseid"]:
        case = _load_case(cfg, int(cid))
        if case is None:
            n_stale += int(_windows_path(cfg, int(cid)).exists())
            continue
        w2, bp_info = apply_bp_mode(case.windows, case.bp_meta, cfg)
        prepared[int(cid)] = w2
        extras[int(cid)] = {**bp_info, **case.qc}
    if n_stale:
        log.warning("%d cached case(s) were built by an older version or other settings and were skipped. "
                    "Run 'download' (or 'run') to rebuild them.", n_stale)

    table, excl = summarize_all(manifest, prepared, cfg, extras)
    if table.empty:
        raise RuntimeError("No usable cases found. Run 'download' first, or check the filters with 'explore'.")
    quota = class_quota(cfg)  # respect the per-class quota, in manifest order, for reproducibility
    table = pd.concat([table[table["label"] == lab].head(q) for lab, q in quota.items()]).reset_index(drop=True)
    table.to_csv(cfg.manifest_dir / CASE_FEATURES_FILE, index=False)
    excl.to_csv(cfg.manifest_dir / SUMMARY_EXCLUSIONS_FILE, index=False)
    log.info("Case-level table: %d cases (%d excluded at summary stage)", len(table), len(excl))
    return table, {int(c): prepared[int(c)] for c in table["caseid"]}, excl


# -------------------------------------------------------------- evaluate
def _jsonable(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
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


def environment_info(cfg: Config) -> dict:
    pkgs = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "vitaldb"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = "not installed"
    return {
        "pipeline_version": __version__,
        "production_date": __production_date__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
        "random_seed": cfg.seed,
        "command_line": " ".join(sys.argv),
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def build_report(payload: dict, table: pd.DataFrame, excl: pd.DataFrame, res: dict) -> str:
    """A short human-readable run summary (the full numbers are in results.json)."""
    L = []
    L.append("# Run summary")
    L.append("")
    L.append(f"Pipeline v{payload['pipeline_version']} (produced {payload['production_date']}). "
             f"Mode: {payload['mode']}. Run: {payload['run_timestamp']}.")
    L.append("")
    L.append("> Research use only. Not a medical device. " + payload["data_note"] + ".")
    L.append("")
    L.append("## Question")
    L.append("")
    L.append("Do physiological signals that a non-invasive device could record change more in cases with higher "
             "blood loss? The end goal is a non-invasive PPH monitor, so results are labelled by what a device "
             "could actually measure.")
    L.append("")
    L.append("## Dataset quality")
    L.append("")
    L.append(f"- Cases used: {res['n_cases']} (higher loss: {res['n_pos']}, lower loss: {res['n_neg']})")
    if len(excl):
        L.append(f"- Cases excluded at summary stage: {len(excl)}. Reasons: "
                 + "; ".join(f"{k.split(' (')[0]} x{v}" for k, v in excl["reason"].value_counts().items()))
    else:
        L.append("- Cases excluded at summary stage: 0")
    if "bp_type" in table:
        counts = table["bp_type"].value_counts().to_dict()
        L.append(f"- Blood-pressure source (mode: {payload['config']['bp_source']}): "
                 + ", ".join(f"{k} {v}" for k, v in counts.items()))
        if "bp_coverage" in table:
            L.append(f"- Median blood-pressure coverage of the record: {table['bp_coverage'].median():.0%}")
    if "qc_accept_frac" in table:
        L.append(f"- PPG windows accepted: median {table['qc_accept_frac'].median():.0%} per case "
                 f"(range {table['qc_accept_frac'].min():.0%} to {table['qc_accept_frac'].max():.0%})")
        tot = int(table["qc_windows_total"].sum())
        for col, label in (("qc_reject_missing", "missing data"), ("qc_reject_flat", "flat signal"),
                           ("qc_reject_clipped", "clipped"), ("qc_reject_few_beats", "too few beats"),
                           ("qc_reject_irregular", "irregular beats"), ("qc_reject_low_sqi", "low quality score"),
                           ("qc_reject_motion", "motion (strict QC)"), ("qc_reject_drift", "drift (strict QC)")):
            if col in table and int(table[col].sum()):
                L.append(f"  - rejected for {label}: {int(table[col].sum())} of {tot} windows")
        if "qc_flag_motion" in table:
            L.append(f"  - kept but flagged: motion {int(table['qc_flag_motion'].sum())}, "
                     f"drift {int(table['qc_flag_drift'].sum())} windows "
                     f"(median beat confidence {table['qc_beat_conf_median'].median():.2f})")
    L.append("")
    L.append("## Model performance (cross-validated AUC, 95% confidence interval)")
    L.append("")
    if res["status"] != "ok":
        L.append(res.get("message", "Not enough data."))
    else:
        L.append("| Model | Features | AUC | 95% CI | Permutation p |")
        L.append("|---|---|---|---|---|")
        for name, m in res["models"].items():
            if m.get("status") != "ok":
                L.append(f"| {name} | | not computed ({m.get('status')}) | | |")
                continue
            L.append(f"| {name} | {', '.join(m['features'])} | {m['auc_mean']:.2f} | "
                     f"{m['auc_ci_low']:.2f} to {m['auc_ci_high']:.2f} | {m['p_permutation']:.3f} |")
        L.append("")
        L.append("## Single features (top 8)")
        L.append("")
        L.append("| Feature | AUC | 95% CI | Device needs |")
        L.append("|---|---|---|---|")
        for r in res["univariate"].head(8).itertuples():
            L.append(f"| {r.feature} | {r.auc:.2f} | {r.auc_ci_low:.2f} to {r.auc_ci_high:.2f} | {r.device_relevance} |")
        L.append("")
        L.append("Device needs: ppg_only = optical pulse sensor alone; needs_bp = also needs blood pressure "
                 "(non-invasive only if from a cuff); monitor_hr = hospital ECG heart rate.")
    L.append("")
    L.append("## How to read this")
    L.append("")
    for n in res.get("notes", []):
        L.append(f"- {n}")
    L.append("- With few cases, treat every number as a test of the method, not evidence about PPH.")
    L.append("- Blood loss here is surgical estimated blood loss, not a postpartum haemorrhage diagnosis, "
             "and there is no bleeding timing, so this is an association study, not an early-warning test.")
    L.append("")
    L.append("## Files in this folder")
    L.append("")
    L.append("results.json (all numbers), case_features.csv, univariate_auc.csv, exclusions.csv, "
             "qc_summary.csv, environment.json, and the figures.")
    return "\n".join(L) + "\n"


def step_evaluate(cfg: Config, table: pd.DataFrame, windows_by_case: dict, mode: str = "vitaldb",
                  exclusions: pd.DataFrame | None = None) -> Path:
    """Run the models, write results + report + figures into a new run folder."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / f"{stamp}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    synthetic = mode == "synthetic"
    note = "[SYNTHETIC DATA - pipeline test only]" if synthetic else ""
    exclusions = exclusions if exclusions is not None else pd.DataFrame(columns=["caseid", "label", "reason"])

    log.info("Evaluating models (%d CV repeats, %d permutations, %d bootstrap resamples)...",
             cfg.cv_repeats, cfg.n_permutations, cfg.n_bootstrap)
    res = evaluate(table, cfg)

    table.to_csv(run_dir / "case_features.csv", index=False)
    exclusions.to_csv(run_dir / "exclusions.csv", index=False)
    qc_cols = ["caseid", "label"] + [c for c in table.columns if c.startswith("qc_") or c.startswith("bp_")]
    table[qc_cols].to_csv(run_dir / "qc_summary.csv", index=False)
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
                     "VitalDB surgical cohort (includes some cesarean sections, but not postpartum records); "
                     "the label is a proxy for blood loss",
        "config": cfg.to_dict(),
        "model_feature_sets": MODEL_SETS,
        "results": {k: v for k, v in res.items() if k != "univariate"},
        "univariate_top": uni.head(10) if isinstance(uni, pd.DataFrame) else None,
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    with open(run_dir / "environment.json", "w", encoding="utf-8") as fh:
        json.dump(_jsonable(environment_info(cfg)), fh, indent=2)
    with open(run_dir / "report.md", "w", encoding="utf-8") as fh:
        fh.write(build_report(payload, table, exclusions, res))
    log.info("Results written to %s", run_dir)
    return run_dir


# ------------------------------------------------------------------ demo
def run_demo(cfg: Config, n_cases: int = 24) -> Path:
    """End-to-end run on synthetic data. No network needed."""
    log.info("Generating %d synthetic cases (offline demo)...", n_cases)
    cases, built = make_synthetic_cohort(cfg, n_cases=n_cases)
    prepared, extras = {}, {}
    for cid, b in built.items():
        w2, bp_info = apply_bp_mode(b.windows, b.bp_meta, cfg)
        prepared[cid] = w2
        extras[cid] = {**bp_info, **b.qc}
    table, excl = summarize_all(cases, prepared, cfg, extras)
    log.info("Case-level table: %d cases", len(table))
    return step_evaluate(cfg, table, prepared, mode="synthetic", exclusions=excl)


# ------------------------------------------------------------ full run
def run_all(cfg: Config, force: bool = False) -> Path:
    step_select(cfg, force=False)
    step_download(cfg, force=force)
    table, windows, excl = step_summarize(cfg)
    return step_evaluate(cfg, table, windows, mode="vitaldb", exclusions=excl)
