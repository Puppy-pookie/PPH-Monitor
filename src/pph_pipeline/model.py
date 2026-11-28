"""Case-level evaluation: single-feature AUCs and cross-validated logistic regression.

What changed in v0.3: one 5-fold split on ~40 cases is unreliable (a covariates-only
AUC of 1.0 turned out to be luck). Every model is now evaluated with
  * repeated stratified cross-validation (default 20 repeats),
  * a bootstrap 95% confidence interval over cases, and
  * a permutation test (shuffle the labels, repeat the whole procedure).

Models are small, pre-specified feature sets (not "everything"), so 40 cases are not
asked to support 16 coefficients:
  covariates_only : what surgery size/type alone explains
  ppg_only        : features a PPG-only device could measure (the core device question)
  physiology_core : PPG features plus the two blood-pressure features
  combined_core   : physiology_core plus covariates
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .case_features import FEATURE_RELEVANCE, SUMMARY_FEATURES
from .config import Config

COVARIATES = ["age", "asa", "emop", "sex_male", "intraop_phe", "intraop_eph", "intraop_epi", "duration_min"]
COVARIATES_CORE = ["age", "asa", "emop", "duration_min", "intraop_phe"]
PPG_ONLY = ["ac_drop_max_pct", "pi_drop_max_pct", "ac_trend_pct_per_h", "ppg_hr_rise_max_pct"]
BP_BASED = ["si_rise_max", "sbp_drop_max_pct"]
PHYSIO_CORE = PPG_ONLY + BP_BASED

MODEL_SETS = {
    "covariates_only": COVARIATES_CORE,
    "ppg_only": PPG_ONLY,
    "physiology_core": PHYSIO_CORE,
    "combined_core": PHYSIO_CORE + COVARIATES_CORE,
}


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sex" in df.columns:
        df["sex_male"] = df["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0})
    return df


def _usable(df: pd.DataFrame, cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() >= 3 and x.nunique() > 1:
                out.append(c)
    return out


def _make_pipe() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def _oof_scores(X: np.ndarray, y: np.ndarray, k: int, repeats: int, seed: int) -> np.ndarray:
    """Out-of-fold decision scores, shape (repeats, n_cases)."""
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed)
    scores = np.zeros((repeats, len(y)))
    for i, (tr, te) in enumerate(cv.split(X, y)):
        pipe = _make_pipe().fit(X[tr], y[tr])
        scores[i // k, te] = pipe.decision_function(X[te])
    return scores


def _mean_auc(scores: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(y, s) for s in scores]))


def _bootstrap_ci(score: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def cv_evaluate(df: pd.DataFrame, features: list[str], cfg: Config) -> dict:
    cols = _usable(df, features)
    if not cols:
        return {"status": "no_features"}
    y = df["label"].astype(int).to_numpy()
    k = min(cfg.cv_folds, int(np.bincount(y).min()))
    if k < 2:
        return {"status": "too_few_cases_per_class"}
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    rng = np.random.default_rng(cfg.seed)

    scores = _oof_scores(X, y, k, cfg.cv_repeats, cfg.seed)
    per_repeat = np.array([roc_auc_score(y, s) for s in scores])
    lo, hi = _bootstrap_ci(scores.mean(axis=0), y, cfg.n_bootstrap, rng)

    # Permutation test: same procedure on shuffled labels
    obs = _mean_auc(_oof_scores(X, y, k, cfg.perm_repeats, cfg.seed + 1), y)
    null = np.empty(cfg.n_permutations)
    for i in range(cfg.n_permutations):
        yp = rng.permutation(y)
        null[i] = _mean_auc(_oof_scores(X, yp, k, cfg.perm_repeats, cfg.seed + 2 + i), yp)
    p_perm = float((1 + np.sum(null >= obs)) / (1 + cfg.n_permutations))

    return {
        "status": "ok",
        "features": cols,
        "folds": k,
        "repeats": cfg.cv_repeats,
        "auc_mean": float(per_repeat.mean()),
        "auc_sd_across_repeats": float(per_repeat.std()),
        "auc_ci_low": lo,
        "auc_ci_high": hi,
        "p_permutation": p_perm,
    }


def univariate_auc(df: pd.DataFrame, features: list[str], cfg: Config | None = None) -> pd.DataFrame:
    rows = []
    y = df["label"].astype(int)
    rng = np.random.default_rng((cfg.seed if cfg else 42) + 99)
    n_boot = max(100, (cfg.n_bootstrap if cfg else 500) // 2)
    for f in _usable(df, features):
        x = pd.to_numeric(df[f], errors="coerce")
        ok = x.notna()
        if y[ok].nunique() < 2:
            continue
        yy, xx = y[ok].to_numpy(), x[ok].to_numpy()
        auc = float(roc_auc_score(yy, xx))
        lo, hi = _bootstrap_ci(xx, yy, n_boot, rng) if min(np.bincount(yy)) >= 2 else (np.nan, np.nan)
        rows.append({
            "feature": f, "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi,
            "auc_direction_free": max(auc, 1 - auc), "n": int(ok.sum()),
            "device_relevance": FEATURE_RELEVANCE.get(f, "covariate"),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("auc_direction_free", ascending=False).reset_index(drop=True) if len(out) else out


def _notes(res: dict, df: pd.DataFrame) -> list[str]:
    notes = []
    models = res.get("models", {})
    ok = {k: v for k, v in models.items() if v.get("status") == "ok"}
    n_pos, n_neg = res["n_pos"], res["n_neg"]
    if min(n_pos, n_neg) < 30:
        notes.append("Fewer than 30 cases in a group: confidence intervals are very wide. Treat this as a test of the method.")
    cov, phys = ok.get("covariates_only"), ok.get("physiology_core")
    if cov and phys and cov["auc_mean"] >= phys["auc_mean"]:
        notes.append("Surgery-size covariates explain at least as much as physiology. There is no evidence yet that physiology adds information.")
    for name, m in ok.items():
        if m["p_permutation"] >= 0.05:
            notes.append(f"{name}: not distinguishable from chance (permutation p = {m['p_permutation']:.2f}).")
    ppg = ok.get("ppg_only")
    if ppg and ppg["auc_ci_low"] <= 0.5:
        notes.append("ppg_only: the 95% confidence interval includes 0.5, so PPG-only features do not separate the groups reliably yet. This is the key question for a PPG-based device.")
    if "bp_invasive" in df.columns and df["bp_invasive"].fillna(False).astype(bool).any():
        n_inv = int(df["bp_invasive"].fillna(False).astype(bool).sum())
        notes.append(f"Blood-pressure features come from arterial lines in {n_inv} of {len(df)} cases. A PPG-only device cannot measure them; use --bp-source cuff for a device-relevant test.")
    return notes


def evaluate(df: pd.DataFrame, cfg: Config) -> dict:
    """Returns {'status', counts, 'univariate' (DataFrame), 'models' (dict), 'notes' (list)}."""
    df = add_derived_columns(df)
    y = df["label"].astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    result = {"n_cases": int(len(df)), "n_pos": n_pos, "n_neg": n_neg}
    if min(n_pos, n_neg) < 4:
        result.update(status="insufficient_data", univariate=pd.DataFrame(), models={}, notes=[],
                      message="Need at least 4 cases in each class. Increase --max-cases.")
        return result

    result["status"] = "ok"
    result["univariate"] = univariate_auc(df, SUMMARY_FEATURES + COVARIATES, cfg)
    result["models"] = {name: cv_evaluate(df, feats, cfg) for name, feats in MODEL_SETS.items()}
    result["notes"] = _notes(result, df)
    return result
