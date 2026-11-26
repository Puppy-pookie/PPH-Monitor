"""Case-level evaluation: single-feature AUCs and cross-validated logistic regression.

Three models are compared so that you can see whether physiology adds anything
beyond obvious confounders (surgery duration, vasopressor doses, age, ASA):
  covariates_only  <  physiology_only  /  combined
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .case_features import SUMMARY_FEATURES
from .config import Config

COVARIATES = ["age", "asa", "emop", "sex_male", "intraop_phe", "intraop_eph", "intraop_epi", "duration_min"]


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sex" in df.columns:
        df["sex_male"] = df["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0})
    return df


def _usable(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 3]


def univariate_auc(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    y = df["label"].astype(int)
    for f in _usable(df, features):
        x = pd.to_numeric(df[f], errors="coerce")
        ok = x.notna()
        if y[ok].nunique() < 2:
            continue
        auc = float(roc_auc_score(y[ok], x[ok]))
        rows.append({"feature": f, "auc": auc, "auc_direction_free": max(auc, 1 - auc), "n": int(ok.sum())})
    out = pd.DataFrame(rows)
    return out.sort_values("auc_direction_free", ascending=False).reset_index(drop=True) if len(out) else out


def cv_auc(df: pd.DataFrame, features: list[str], cfg: Config) -> dict:
    cols = _usable(df, features)
    if not cols:
        return {"status": "no_features"}
    y = df["label"].astype(int).to_numpy()
    k = min(cfg.cv_folds, int(np.bincount(y).min()))
    if k < 2:
        return {"status": "too_few_cases_per_class"}
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return {
        "status": "ok",
        "n_features": len(cols),
        "features": cols,
        "folds": k,
        "auc_mean": float(np.mean(scores)),
        "auc_sd": float(np.std(scores)),
    }


def evaluate(df: pd.DataFrame, cfg: Config) -> dict:
    """Returns {'status', 'n_cases', 'n_pos', 'n_neg', 'univariate' (DataFrame), 'models' (dict)}."""
    df = add_derived_columns(df)
    y = df["label"].astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    result = {"n_cases": int(len(df)), "n_pos": n_pos, "n_neg": n_neg}
    if min(n_pos, n_neg) < 4:
        result["status"] = "insufficient_data"
        result["message"] = "Need at least 4 cases in each class. Increase --max-cases."
        result["univariate"] = pd.DataFrame()
        result["models"] = {}
        return result

    result["status"] = "ok"
    result["univariate"] = univariate_auc(df, SUMMARY_FEATURES + COVARIATES)
    result["models"] = {
        "covariates_only": cv_auc(df, COVARIATES, cfg),
        "physiology_only": cv_auc(df, SUMMARY_FEATURES, cfg),
        "combined": cv_auc(df, SUMMARY_FEATURES + COVARIATES, cfg),
    }
    if min(n_pos, n_neg) < 20:
        result["warning"] = (
            "Fewer than 20 cases in a class: AUC estimates are very noisy. "
            "Use for pipeline testing, not conclusions."
        )
    return result
