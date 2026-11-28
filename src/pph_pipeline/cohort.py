"""Case labelling and selection from VitalDB's clinical table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

COVARIATE_COLUMNS = [
    "subjectid", "age", "sex", "asa", "emop", "department", "optype", "opname",
    "intraop_ebl", "intraop_rbc", "intraop_phe", "intraop_eph", "intraop_epi",
    "opstart", "opend",
]


def make_label(cases: pd.DataFrame, cfg: Config) -> pd.Series:
    """1 = 'high blood loss' case, 0 = not, NaN = unknown.

    Proxy label only: estimated blood loss in surgery is NOT a postpartum
    haemorrhage diagnosis.
    """
    if cfg.label_mode == "ebl":
        col = pd.to_numeric(cases.get("intraop_ebl"), errors="coerce")
        return (col >= cfg.ebl_threshold_ml).astype(float).where(col.notna())
    if cfg.label_mode == "any_rbc":
        col = pd.to_numeric(cases.get("intraop_rbc"), errors="coerce")
        return (col > 0).astype(float).where(col.notna())
    raise ValueError(f"Unknown label_mode: {cfg.label_mode!r} (use 'ebl' or 'any_rbc')")


def candidate_cases(cases: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Filter the clinical table and attach the label. Shuffled with cfg.seed."""
    df = cases.copy()
    df.columns = [c.lower() for c in df.columns]
    df["label"] = make_label(df, cfg)
    df = df[df["label"].notna()]

    if cfg.department:
        if "department" not in df.columns:
            raise KeyError("Clinical table has no 'department' column")
        df = df[df["department"].astype(str).str.contains(cfg.department, case=False, na=False)]
    if cfg.operation:
        if "opname" not in df.columns:
            raise KeyError("Clinical table has no 'opname' column")
        df = df[df["opname"].astype(str).str.contains(cfg.operation, case=False, na=False, regex=False)]
    if cfg.require_aline and "aline1" in df.columns:
        df = df[df["aline1"].notna()]

    keep = ["caseid", "label"] + [c for c in COVARIATE_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df["caseid"] = df["caseid"].astype(int)
    df["label"] = df["label"].astype(int)
    return df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)


def class_quota(cfg: Config) -> dict[int, int]:
    """Equal number of cases per class (rounded up for class 1)."""
    pos = (cfg.max_cases + 1) // 2
    return {1: pos, 0: cfg.max_cases - pos}


def build_manifest(cands: pd.DataFrame, cfg: Config, overdraw: int = 3) -> pd.DataFrame:
    """Candidate list: up to ``overdraw`` x quota cases per class, so cases with
    missing tracks can be skipped without running out."""
    quota = class_quota(cfg)
    parts = []
    for label, q in quota.items():
        sub = cands[cands["label"] == label].head(max(q * overdraw, q))
        parts.append(sub)
    out = pd.concat(parts).reset_index(drop=True)
    return out
