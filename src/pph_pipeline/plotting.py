"""Figures. Uses the non-interactive Agg backend so it works on any machine."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402

LABEL_NAMES = {0: "lower blood loss", 1: "higher blood loss"}
COLORS = {0: "#3b7dd8", 1: "#d8553b"}


def plot_feature_boxplots(df: pd.DataFrame, univariate: pd.DataFrame, out_path: Path,
                          note: str = "", top_n: int = 4) -> None:
    if univariate is None or univariate.empty:
        return
    feats = [f for f in univariate["feature"].tolist() if f in df.columns][:top_n]
    if not feats:
        return
    fig, axes = plt.subplots(1, len(feats), figsize=(3.2 * len(feats), 3.8), squeeze=False)
    for ax, f in zip(axes[0], feats):
        data = [pd.to_numeric(df.loc[df["label"] == k, f], errors="coerce").dropna() for k in (0, 1)]
        ax.boxplot(data, widths=0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["low", "high"])
        ax.set_title(f, fontsize=9)
    fig.suptitle(f"Top single features by AUC {note}".strip(), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _origin(w: pd.DataFrame, op_start, cfg: Config) -> float:
    if op_start is not None and np.isfinite(op_start):
        return float(op_start)
    return float(w["t_start"].min() + cfg.baseline_s)


def plot_trajectories(cases: pd.DataFrame, windows_by_case: dict, cfg: Config, out_path: Path,
                      note: str = "", bin_min: float = 5.0) -> None:
    """Median (and IQR) trajectory of normalised PPG amplitude and shock index, by group."""
    recs = []
    for _, row in cases.iterrows():
        cid = int(row["caseid"])
        w = windows_by_case.get(cid)
        if w is None or w.empty:
            continue
        t0 = _origin(w, row.get("opstart", np.nan), cfg)
        base = w[(w.t_start >= t0 - cfg.baseline_s) & (w.t_start < t0)]
        base_ac = base["ppg_ac"].median()
        base_si = base["shock_index"].median()
        if not np.isfinite(base_ac) or base_ac <= 0:
            continue
        body = w[w.t_start >= t0].copy()
        body["bin"] = np.floor((body["t_start"] - t0) / 60.0 / bin_min) * bin_min
        body["ac_rel"] = body["ppg_ac"] / base_ac * 100.0
        body["si_delta"] = body["shock_index"] - base_si
        g = body.groupby("bin")[["ac_rel", "si_delta"]].median().reset_index()
        g["label"] = int(row["label"])
        g["caseid"] = cid
        recs.append(g)
    if not recs:
        return
    allr = pd.concat(recs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, ylab in ((axes[0], "ac_rel", "PPG amplitude (% of baseline)"),
                          (axes[1], "si_delta", "Shock index change vs baseline")):
        for label in (0, 1):
            sub = allr[allr["label"] == label]
            agg = sub.groupby("bin")[col].agg(["median", lambda s: s.quantile(0.25),
                                                lambda s: s.quantile(0.75), "count"])
            agg.columns = ["med", "q25", "q75", "n"]
            agg = agg[agg["n"] >= 3]
            if agg.empty:
                continue
            ax.plot(agg.index, agg["med"], color=COLORS[label], label=LABEL_NAMES[label])
            ax.fill_between(agg.index, agg["q25"], agg["q75"], color=COLORS[label], alpha=0.15)
        ax.set_xlabel("minutes since surgery start")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Median trajectories (IQR band) {note}".strip(), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
