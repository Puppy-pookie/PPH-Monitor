import json
from pathlib import Path

from pph_pipeline.config import Config
from pph_pipeline.pipeline import run_demo
from pph_pipeline.version import __production_date__, __version__

FAST = dict(cv_repeats=3, perm_repeats=2, n_permutations=20, n_bootstrap=80)


def test_demo_runs_end_to_end(tmp_path):
    cfg = Config(output_dir=tmp_path / "out", data_dir=tmp_path / "data", **FAST)
    run_dir = Path(run_demo(cfg, n_cases=16))
    res = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert res["pipeline_version"] == __version__
    assert res["production_date"] == __production_date__
    assert res["mode"] == "synthetic"
    assert res["results"]["status"] == "ok"
    models = res["results"]["models"]
    # the planted PPG signal must be recoverable; confounders alone must not explain it
    assert models["ppg_only"]["auc_mean"] > 0.8
    assert models["covariates_only"]["auc_mean"] < models["ppg_only"]["auc_mean"]
    for name in ("fig_trajectories.png", "fig_feature_boxplots.png", "case_features.csv", "report.md",
                 "environment.json", "exclusions.csv", "qc_summary.csv", "univariate_auc.csv"):
        assert (run_dir / name).exists(), name
    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert env["pipeline_version"] == __version__ and env["random_seed"] == cfg.seed
    assert "numpy" in env["packages"]


def test_demo_with_cuff_blood_pressure_records_non_invasive_source(tmp_path):
    cfg = Config(output_dir=tmp_path / "out", data_dir=tmp_path / "data", bp_source="cuff", **FAST)
    run_dir = Path(run_demo(cfg, n_cases=16))
    import pandas as pd
    table = pd.read_csv(run_dir / "case_features.csv")
    assert set(table["bp_type"]) == {"cuff"} and not table["bp_invasive"].any()
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "cuff" in report and "arterial lines in" not in report
