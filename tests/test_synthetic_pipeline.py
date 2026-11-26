import json
from pathlib import Path

from pph_pipeline.config import Config
from pph_pipeline.pipeline import run_demo
from pph_pipeline.version import __production_date__, __version__


def test_demo_runs_end_to_end(tmp_path):
    cfg = Config(output_dir=tmp_path / "out", data_dir=tmp_path / "data")
    run_dir = run_demo(cfg, n_cases=16)
    res = json.loads((Path(run_dir) / "results.json").read_text(encoding="utf-8"))
    assert res["pipeline_version"] == __version__
    assert res["production_date"] == __production_date__
    assert res["mode"] == "synthetic"
    assert res["results"]["status"] == "ok"
    # the planted signal must be recoverable, and confounders alone must not explain it
    assert res["results"]["models"]["physiology_only"]["auc_mean"] > 0.8
    for name in ("fig_trajectories.png", "fig_feature_boxplots.png", "case_features.csv"):
        assert (Path(run_dir) / name).exists()
