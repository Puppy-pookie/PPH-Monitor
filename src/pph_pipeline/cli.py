"""Command line interface:  python -m pph_pipeline <command>  (or: pph-pipeline <command>)"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import pipeline
from .recipes import format_recipes
from .config import Config
from .version import __production_date__, __version__


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data-dir", default="data", help="where downloads/features are cached")
    p.add_argument("--output-dir", default="outputs", help="where results are written")
    p.add_argument("--max-cases", type=int, default=20,
                   help="total usable cases wanted, split evenly between higher- and lower-blood-loss groups")
    p.add_argument("--department", default=None,
                   help='keep only cases whose Department contains this text, e.g. "Gynecology" (run "explore" to see the list)')
    p.add_argument("--label-mode", choices=["ebl", "any_rbc"], default="ebl",
                   help="ebl = estimated blood loss >= --ebl-threshold; any_rbc = any red-cell transfusion")
    p.add_argument("--ebl-threshold", type=float, default=500.0,
                   help="blood-loss cut-off in mL that defines the higher-loss group (only for --label-mode ebl)")
    p.add_argument("--require-aline", action="store_true",
                   help="only cases with an arterial line (beat-to-beat blood pressure, better shock-index data)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="re-download / recompute cached items")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    parser = argparse.ArgumentParser(
        prog="pph-pipeline",
        description="PPH early-detection research pipeline (VitalDB stage).")
    parser.add_argument("--version", action="version",
                        version=f"pph-pipeline {__version__} (produced {__production_date__})")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", parents=[common], help="offline end-to-end run on SYNTHETIC data")
    d.add_argument("--n-cases", type=int, default=24)
    sub.add_parser("recipes", help="print copy-and-paste example commands for common goals")
    sub.add_parser("explore", parents=[common],
                   help="show how many cases each option would give (departments, thresholds); no signals downloaded")
    sub.add_parser("select", parents=[common], help="download clinical table, pick candidate cases")
    sub.add_parser("download", parents=[common], help="download signals, compute window features")
    sub.add_parser("summarize", parents=[common], help="build the case-level feature table")
    sub.add_parser("evaluate", parents=[common], help="models + figures from the case-level table")
    sub.add_parser("run", parents=[common], help="select + download + summarize + evaluate")
    return parser


def config_from_args(args) -> Config:
    return Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        max_cases=args.max_cases,
        department=args.department,
        label_mode=args.label_mode,
        ebl_threshold_ml=args.ebl_threshold,
        require_aline=args.require_aline,
        seed=args.seed,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "recipes":
        print(format_recipes())
        return 0
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = config_from_args(args)
    logging.getLogger(__name__).info("pph-pipeline %s (produced %s)", __version__, __production_date__)

    if args.cmd == "demo":
        out = pipeline.run_demo(cfg, n_cases=args.n_cases)
    elif args.cmd == "explore":
        pipeline.step_explore(cfg)
        return 0
    elif args.cmd == "select":
        pipeline.step_select(cfg, force=args.force)
        return 0
    elif args.cmd == "download":
        pipeline.step_download(cfg, force=args.force)
        return 0
    elif args.cmd == "summarize":
        pipeline.step_summarize(cfg)
        return 0
    elif args.cmd == "evaluate":
        table, windows = pipeline.step_summarize(cfg)
        out = pipeline.step_evaluate(cfg, table, windows, mode="vitaldb")
    elif args.cmd == "run":
        out = pipeline.run_all(cfg, force=args.force)
    else:  # pragma: no cover
        return 2
    print(f"\nDone. Results in: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
