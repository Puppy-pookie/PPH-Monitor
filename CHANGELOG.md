# Changelog

All notable changes to this pipeline, newest first.
Managed by `scripts/bump_version.py` - add entries with that script, not by hand.

<!-- CHANGELOG:INSERT -->

## [0.3.0] - 2026-11-27

- Data integrity: the blood-pressure source, its coverage and the missing-window count are saved for every case and kept for cached cases (the v0.2 cache bug is fixed)
- Arterial and cuff pressure are stored separately and chosen with --bp-source arterial, cuff or either; cuff readings are carried forward for a limited time; an invasive reading is never substituted silently
- PPG quality control: missing, flat, clipped, too few or irregular beats and low-quality windows are rejected with a recorded reason; motion (beat similarity) and baseline drift are flagged, and rejected with --strict-qc
- Fix RMSSD so it only uses differences between adjacent valid beats
- Honest evaluation: repeated cross-validation, bootstrap 95% confidence intervals and permutation tests, with small pre-specified models including a PPG-only model; reports say how to read the results
- Every feature is tagged by what a non-invasive device would need (ppg_only, needs_bp, monitor_hr)
- New options: --operation (for example Cesarean), --bp-source, --cuff-carry-minutes, --analysis-minutes, --strict-qc, --fast; explore now lists operations and patients with several surgeries
- New guide command and refreshed recipes; README command tables are generated from the code and a test checks they stay in sync
- New outputs: report.md, exclusions.csv, qc_summary.csv, environment.json; trajectory plots show the number of cases in every time bin
- Cached downloads rebuild automatically when the version or quality settings change
- README corrected: VitalDB's Gynecology group includes cesarean sections; limitations now quote the WHO 2025 PPH definition

## [0.2.2] - 2026-11-26

- Add MIT license, license metadata, and a GitHub Actions workflow that runs the tests on Windows and Linux

## [0.2.1] - 2026-11-26

- Add recipes command and copy-and-paste example commands (CLI and README); document what the data can and cannot answer

## [0.2.0] - 2026-11-26

- Add explore command and plain-English options guide; clearer --help text for every option

## [0.1.0] - 2026-11-26

- Initial release: VitalDB cohort selection, PPG and shock-index window features, case-level baseline-normalised summary, cross-validated evaluation (covariates-only vs physiology-only vs combined), offline synthetic demo, unit tests.
