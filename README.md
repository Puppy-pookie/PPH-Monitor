# PPH Pipeline - early haemorrhage signal discovery (VitalDB stage)

<!-- VERSION:START -->
**Pipeline version:** 0.1.0  
**Production date:** 2026-11-26
<!-- VERSION:END -->

> **Research use only.** This is exploratory analysis code. It is not a medical device, is not validated, and must not be used to make clinical decisions.

## What it does

Stage 1 of the PPH monitor project (software first, no new hardware). It asks:
*in real surgical patients, do cases with more blood loss show an earlier or bigger haemodynamic-compensation signature - rising shock index, falling PPG pulse amplitude - than cases with less?*

```
select  ->  download  ->  summarize  ->  evaluate
(clinical   (signals +     (per-case,      (models, AUCs,
 table)      30 s windows)  baseline-       figures,
                            normalised)     results.json)
```

Per 30-second window it extracts PPG pulse amplitude (AC), DC, perfusion index, rise time, pulse rate, inter-beat variability and a signal-quality score, plus heart rate, systolic BP and shock index (HR/SBP). Per case it compares surgery against each patient's own baseline (10 minutes before incision).

Evaluation compares three logistic-regression models so you can see whether physiology adds anything beyond confounders (surgery duration, age, ASA, vasopressor doses):
`covariates_only` vs `physiology_only` vs `combined`.

## Quick start (Windows)

1. Install **Python 3.10-3.13** from python.org (tick *Add python.exe to PATH*) and **Git for Windows**.
2. Open the folder in VS Code (`File > Open Folder`), then open a terminal (`Terminal > New Terminal`).
3. Run in PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
pytest
python -m pph_pipeline demo
```

- If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` in Command Prompt.
- In VS Code press `Ctrl+Shift+P > Python: Select Interpreter` and choose `.venv`.
- `demo` runs fully offline on **synthetic** data in a few seconds. If it prints `Done. Results in: outputs\...`, your install works.

## Running on real VitalDB data

```powershell
python -m pph_pipeline run --max-cases 10
```

Start small: each case downloads one `.vital` file from the public VitalDB dataset (needs internet), and results are cached under `data\` so reruns are fast. Useful options:

| Option | Meaning |
|---|---|
| `--max-cases 40` | total usable cases, split evenly between classes |
| `--department Gynecology` | restrict to one department |
| `--label-mode ebl` / `any_rbc` | label = estimated blood loss >= threshold, or any red-cell transfusion |
| `--ebl-threshold 500` | mL cut-off for `ebl` |
| `--require-aline` | only cases with an arterial line (beat-to-beat BP) |
| `--force` | re-download / recompute cached cases |

Individual steps: `select`, `download`, `summarize`, `evaluate`. The same commands exist as `pph-pipeline <command>`.

## Outputs

Each run creates `outputs\<timestamp>_<mode>\` containing `results.json` (includes pipeline version, production date, config and metrics), `case_features.csv`, `univariate_auc.csv`, `fig_feature_boxplots.png` and `fig_trajectories.png`. Demo runs are labelled SYNTHETIC in the folder name, the JSON and the figure titles.

## Dependencies

Python 3.10+ and: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `vitaldb`. Dev/testing: `pytest`. See `requirements.txt`.

## Version control

```powershell
git init
git add -A
git commit -m "Initial commit: pph-pipeline v0.1.0"
git tag -a v0.1.0 -m "v0.1.0"
```

To release a change, update the version, production date, README block and changelog **in one step**:

```powershell
python scripts/bump_version.py patch --note "Describe the change"
git add -A
git commit -m "Release vX.Y.Z"
```

(`patch` = fix, `minor` = new feature, `major` = breaking change, or `--set 0.2.0`.) A test (`tests/test_version_sync.py`) fails if README/CHANGELOG drift from `src/pph_pipeline/version.py`. `data/` and `outputs/` are git-ignored.

## Limitations - read before interpreting results

- **Not obstetric.** VitalDB is non-cardiac surgery (general, thoracic, urological, gynaecological) at one Korean hospital, so results do not transfer to postpartum haemorrhage.
- **Anaesthetised patients.** Anaesthetics and vasopressors change perfusion and are partly adjusted for only through crude covariates. Motion artefact is lower than on a postpartum ward, so performance will look optimistic.
- **Label is a proxy.** Total *estimated* blood loss per surgery is not a PPH diagnosis, and it carries no bleeding *timing*, so v0.1 is a case-level association study, not an early-warning test.
- **PPG units are arbitrary.** Only within-patient changes are used.
- **Small samples.** With under about 20 cases per class, AUCs are very noisy; the pipeline warns about this.

## Roadmap

- v0.2: use laboratory haemoglobin draws to localise bleeding in time and test true lead time versus blood-pressure change.
- v0.3: threshold/alarm analysis (false-alarm rate per hour), motion-robust features.
- Later: validate on real postpartum data with measured blood loss (needs ethics approval and a clinical partner).

## Data credit

Data: VitalDB (CC BY 4.0). Lee HC, Park Y, Yoon SB, et al. *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients.* Sci Data 9, 279 (2022). https://doi.org/10.1038/s41597-022-01411-5

## Project layout

```
pph-pipeline/
  README.md  CHANGELOG.md  pyproject.toml  requirements*.txt
  scripts/bump_version.py
  src/pph_pipeline/   version.py config.py vitaldb_client.py cohort.py
                      signal_features.py case_features.py model.py
                      plotting.py synthetic.py pipeline.py cli.py
  tests/              offline unit tests
  data/  outputs/     (git-ignored, created at run time)
```
