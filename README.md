# PPH Pipeline - early haemorrhage signal discovery (VitalDB stage)

<!-- VERSION:START -->
**Pipeline version:** 0.2.1  
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

Individual steps: `select`, `download`, `summarize`, `evaluate`.

### Choosing options (plain English)

Run `explore` first. It needs only the clinical table (no signals are downloaded) and prints how many cases each choice would give you:

```powershell
python -m pph_pipeline explore
python -m pph_pipeline explore --department Gynecology --ebl-threshold 300
```

| You want | Use | What to know |
|---|---|---|
| A bigger sample | `--max-cases 40` | Split evenly between higher- and lower-loss groups. Start with 10, then 40 or more. |
| One specialty only | `--department Gynecology` | Matches any part of the department name; `explore` lists the names. |
| A different blood-loss cut-off | `--ebl-threshold 300` | Lower cut-off gives more higher-loss cases but a weaker contrast between groups. |
| Transfusion instead of blood loss | `--label-mode any_rbc` | Labels a case higher-loss if any red cells were given. |
| Cleaner blood-pressure data | `--require-aline` | Keeps only cases with an arterial line. Fewer cases, better shock index. |
| A different random sample | `--seed 7` | Same seed gives the same cases each time. |
| Redo downloads | `--force` | Ignores the cache. |

### How the pieces fit together

Every command has the same shape: `python -m pph_pipeline <command> [options]`.

- **Command** says what to do: `explore` (look), `run` (do everything), the four steps `select`, `download`, `summarize`, `evaluate`, plus `demo` and `recipes`.
- **Options** say which cases to use: `--department`, `--label-mode`, `--ebl-threshold`, `--require-aline`, `--max-cases`, `--seed`. They go **after** the command. If you run the steps one by one, repeat the same options each time.
- **Data** is fetched for you, so there is nothing to download by hand. The first `explore` or `run` saves the small clinical table to `data\raw\cases.csv`. `run` and `download` then fetch each case's signals (internet needed) and cache the computed features in `data\features\`.
- **Results** appear in `outputs\<timestamp>_vitaldb\`: open `results.json`, the two PNG figures, and `case_features.csv` (opens in Excel).

Print the recipes below any time with `python -m pph_pipeline recipes`.

### What this data can and cannot answer

- No dataset in this pipeline is labelled "PPH". VitalDB is surgical data, so the closest you can get is gynaecological surgery with higher blood loss or transfusion.
- All recordings are retrospective. Nothing here is live.
- There is no direct blood-flow measurement. The pipeline uses proxies: PPG pulse amplitude and perfusion index (peripheral perfusion) and arterial pressure (best with `--require-aline`). VitalDB may hold cardiac-output tracks for some cases, which this version does not use.

### Copy-and-paste recipes

<!-- RECIPES:START -->
| Goal | Command | What it does |
|---|---|---|
| Look before you download | `python -m pph_pipeline explore` | Prints case counts per department, blood-loss cut-off, transfusion and arterial line. Downloads only the small clinical table. |
| Look at gynaecology with a lower blood-loss cut-off | `python -m pph_pipeline explore --department Gyn --ebl-threshold 300` | Same report, but for departments containing 'Gyn' and a 300 mL cut-off. |
| Smoke test on real data (10 cases) | `python -m pph_pipeline run --max-cases 10` | Whole pipeline on 5 higher-loss and 5 lower-loss cases. Use this to check that everything works. |
| First real analysis, best blood-pressure data | `python -m pph_pipeline run --max-cases 40 --require-aline` | Only cases with an arterial line (beat-to-beat pressure), so the shock index is more reliable. |
| Gynaecology only (closest to obstetrics here) | `python -m pph_pipeline run --department Gyn --max-cases 30 --ebl-threshold 300` | Gynaecological surgery, higher-loss group = 300 mL or more. Still not childbirth. |
| Heavy-bleeding focus (1000 mL or more) | `python -m pph_pipeline run --max-cases 30 --ebl-threshold 1000` | Run explore with the same threshold first: few cases bleed this much, so you may not get 15. |
| Transfusion as the label instead of blood loss | `python -m pph_pipeline run --label-mode any_rbc --max-cases 40` | Higher-loss group = any red-cell transfusion during surgery. |
| Gynaecology + transfusion + arterial line | `python -m pph_pipeline run --department Gyn --label-mode any_rbc --require-aline --max-cases 30` | Several filters together. Check with explore that enough cases remain. |
| A different random sample of the same size | `python -m pph_pipeline run --max-cases 40 --seed 7` | Same options, different cases. A result that survives a new seed is more believable. |
| Step 1 of 4 (step by step): choose cases | `python -m pph_pipeline select --department Gyn --max-cases 30` | Writes the candidate list. Repeat the SAME options on every step. |
| Step 2 of 4: download signals | `python -m pph_pipeline download --department Gyn --max-cases 30` | Downloads each case and computes 30-second window features. Cached, so reruns are fast. |
| Step 3 of 4: build the case table | `python -m pph_pipeline summarize --department Gyn --max-cases 30` | One baseline-normalised row per case. |
| Step 4 of 4: models and figures | `python -m pph_pipeline evaluate --department Gyn --max-cases 30` | Writes results.json, tables and figures into a new outputs folder. |
| Redo downloads from scratch | `python -m pph_pipeline run --max-cases 10 --force` | Ignores the cache. Slower; only needed if a download looks corrupted. |
| Offline practice run (synthetic data) | `python -m pph_pipeline demo` | No internet needed. Results are labelled SYNTHETIC. |
<!-- RECIPES:END -->

Department names with spaces need quotes, for example `--department "General surgery"`. `Gyn` matches any department containing those letters, and `explore` shows the exact names. Gynaecological surgery is still not childbirth. Treat results as a test of the method, not evidence about PPH. The same commands exist as `pph-pipeline <command>`.

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
