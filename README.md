# PPH Pipeline - early haemorrhage signal discovery (VitalDB stage)

<!-- VERSION:START -->
**Pipeline version:** 0.3.0  
**Production date:** 2026-11-27
<!-- VERSION:END -->

> **Research use only.** This is exploratory analysis code. It is not a medical device, is not validated, and must not be used to make clinical decisions.

## End goal

The project's goal is a **non-invasive device that monitors for postpartum haemorrhage (PPH)**. This pipeline is the software stage: it uses public hospital data to find out which signals are worth building a device around. So every result is labelled by what a non-invasive device could actually measure:

| Label | Meaning | Examples |
|---|---|---|
| `ppg_only` | An optical pulse sensor alone | PPG pulse amplitude, perfusion index, PPG pulse rate |
| `needs_bp` | Also needs blood pressure (non-invasive only if it comes from a cuff) | shock index, systolic pressure drop |
| `monitor_hr` | Heart rate from the hospital ECG monitor | monitor heart-rate rise |

Arterial-line pressure is a research reference only, because a device cannot measure it. By default the program says so in every report, and `--bp-source cuff` runs the device-relevant version.

## What it does

```
select  ->  download  ->  summarize  ->  evaluate
(clinical   (signals +     (pick blood-    (models, confidence
 table)      30 s windows,  pressure source, intervals, report,
             quality check) per-case table) figures)
```

Per 30-second window it extracts PPG pulse amplitude (AC), DC, perfusion index, rise time, pulse rate, beat-to-beat variability and quality measures, plus heart rate and systolic pressure from both arterial and cuff sources. Per case it compares surgery against each patient's own baseline (10 minutes before incision).

Evaluation compares four small, pre-specified logistic-regression models: `covariates_only` (what surgery size explains), `ppg_only` (the core device question), `physiology_core` and `combined_core`. Each gets repeated cross-validation, a bootstrap 95% confidence interval and a permutation test.

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
python -m pph_pipeline demo --fast
```

- If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` in Command Prompt.
- In VS Code press `Ctrl+Shift+P > Python: Select Interpreter` and choose `.venv`.
- `demo` runs fully offline on **synthetic** data. If it prints `Done. Results in: outputs\...`, your install works.
- Upgrading from an older version: your cached downloads rebuild automatically, because the quality checks changed what is stored.

## Command guide

Every command has the same shape: `python -m pph_pipeline <command> [options]`. Options go **after** the command. If you run the steps one at a time, repeat the same options on each step. Print this table any time with `python -m pph_pipeline guide`.

<!-- GUIDE:START -->
| Command | What it does | Needs internet | Writes | Use it when |
|---|---|---|---|---|
| `guide` | Prints this reference. | No | Nothing | You want a plain explanation of every command. |
| `recipes` | Prints copy-and-paste example commands for common goals. | No | Nothing | You are not sure what to type. |
| `explore` | Counts cases by department, operation, blood-loss cut-off, transfusion and arterial line, and shows how many cases your options would give. Downloads only the small clinical table. | First time only | data\raw\cases.csv | Before choosing options, so you do not have to guess. |
| `select` | Chooses the candidate cases from your options. | First time only | data\manifests\candidates.csv | Running step by step. |
| `download` | Fetches each case's signals and computes 30-second window features, with a quality check on every PPG window. Skips cases with no PPG track and remembers them. Reuses cached cases. | Yes | data\features\ (windows, quality records) and data\manifests\download_status.csv | Running step by step. This is the slow step. |
| `summarize` | Chooses the blood-pressure source, then builds one baseline-normalised row per case. | No | data\manifests\case_features.csv | Running step by step, or after changing --bp-source. |
| `evaluate` | Runs the models (repeated cross-validation, confidence intervals, permutation test), then writes the report and figures. | No | A new folder in outputs\ (report.md, results.json, tables, figures) | Running step by step, or after changing analysis options. |
| `run` | Does select, download, summarize and evaluate in one go. | Yes | Everything above | Normal use. Time depends on the number of cases and your connection. |
| `demo` | Runs the whole pipeline on synthetic data, with no internet. | No | outputs\..._synthetic | After installing, to check that everything works. |
<!-- GUIDE:END -->

## Running on real VitalDB data

```powershell
python -m pph_pipeline explore
python -m pph_pipeline run --max-cases 10 --fast
```

Start small: each case downloads one `.vital` file from the public VitalDB dataset (needs internet). Results are cached under `data\` so reruns are fast. Full statistics (the default) take a few minutes; `--fast` uses fewer repeats and permutations for quick checks.

### Choosing options (plain English)

Run `explore` first. It prints how many cases each choice would give you before anything large is downloaded.

| You want | Use | What to know |
|---|---|---|
| A bigger sample | `--max-cases 40` | Split evenly between higher- and lower-loss groups. Start with 10, then 40 or more. |
| One specialty | `--department Gyn` | Matches any part of the department name; `explore` lists them. |
| One kind of operation | `--operation Cesarean` | Matches any part of the operation name. Compares like with like. |
| A different blood-loss cut-off | `--ebl-threshold 300` | Lower cut-off gives more higher-loss cases but a weaker contrast. |
| Transfusion instead of blood loss | `--label-mode any_rbc` | Higher-loss group = any red-cell transfusion. |
| Non-invasive blood pressure | `--bp-source cuff` | Cuff readings carried forward up to `--cuff-carry-minutes` (default 10). |
| Arterial or cuff, automatically | `--bp-source either` | Arterial if well covered, otherwise cuff. The type used is recorded per case. |
| Only the first part of surgery | `--analysis-minutes 60` | Stops long surgeries having more chances to hit extreme values. |
| Stricter signal quality | `--strict-qc` | Also rejects windows flagged for motion or baseline drift. |
| Cases with an arterial line | `--require-aline` | Reference experiment only. |
| A quick check | `--fast` | Fewer repeats, permutations and resamples. |
| A different random sample | `--seed 7` | Same seed gives the same cases each time. |
| Redo downloads | `--force` | Ignores the cache. |

### How the pieces fit together

- **Command** says what to do: `explore` (look), `run` (do everything), the four steps `select`, `download`, `summarize`, `evaluate`, plus `demo`, `recipes` and `guide`.
- **Options** say which cases and which analysis to use. They go after the command.
- **Data** is fetched for you. The first `explore` or `run` saves the small clinical table to `data\raw\cases.csv`. `run` and `download` then fetch each case's signals and cache the computed features in `data\features\`.
- **Switching `--bp-source`, `--analysis-minutes` or the label needs no re-download.** Only changing the signal-quality settings or upgrading the pipeline rebuilds the cache.
- **Results** appear in `outputs\<timestamp>_vitaldb\`. Start with `report.md`.

### What this data can and cannot answer

- No dataset in this pipeline is labelled "PPH". VitalDB is surgical data. Its Gynecology group includes some **cesarean sections** (use `explore --operation Cesarean` to count them), but these are surgical records, not postpartum monitoring.
- All recordings are retrospective. Nothing here is live.
- There is no direct blood-flow measurement. The pipeline uses proxies: PPG pulse amplitude and perfusion index (peripheral perfusion) and blood pressure.
- There is only total blood loss per surgery, with no timing, so this version tests association, not early warning.

### Copy-and-paste recipes

Print these any time with `python -m pph_pipeline recipes`.

<!-- RECIPES:START -->
| Goal | Command | What it does |
|---|---|---|
| Look before you download | `python -m pph_pipeline explore` | Prints case counts per department, operation, blood-loss cut-off, transfusion and arterial line. Downloads only the small clinical table. |
| Look at cesarean sections | `python -m pph_pipeline explore --operation Cesarean` | Same report restricted to operations whose name contains 'Cesarean'. VitalDB includes some under Gynecology. |
| Smoke test on real data (10 cases, quick statistics) | `python -m pph_pipeline run --max-cases 10 --fast` | Whole pipeline on 5 higher-loss and 5 lower-loss cases, with fewer repeats so it finishes quickly. |
| First real analysis | `python -m pph_pipeline run --max-cases 40` | Default settings: arterial blood pressure, whole surgery, full statistics. |
| Device-relevant test: cuff blood pressure only | `python -m pph_pipeline run --max-cases 40 --bp-source cuff` | Shock index from the non-invasive cuff (carried forward 10 minutes). Closest to what a device could measure. |
| Cesarean sections only | `python -m pph_pipeline run --operation Cesarean --max-cases 30 --ebl-threshold 500` | Compares like with like, so surgery type cannot explain the difference. Check with explore that enough cases exist. |
| Gynaecology only | `python -m pph_pipeline run --department Gyn --max-cases 30 --ebl-threshold 300` | Gynaecological surgery, higher-loss group = 300 mL or more. |
| First hour only (stops long surgeries dominating) | `python -m pph_pipeline run --max-cases 40 --analysis-minutes 60` | Only the first 60 minutes after incision are analysed for every case. |
| Stricter signal quality | `python -m pph_pipeline run --max-cases 40 --strict-qc` | Also rejects (not just flags) windows with motion or baseline drift. |
| Heavy-bleeding focus (1000 mL or more) | `python -m pph_pipeline run --max-cases 30 --ebl-threshold 1000` | Run explore with the same threshold first: few cases bleed this much. |
| Transfusion as the label instead of blood loss | `python -m pph_pipeline run --label-mode any_rbc --max-cases 40` | Higher-loss group = any red-cell transfusion during surgery. |
| Cleaner arterial data only | `python -m pph_pipeline run --max-cases 40 --require-aline` | Only cases with an arterial line. Reference experiment, not device-relevant. |
| A different random sample of the same size | `python -m pph_pipeline run --max-cases 40 --seed 7` | Same options, different cases. A result that survives a new seed is more believable. |
| Step 1 of 4 (step by step): choose cases | `python -m pph_pipeline select --department Gyn --max-cases 30` | Writes the candidate list. Repeat the SAME options on every step. |
| Step 2 of 4: download signals | `python -m pph_pipeline download --department Gyn --max-cases 30` | Downloads each case and computes window features. Cached, so reruns are fast. |
| Step 3 of 4: build the case table | `python -m pph_pipeline summarize --department Gyn --max-cases 30 --bp-source cuff` | One baseline-normalised row per case. Changing --bp-source here needs no re-download. |
| Step 4 of 4: models and figures | `python -m pph_pipeline evaluate --department Gyn --max-cases 30 --bp-source cuff` | Writes report.md, results.json, tables and figures into a new outputs folder. |
| Redo downloads from scratch | `python -m pph_pipeline run --max-cases 10 --force` | Ignores the cache. Slower; only needed if a download looks corrupted. |
| Offline practice run (synthetic data) | `python -m pph_pipeline demo --fast` | No internet needed. Results are labelled SYNTHETIC. |
| Print the plain-English command reference | `python -m pph_pipeline guide` | What every command reads, writes and needs. |
<!-- RECIPES:END -->

Department and operation names with spaces need quotes, for example `--department "General surgery"`. `Gyn` matches any department containing those letters. Gynaecological surgery is still not childbirth. Treat results as a test of the method, not evidence about PPH.

## Outputs

Each run creates `outputs\<timestamp>_<mode>\` containing:

| File | What it is |
|---|---|
| `report.md` | Plain-language run summary: data quality, model table with confidence intervals, notes on how to read it |
| `results.json` | Every number, plus the exact configuration |
| `case_features.csv` | One row per case, including the blood-pressure source and quality summary |
| `univariate_auc.csv` | Single-feature AUCs with confidence intervals and device relevance |
| `exclusions.csv` | Cases left out and why |
| `qc_summary.csv` | Per-case signal-quality and blood-pressure source summary |
| `environment.json` | Python and package versions, seed, command line |
| `fig_*.png` | Feature box plots and trajectories (with case counts per time bin) |

Demo runs are labelled SYNTHETIC in the folder name, the JSON, the report and the figures.

## Quality control and blood-pressure sources

- Every PPG window is checked for missing data, flat signal, clipping, too few or irregular beats, and low quality score. **Rejected windows are counted by reason**, not silently dropped. Windows are also flagged for motion (low beat-to-beat similarity) and baseline drift. Flagged windows are kept unless `--strict-qc` is used.
- The quality thresholds are provisional. Use the quality summary in `report.md` to judge them before relying on them.
- Beat-to-beat variability (RMSSD) now uses only adjacent valid beats.
- Blood pressure: the source (arterial or cuff), its coverage and the number of missing windows are stored for every case. An invasive reading is never substituted silently: `--bp-source arterial` uses arterial only, `cuff` uses cuff only, and `either` records which it chose. If neither exists, shock index is left missing rather than guessed.

## Dependencies

Python 3.10+ and: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `vitaldb`. Dev/testing: `pytest`. See `requirements.txt`.

## Version control

```powershell
git add -A
git commit -m "Release vX.Y.Z"
git tag -a vX.Y.Z -m "vX.Y.Z"
git push
git push origin vX.Y.Z
```

To release a change, update the version, production date, README block and changelog **in one step**:

```powershell
python scripts/bump_version.py minor --note "First change" --note "Second change"
```

(`patch` = fix, `minor` = new feature, `major` = breaking change, or `--set 0.4.0`.) Tests fail if the README, the command tables or the changelog drift from the code. After editing `recipes.py` or `guide.py`, run `python scripts/sync_readme.py` to refresh the tables in this file. `data/` and `outputs/` are git-ignored.

## Limitations - read before interpreting results

- **Surgical, not postpartum.** VitalDB comes from one Korean hospital. Cesarean sections in the Gynecology group are obstetric operations, but spinal anaesthesia, large phenylephrine doses and surgical blood-loss estimates differ from the postpartum period, and can mimic or mask a haemorrhage signal.
- **Anaesthetised patients.** Anaesthetics and vasopressors change perfusion. Motion artefact is lower than on a postpartum ward, so performance will look optimistic.
- **The label is a proxy.** Estimated blood loss in surgery is not a PPH diagnosis. WHO (2025) defines PPH as measured blood loss of at least 300 mL with an abnormal sign (pulse above 100, systolic below 100, diastolic below 60 or shock index above 1) or at least 500 mL within 24 hours of birth. Most of those signs are blood-pressure based.
- **No bleeding timing.** Only totals exist, so v0.3 is an association study, not an early-warning test.
- **PPG units are arbitrary.** Only within-patient changes are used.
- **Small samples.** With under about 30 cases per group, confidence intervals are very wide; the report says so.
- **Arterial pressure is not device-relevant.** Shock index from arterial lines cannot be measured by a PPG-only device.

## Roadmap

- v0.4: three experiments (PPG-only, non-invasive combined, clinical reference), fixed-time and rolling-window features, more PPG features, WHO abnormal-sign flags.
- v0.5: baseline versus intraoperative covariates, calibration and other metrics, grouped cross-validation, a rough warning-time test using timed haemoglobin draws.
- v0.6: obstetric dataset adapter, WHO-aligned labels, timed bleeding records, external validation.
- Later: raw-signal cache for faster reruns, dashboard.

## License

Code is released under the MIT License (see `LICENSE`). The VitalDB data are separate and carry their own terms (CC BY 4.0). The software is provided as-is for research use and is not a medical device.

## Continuous testing

`.github/workflows/tests.yml` runs the offline test suite on Windows and Linux every time you push to GitHub. Check the *Actions* tab of your repository for a green tick.

## Data credit

Data: VitalDB (CC BY 4.0). Lee HC, Park Y, Yoon SB, et al. *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients.* Sci Data 9, 279 (2022). https://doi.org/10.1038/s41597-022-01411-5

## Project layout

```
pph-pipeline/
  README.md  CHANGELOG.md  LICENSE  pyproject.toml  requirements*.txt
  scripts/            bump_version.py  sync_readme.py
  src/pph_pipeline/   version.py config.py vitaldb_client.py cohort.py
                      signal_features.py bp.py case_features.py model.py
                      plotting.py synthetic.py pipeline.py cli.py
                      guide.py recipes.py
  tests/              offline unit tests
  data/  outputs/     (git-ignored, created at run time)
```
