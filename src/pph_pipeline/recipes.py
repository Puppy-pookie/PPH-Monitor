"""Copy-and-paste command examples. Printed by:  python -m pph_pipeline recipes

Every command here is checked by tests/test_recipes.py: it must parse with the real
command-line parser and must also appear in README.md.
"""

from __future__ import annotations

PREFIX = "python -m pph_pipeline"

# (goal, command arguments after the prefix, what it does)
RECIPES = [
    ("Look before you download",
     "explore",
     "Prints case counts per department, operation, blood-loss cut-off, transfusion and arterial line. Downloads only the small clinical table."),
    ("Look at cesarean sections",
     "explore --operation Cesarean",
     "Same report restricted to operations whose name contains 'Cesarean'. VitalDB includes some under Gynecology."),
    ("Smoke test on real data (10 cases, quick statistics)",
     "run --max-cases 10 --fast",
     "Whole pipeline on 5 higher-loss and 5 lower-loss cases, with fewer repeats so it finishes quickly."),
    ("First real analysis",
     "run --max-cases 40",
     "Default settings: arterial blood pressure, whole surgery, full statistics."),
    ("Device-relevant test: cuff blood pressure only",
     "run --max-cases 40 --bp-source cuff",
     "Shock index from the non-invasive cuff (carried forward 10 minutes). Closest to what a device could measure."),
    ("Cesarean sections only",
     "run --operation Cesarean --max-cases 30 --ebl-threshold 500",
     "Compares like with like, so surgery type cannot explain the difference. Check with explore that enough cases exist."),
    ("Gynaecology only",
     "run --department Gyn --max-cases 30 --ebl-threshold 300",
     "Gynaecological surgery, higher-loss group = 300 mL or more."),
    ("First hour only (stops long surgeries dominating)",
     "run --max-cases 40 --analysis-minutes 60",
     "Only the first 60 minutes after incision are analysed for every case."),
    ("Stricter signal quality",
     "run --max-cases 40 --strict-qc",
     "Also rejects (not just flags) windows with motion or baseline drift."),
    ("Heavy-bleeding focus (1000 mL or more)",
     "run --max-cases 30 --ebl-threshold 1000",
     "Run explore with the same threshold first: few cases bleed this much."),
    ("Transfusion as the label instead of blood loss",
     "run --label-mode any_rbc --max-cases 40",
     "Higher-loss group = any red-cell transfusion during surgery."),
    ("Cleaner arterial data only",
     "run --max-cases 40 --require-aline",
     "Only cases with an arterial line. Reference experiment, not device-relevant."),
    ("A different random sample of the same size",
     "run --max-cases 40 --seed 7",
     "Same options, different cases. A result that survives a new seed is more believable."),
    ("Step 1 of 4 (step by step): choose cases",
     "select --department Gyn --max-cases 30",
     "Writes the candidate list. Repeat the SAME options on every step."),
    ("Step 2 of 4: download signals",
     "download --department Gyn --max-cases 30",
     "Downloads each case and computes window features. Cached, so reruns are fast."),
    ("Step 3 of 4: build the case table",
     "summarize --department Gyn --max-cases 30 --bp-source cuff",
     "One baseline-normalised row per case. Changing --bp-source here needs no re-download."),
    ("Step 4 of 4: models and figures",
     "evaluate --department Gyn --max-cases 30 --bp-source cuff",
     "Writes report.md, results.json, tables and figures into a new outputs folder."),
    ("Redo downloads from scratch",
     "run --max-cases 10 --force",
     "Ignores the cache. Slower; only needed if a download looks corrupted."),
    ("Offline practice run (synthetic data)",
     "demo --fast",
     "No internet needed. Results are labelled SYNTHETIC."),
    ("Print the plain-English command reference",
     "guide",
     "What every command reads, writes and needs."),
]


def format_recipes() -> str:
    lines = [
        "Pattern:  python -m pph_pipeline <command> [options]",
        "Options go AFTER the command. If you use several steps, repeat the same options on each.",
        "",
    ]
    for i, (goal, args, what) in enumerate(RECIPES, 1):
        lines.append(f"{i:>2}. {goal}")
        lines.append(f"    {PREFIX} {args}")
        lines.append(f"    -> {what}")
        lines.append("")
    lines.append("Notes: department names with spaces need quotes, e.g. --department \"General surgery\".")
    lines.append("'Gyn' matches any department containing those letters; run explore to see exact names.")
    return "\n".join(lines)
