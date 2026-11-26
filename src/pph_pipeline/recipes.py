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
     "Prints case counts per department, blood-loss cut-off, transfusion and arterial line. Downloads only the small clinical table."),
    ("Look at gynaecology with a lower blood-loss cut-off",
     "explore --department Gyn --ebl-threshold 300",
     "Same report, but for departments containing 'Gyn' and a 300 mL cut-off."),
    ("Smoke test on real data (10 cases)",
     "run --max-cases 10",
     "Whole pipeline on 5 higher-loss and 5 lower-loss cases. Use this to check that everything works."),
    ("First real analysis, best blood-pressure data",
     "run --max-cases 40 --require-aline",
     "Only cases with an arterial line (beat-to-beat pressure), so the shock index is more reliable."),
    ("Gynaecology only (closest to obstetrics here)",
     "run --department Gyn --max-cases 30 --ebl-threshold 300",
     "Gynaecological surgery, higher-loss group = 300 mL or more. Still not childbirth."),
    ("Heavy-bleeding focus (1000 mL or more)",
     "run --max-cases 30 --ebl-threshold 1000",
     "Run explore with the same threshold first: few cases bleed this much, so you may not get 15."),
    ("Transfusion as the label instead of blood loss",
     "run --label-mode any_rbc --max-cases 40",
     "Higher-loss group = any red-cell transfusion during surgery."),
    ("Gynaecology + transfusion + arterial line",
     "run --department Gyn --label-mode any_rbc --require-aline --max-cases 30",
     "Several filters together. Check with explore that enough cases remain."),
    ("A different random sample of the same size",
     "run --max-cases 40 --seed 7",
     "Same options, different cases. A result that survives a new seed is more believable."),
    ("Step 1 of 4 (step by step): choose cases",
     "select --department Gyn --max-cases 30",
     "Writes the candidate list. Repeat the SAME options on every step."),
    ("Step 2 of 4: download signals",
     "download --department Gyn --max-cases 30",
     "Downloads each case and computes 30-second window features. Cached, so reruns are fast."),
    ("Step 3 of 4: build the case table",
     "summarize --department Gyn --max-cases 30",
     "One baseline-normalised row per case."),
    ("Step 4 of 4: models and figures",
     "evaluate --department Gyn --max-cases 30",
     "Writes results.json, tables and figures into a new outputs folder."),
    ("Redo downloads from scratch",
     "run --max-cases 10 --force",
     "Ignores the cache. Slower; only needed if a download looks corrupted."),
    ("Offline practice run (synthetic data)",
     "demo",
     "No internet needed. Results are labelled SYNTHETIC."),
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
