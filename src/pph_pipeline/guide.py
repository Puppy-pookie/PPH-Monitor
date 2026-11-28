"""Plain-English reference for every command. Printed by:  python -m pph_pipeline guide

tests/test_guide.py checks that every command the program accepts is described here.
"""

from __future__ import annotations

# (command, what it does, needs internet, writes, use it when)
GUIDE = [
    ("guide", "Prints this reference.", "No", "Nothing",
     "You want a plain explanation of every command."),
    ("recipes", "Prints copy-and-paste example commands for common goals.", "No", "Nothing",
     "You are not sure what to type."),
    ("explore", "Counts cases by department, operation, blood-loss cut-off, transfusion and arterial line, "
                "and shows how many cases your options would give. Downloads only the small clinical table.",
     "First time only", r"data\raw\cases.csv",
     "Before choosing options, so you do not have to guess."),
    ("select", "Chooses the candidate cases from your options.", "First time only",
     r"data\manifests\candidates.csv", "Running step by step."),
    ("download", "Fetches each case's signals and computes 30-second window features, with a quality check "
                 "on every PPG window. Skips cases with no PPG track and remembers them. Reuses cached cases.",
     "Yes", r"data\features\ (windows, quality records) and data\manifests\download_status.csv",
     "Running step by step. This is the slow step."),
    ("summarize", "Chooses the blood-pressure source, then builds one baseline-normalised row per case.",
     "No", r"data\manifests\case_features.csv", "Running step by step, or after changing --bp-source."),
    ("evaluate", "Runs the models (repeated cross-validation, confidence intervals, permutation test), "
                 "then writes the report and figures.",
     "No", r"A new folder in outputs\ (report.md, results.json, tables, figures)",
     "Running step by step, or after changing analysis options."),
    ("run", "Does select, download, summarize and evaluate in one go.", "Yes", "Everything above",
     "Normal use. Time depends on the number of cases and your connection."),
    ("demo", "Runs the whole pipeline on synthetic data, with no internet.", "No",
     r"outputs\..._synthetic", "After installing, to check that everything works."),
]

PATTERN = "python -m pph_pipeline <command> [options]"
PATTERN_NOTE = ("Options go AFTER the command. If you run the steps one at a time, repeat the same options "
                "on each step. Run 'python -m pph_pipeline <command> --help' for the options a command accepts.")


def format_guide() -> str:
    lines = [f"Pattern:  {PATTERN}", PATTERN_NOTE, ""]
    for cmd, what, net, writes, use in GUIDE:
        lines.append(f"{cmd}")
        lines.append(f"    What it does : {what}")
        lines.append(f"    Internet     : {net}")
        lines.append(f"    Writes       : {writes}")
        lines.append(f"    Use it when  : {use}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def guide_table_markdown() -> str:
    rows = ["| Command | What it does | Needs internet | Writes | Use it when |", "|---|---|---|---|---|"]
    for cmd, what, net, writes, use in GUIDE:
        rows.append(f"| `{cmd}` | {what} | {net} | {writes.replace('|', '/')} | {use} |")
    return "\n".join(rows)


def command_one_liners() -> str:
    """Short list used in the top-level --help."""
    width = max(len(c) for c, *_ in GUIDE)
    return "\n".join(f"  {c.ljust(width)}  {what.split('.')[0]}." for c, what, *_ in GUIDE)
