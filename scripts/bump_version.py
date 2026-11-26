"""Bump the pipeline version and keep version.py, README.md and CHANGELOG.md in sync.

Examples (run from the project root):
    python scripts/bump_version.py patch --note "Fix PPG window rejection"
    python scripts/bump_version.py minor --note "Add Hb lab timing"
    python scripts/bump_version.py --set 0.3.0 --note "First obstetric dataset"
Optional: --date 2026-12-01 (defaults to today's date on your computer)
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_PY = ROOT / "src" / "pph_pipeline" / "version.py"
README = ROOT / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"


def read_version() -> str:
    m = re.search(r'__version__ = "([^"]+)"', VERSION_PY.read_text(encoding="utf-8"))
    return m.group(1)


def bump(current: str, part: str) -> str:
    major, minor, patch = (int(x) for x in current.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("part", nargs="?", choices=["major", "minor", "patch"])
    p.add_argument("--set", dest="explicit", help="set an explicit version, e.g. 0.3.0")
    p.add_argument("--note", required=True, help="one-line description for the changelog")
    p.add_argument("--date", default=date.today().isoformat(), help="production date (YYYY-MM-DD)")
    args = p.parse_args()

    if not args.part and not args.explicit:
        p.error("give major|minor|patch or --set X.Y.Z")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        p.error("--date must look like 2026-12-01")

    old = read_version()
    new = args.explicit or bump(old, args.part)
    if not re.fullmatch(r"\d+\.\d+\.\d+", new):
        p.error("version must look like 1.2.3")

    # 1) version.py
    text = VERSION_PY.read_text(encoding="utf-8")
    text = re.sub(r'__version__ = "[^"]+"', f'__version__ = "{new}"', text)
    text = re.sub(r'__production_date__ = "[^"]+"', f'__production_date__ = "{args.date}"', text)
    VERSION_PY.write_text(text, encoding="utf-8")

    # 2) README version block
    block = (f"<!-- VERSION:START -->\n"
             f"**Pipeline version:** {new}  \n"
             f"**Production date:** {args.date}\n"
             f"<!-- VERSION:END -->")
    readme = README.read_text(encoding="utf-8")
    readme = re.sub(r"<!-- VERSION:START -->.*?<!-- VERSION:END -->", block, readme, flags=re.S)
    README.write_text(readme, encoding="utf-8")

    # 3) CHANGELOG entry (newest first)
    entry = f"<!-- CHANGELOG:INSERT -->\n\n## [{new}] - {args.date}\n\n- {args.note}\n"
    log = CHANGELOG.read_text(encoding="utf-8")
    log = log.replace("<!-- CHANGELOG:INSERT -->\n", entry, 1)
    CHANGELOG.write_text(log, encoding="utf-8")

    print(f"Version {old} -> {new} (production date {args.date})")
    print("Next:")
    print("  git add -A")
    print(f'  git commit -m "Release v{new}: {args.note}"')
    print(f'  git tag -a v{new} -m "v{new}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
