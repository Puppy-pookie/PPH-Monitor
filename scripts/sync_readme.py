"""Refresh the generated tables in README.md from guide.py and recipes.py.

Run from the project root after editing either file:
    python scripts/sync_readme.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pph_pipeline.guide import guide_table_markdown  # noqa: E402
from pph_pipeline.recipes import PREFIX, RECIPES  # noqa: E402


def recipes_table_markdown() -> str:
    rows = ["| Goal | Command | What it does |", "|---|---|---|"]
    rows += [f"| {g} | `{PREFIX} {a}` | {w} |" for g, a, w in RECIPES]
    return "\n".join(rows)


def render(readme_text: str) -> str:
    for tag, body in (("GUIDE", guide_table_markdown()), ("RECIPES", recipes_table_markdown())):
        pattern = rf"<!-- {tag}:START -->.*?<!-- {tag}:END -->"
        block = f"<!-- {tag}:START -->\n{body}\n<!-- {tag}:END -->"
        readme_text = re.sub(pattern, lambda m: block, readme_text, flags=re.S)
    return readme_text


def main() -> int:
    path = ROOT / "README.md"
    old = path.read_text(encoding="utf-8")
    new = render(old)
    path.write_text(new, encoding="utf-8")
    print("README.md tables " + ("updated" if new != old else "already up to date"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
