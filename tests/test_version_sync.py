"""README.md and CHANGELOG.md must carry the same version/date as version.py."""
import re
from pathlib import Path

from pph_pipeline.version import __production_date__, __version__

ROOT = Path(__file__).resolve().parent.parent


def test_readme_block_matches():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    block = re.search(r"<!-- VERSION:START -->(.*?)<!-- VERSION:END -->", text, re.S).group(1)
    assert __version__ in block
    assert __production_date__ in block


def test_changelog_has_current_version():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{__version__}] - {__production_date__}" in text
