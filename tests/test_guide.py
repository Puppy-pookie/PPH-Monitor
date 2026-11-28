"""The command guide, the parser and the README tables must all agree."""
import importlib.util
from pathlib import Path

from pph_pipeline.cli import build_parser
from pph_pipeline.guide import GUIDE, format_guide

ROOT = Path(__file__).resolve().parent.parent


def _parser_commands():
    sub = [a for a in build_parser()._actions if a.dest == "cmd"][0]
    return set(sub.choices)


def test_every_command_is_in_the_guide_and_vice_versa():
    assert _parser_commands() == {row[0] for row in GUIDE}


def test_guide_text_mentions_each_command_and_the_pattern():
    text = format_guide()
    for cmd in _parser_commands():
        assert cmd in text
    assert "Options go AFTER the command" in text


def test_readme_tables_are_in_sync_with_code():
    spec = importlib.util.spec_from_file_location("sync_readme", ROOT / "scripts" / "sync_readme.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert mod.render(text) == text, "README tables are out of date: run python scripts/sync_readme.py"
