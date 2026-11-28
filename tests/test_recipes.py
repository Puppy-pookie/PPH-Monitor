"""Every example command must actually work with the real parser and appear in the README."""
import shlex
from pathlib import Path

from pph_pipeline.cli import build_parser, main
from pph_pipeline.recipes import PREFIX, RECIPES

ROOT = Path(__file__).resolve().parent.parent


def test_every_recipe_parses():
    parser = build_parser()
    for _goal, args, _what in RECIPES:
        parser.parse_args(shlex.split(args))  # raises SystemExit if the example is invalid


def test_readme_contains_every_recipe():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for _goal, args, _what in RECIPES:
        assert f"{PREFIX} {args}" in text


def test_recipes_command_prints(capsys):
    assert main(["recipes"]) == 0
    out = capsys.readouterr().out
    assert f"{PREFIX} explore --operation Cesarean" in out
    assert "Options go AFTER the command" in out
