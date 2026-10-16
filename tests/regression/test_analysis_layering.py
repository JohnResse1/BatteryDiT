"""Enforce the analysis / figure layering contract.

`analysis/` computes statistics from `results/`; `figures/` only draws from
`analysis/outputs/`.  Crossing that boundary is what made the previous
repository impossible to audit, so the rule is checked mechanically.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ANALYSIS_LAYER = REPO / "analysis"
FIGURE_ROOT = REPO / "figures"
FIGURE_DIRS = sorted(p for p in FIGURE_ROOT.glob("fig*") if p.is_dir())


def _python_files(root: Path):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", _python_files(ANALYSIS_LAYER), ids=lambda p: p.name)
def test_analysis_layer_does_not_plot(path: Path) -> None:
    offenders = {n for n in _imports(path)
                 if n.split(".")[0] in {"matplotlib", "seaborn", "plotly"}}
    assert not offenders, f"{path} is in the analysis layer but imports {offenders}"


def test_figures_live_outside_analysis() -> None:
    assert not any((ANALYSIS_LAYER / name).exists() for name in ("figures", "supplementary")), \
        "figure code must live in the top-level figures/ directory"


def _string_literals(path: Path) -> list[str]:
    """Every string constant in the module (docstrings included would be noise,
    so only expression-level strings are inspected)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    literals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append(node.value)
    return literals


@pytest.mark.parametrize("layer", FIGURE_DIRS, ids=lambda p: p.name)
def test_figure_layer_does_not_read_results(layer: Path) -> None:
    banned_names = {"RESULTS_ROOT", "FormalRun", "condition_matrix", "require_runs"}
    for path in _python_files(layer):
        imports = _imports(path)
        offenders = {n for n in imports if n.split(".")[-1] in banned_names}
        assert not offenders, (
            f"{path} imports formal-results helpers ({offenders}); figures must "
            "consume analysis/outputs/")
        # a path literal that walks into the formal results tree
        for value in _string_literals(path):
            assert not value.startswith("results/"), (
                f"{path} opens a results/ path literal ({value!r})")


@pytest.mark.parametrize("layer", FIGURE_DIRS, ids=lambda p: p.name)
def test_figure_layer_does_not_compute_statistics(layer: Path) -> None:
    banned = {"scipy", "sklearn", "statsmodels"}
    for path in _python_files(layer):
        offenders = {n for n in _imports(path) if n.split(".")[0] in banned}
        assert not offenders, (
            f"{path} computes statistics ({offenders}); move it to analysis/")


def test_every_figure_has_a_plotting_entrypoint() -> None:
    for layer in FIGURE_DIRS:
        has_plot_module = (layer / "plot.py").is_file()
        has_make_module = any(layer.glob("make_*.py"))
        assert has_plot_module or has_make_module, (
            f"{layer} has no plot.py or make_*.py plotting entry point"
        )
