"""Unified dispatcher for the current manuscript-figure tree.

Examples
--------
    python -m figures.main --list
    python -m figures.main --figure fig2
    python -m figures.main --figure fig2 --figure figS6
    python -m figures.main --all

The dispatcher follows the files that actually exist under ``figures/``:

* ``figures/<id>/plot.py`` uses its public ``draw()`` function;
* if ``plot.py`` is absent, the first ``make_*.py`` uses its public ``main()``
  or ``draw()`` function.

The figure layer does not need a run identifier.  It reads the current
analysis-layer products and writes the current figure outputs in the matching
figure directory.  Versioned run directories, when used by an analysis
workflow, are an upstream provenance concern rather than a plotting CLI
argument.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import importlib
from pathlib import Path
import re
import sys
from typing import Any, Callable


FIGURE_ROOT = Path(__file__).resolve().parent


DESCRIPTIONS = {
    "fig2": "unified trajectory recovery and robust health prediction",
    "fig3": "robustness boundaries under progressive degradation",
    "fig4": "distinct recoverability of population and individual degradation",
    "fig5": "Population–Individual properties",
    "fig6": "generality across heterogeneous battery systems",
    "figS1": "reconstruction examples",
    "figS2": "per-cycle waveform error distributions",
    "figS3": "SOH parity",
    "figS4": "downstream comparison",
    "figS5": "stress reconstruction trajectories",
    "figS6": "battery-level Population and Individual recovery under observation degradation",
    "figS7": "R4 longitudinal, spatial, and prognostic properties",
    "figS8": "external representative trajectory recovery",
    "figS9": "system-dependent Individual-fidelity/downstream-error association",
    "figS10": "PC-mask routing schematic",
}


@dataclass(frozen=True)
class FigureSpec:
    """One executable figure entry discovered from the current tree."""

    figure_id: str
    module_name: str | None
    entry_name: str | None
    source: str | None

    @property
    def description(self) -> str:
        return DESCRIPTIONS.get(self.figure_id, "")

    @property
    def ready(self) -> bool:
        # make-only entries intentionally leave entry_name unset; the runtime
        # resolver selects their public draw() or main() function.
        return self.module_name is not None and self.source is not None


def _sort_key(path: Path) -> tuple[int, int, str]:
    match = re.fullmatch(r"fig(S?)(\d+)", path.name)
    if not match:
        return (2, 0, path.name)
    supplement = 1 if match.group(1) else 0
    return (supplement, int(match.group(2)), path.name)


def discover_figures(root: Path = FIGURE_ROOT) -> tuple[FigureSpec, ...]:
    """Discover figure entries without importing any plotting module."""

    specs: list[FigureSpec] = []
    for directory in sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith("fig")),
        key=_sort_key,
    ):
        figure_id = directory.name
        plot = directory / "plot.py"
        if plot.is_file():
            specs.append(
                FigureSpec(
                    figure_id,
                    f"figures.{figure_id}.plot",
                    "draw",
                    str(plot.relative_to(root.parent)),
                )
            )
            continue

        makers = sorted(directory.glob("make_*.py"))
        if makers:
            # Current make-only figures expose main(); the runtime resolver
            # also accepts draw() for a future make-only figure.
            maker = makers[0]
            specs.append(
                FigureSpec(
                    figure_id,
                    f"figures.{figure_id}.{maker.stem}",
                    None,
                    str(maker.relative_to(root.parent)),
                )
            )
            continue

        specs.append(FigureSpec(figure_id, None, None, None))
    return tuple(specs)


# Kept as a public compatibility mapping for callers that used the old module.
FIGURES = {spec.figure_id: spec.description for spec in discover_figures()}


def parser() -> argparse.ArgumentParser:
    specs = discover_figures()
    choices = tuple(spec.figure_id for spec in specs)
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--figure",
        dest="figures",
        action="append",
        choices=choices,
        metavar="ID",
        help="figure ID; repeat the option to render several figures",
    )
    command.add_argument("--all", action="store_true", help="render every discovered figure")
    command.add_argument("--list", action="store_true", help="show discovered figure entries")
    command.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue rendering other figures after one figure fails",
    )
    return command


def _resolve_entry(module: Any, spec: FigureSpec) -> Callable[[], Any]:
    """Resolve the public callable after importing one figure module."""

    if spec.entry_name is not None:
        candidate = getattr(module, spec.entry_name, None)
        if callable(candidate):
            return candidate

    for name in ("draw", "main"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate

    raise AttributeError(
        f"{spec.figure_id} has no callable draw() or main() in {spec.source}"
    )


def _print_result(result: Any) -> None:
    """Print only useful output paths from heterogeneous figure return values."""

    if isinstance(result, Mapping):
        # Fig. 5 returns assets + outputs + qc; only report the generated
        # output paths, not large DataFrames or internal plotting state.
        if "outputs" in result:
            _print_result(result["outputs"])
            if "qc" in result:
                _print_result(result["qc"])
            return
        for value in result.values():
            if isinstance(value, (str, Path)):
                print(value)
        return
    if isinstance(result, (str, Path)):
        print(result)
        return
    if isinstance(result, (list, tuple, set)):
        for value in result:
            _print_result(value)


def run_figure(spec: FigureSpec) -> None:
    """Import and run one discovered figure entry."""

    if not spec.ready:
        raise RuntimeError(f"{spec.figure_id} has no plot.py or make_*.py entry point")
    assert spec.module_name is not None
    module = importlib.import_module(spec.module_name)
    entry = _resolve_entry(module, spec)
    print(f"[{spec.figure_id}] {spec.source}")
    _print_result(entry())


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parser().parse_args()
    specs = discover_figures()
    by_id = {spec.figure_id: spec for spec in specs}

    if args.list:
        for spec in specs:
            status = "ready" if spec.ready else "missing-entry"
            source = spec.source or "-"
            print(f"{spec.figure_id:7s} [{status:13s}] {source:35s} {spec.description}")
        return 0

    if args.all and args.figures:
        raise SystemExit("--all and --figure cannot be used together")
    selected_ids = [spec.figure_id for spec in specs] if args.all else (args.figures or [])
    if not selected_ids:
        raise SystemExit("use --list, --all, or at least one --figure ID")

    failures: list[tuple[str, Exception]] = []
    for figure_id in selected_ids:
        try:
            run_figure(by_id[figure_id])
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((figure_id, exc))
            print(f"[{figure_id}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    if failures:
        raise SystemExit(f"{len(failures)} figure(s) failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
