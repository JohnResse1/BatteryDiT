"""Single entry point for the analysis layer.

    python -m analysis.main --list
    python -m analysis.main --task benchmark
    python -m analysis.main --task workflow:longitudinal

Each task reads `results/` and writes machine-readable tables into
`analysis/outputs/`, which is the only input the figure layer reads.
"""
from __future__ import annotations

import argparse

from analysis import (benchmark, multi_dataset, paper_results,
                      population_individual, stress_test, stress_trajectories,
                      supplementary)
from analysis.workflows import (cross_method_canonical, cross_system,
                                 longitudinal_canonical,
                                 sliding_occlusion_canonical)


def _run_sliding_occlusion() -> None:
    """Run the canonical sliding-occlusion workflow with its defaults."""
    sliding_occlusion_canonical.run()

TASKS = {
    "benchmark": benchmark.main,
    "stress_test": stress_test.main,
    "stress_trajectories": stress_trajectories.main,
    "multi_dataset": multi_dataset.main,
    "population_individual": population_individual.main,
    "supplementary": supplementary.main,
    "paper_results": paper_results.main,
    "workflow:cross_system": cross_system.run,
    "workflow:cross_method": cross_method_canonical.run,
    "workflow:longitudinal": longitudinal_canonical.run,
    "workflow:sliding_occlusion": _run_sliding_occlusion,
}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--task", choices=tuple(TASKS))
    command.add_argument("--list", action="store_true")
    return command


def main(args: argparse.Namespace) -> None:
    if args.list:
        for name in TASKS:
            print(name)
        return
    if not args.task:
        raise SystemExit("--task is required (or use --list)")
    TASKS[args.task]()


if __name__ == "__main__":
    main(parser().parse_args())
