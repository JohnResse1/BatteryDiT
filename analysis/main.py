"""Single entry point for generating paper figures from formal results."""
from __future__ import annotations

import argparse
from pathlib import Path

from common.paths import RESULTS_ROOT
from figures import benchmark, multi_dataset, population_individual, stress_test
from supplementary import benchmark as supplementary_benchmark, recoverability


FIGURES = {
    "benchmark": benchmark.draw,
    "stress_test": stress_test.draw,
    "multi_dataset": multi_dataset.draw,
    "population_individual": population_individual.draw,
    "manifold_validation": recoverability.draw_manifold_validation,
    "individual_fidelity": recoverability.draw_individual_fidelity,
    "signature_linkage": recoverability.draw_signature_linkage,
    "reconstruction_examples": supplementary_benchmark.draw_reconstruction_examples,
    "waveform_distributions": supplementary_benchmark.draw_waveform_distributions,
    "soh_parity": supplementary_benchmark.draw_soh_parity,
    "downstream_comparison": supplementary_benchmark.draw_downstream_comparison,
}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--figure", required=True, choices=tuple(FIGURES))
    command.add_argument("--dataset", default="mit")
    command.add_argument("--results_dir", type=Path, default=RESULTS_ROOT)
    return command


def main(args: argparse.Namespace) -> None:
    outputs = FIGURES[args.figure](args.dataset, results_root=args.results_dir)
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main(parser().parse_args())
