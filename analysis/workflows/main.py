"""Canonical Phase-2 workflow driver (R3-R5)."""
from __future__ import annotations

import argparse

from . import (
    cross_method_canonical,
    cross_system_sensitivity,
    computational_efficiency,
    framework_generality,
    future_prognosis_infrastructure,
    future_prognosis_canonical,
    longitudinal_canonical,
    parameter_sensitivity,
    parameter_sensitivity_stress,
    provenance_export,
    reconstruction_prognosis_infrastructure,
    reconstruction_prognosis_canonical,
    sliding_occlusion_canonical,
    tju_ncm_readiness,
)

WORKFLOWS = {
    "population_provenance": provenance_export.main,
    "r3_sensitivity": parameter_sensitivity.main,
    "r3_stress_sensitivity": parameter_sensitivity_stress.main,
    "r4_longitudinal": longitudinal_canonical.main,
    "r4_occlusion_schema": sliding_occlusion_canonical.main,
    "r4_occlusion": sliding_occlusion_canonical.main,
    "r4_prognosis_infrastructure": future_prognosis_infrastructure.main,
    "r4_prognosis": future_prognosis_canonical.main,
    "r4_reconstruction_prognosis_infrastructure": reconstruction_prognosis_infrastructure.main,
    "r4_reconstruction_prognosis": reconstruction_prognosis_canonical.main,
    "r5_threshold_sensitivity": cross_system_sensitivity.main,
    "r5_cross_method": cross_method_canonical.main,
    "computational_efficiency": computational_efficiency.main,
    "r5_framework_generality": framework_generality.main,
    "tju_ncm_readiness": tju_ncm_readiness.main,
}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Canonical Phase-2 R3-R5 workflows")
    command.add_argument("--workflow", required=True, choices=tuple(WORKFLOWS))
    command.add_argument("--dataset")
    command.add_argument("--run-id")
    command.add_argument("--datasets", nargs="*", default=None)
    command.add_argument("--device")
    command.add_argument("--smoke", action="store_true")
    return command


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parser().parse_args()
    if args.workflow == "population_provenance":
        provenance_export.run(args.dataset or "mit", args.run_id, args.smoke)
        return
    if args.workflow == "r3_sensitivity":
        parameter_sensitivity.run(args.run_id, args.datasets, args.smoke)
        return
    if args.workflow == "r3_stress_sensitivity":
        parameter_sensitivity_stress.run(
            args.run_id, args.datasets, args.smoke, args.device or "cuda"
        )
        return
    if args.workflow == "r4_longitudinal":
        longitudinal_canonical.run(args.dataset or "mit", args.run_id, smoke=args.smoke)
        return
    if args.workflow == "r4_occlusion_schema":
        sliding_occlusion_canonical.smoke(args.run_id)
        return
    if args.workflow == "r4_occlusion":
        sliding_occlusion_canonical.run(
            args.run_id, device=args.device, smoke=args.smoke
        )
        return
    if args.workflow == "r4_prognosis_infrastructure":
        future_prognosis_infrastructure.smoke(args.dataset or "mit", args.run_id, smoke=args.smoke)
        return
    if args.workflow == "r4_prognosis":
        future_prognosis_canonical.run(args.dataset or "mit", args.run_id, args.smoke)
        return
    if args.workflow == "r4_reconstruction_prognosis_infrastructure":
        reconstruction_prognosis_infrastructure.smoke(args.run_id)
        return
    if args.workflow == "r4_reconstruction_prognosis":
        reconstruction_prognosis_canonical.run(args.run_id, args.smoke)
        return
    if args.workflow == "r5_threshold_sensitivity":
        cross_system_sensitivity.run(args.run_id, args.datasets, args.smoke)
        return
    if args.workflow == "r5_cross_method":
        cross_method_canonical.run(args.run_id, args.smoke)
        return
    if args.workflow == "computational_efficiency":
        computational_efficiency.run(
            args.run_id, "all", args.device or "cuda", smoke_flag=args.smoke
        )
        return
    if args.workflow == "r5_framework_generality":
        framework_generality.run(args.run_id)
        return
    if args.workflow == "tju_ncm_readiness":
        tju_ncm_readiness.run(args.run_id)
        return
    raise ValueError(args.workflow)


if __name__ == "__main__":
    main()
