"""C7: TJU-NCM canonical Pop/Ind upstream readiness check.

Determines whether the existing frozen reconstruction and downstream outputs
are sufficient to run PopulationIndividual without new inference. Only a
lightweight inventory/schema check is performed; no formal Pop/Ind inference
is started.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.utils.paths import REPOSITORY_ROOT, RESULTS_ROOT
from .provenance import canonical_run

REQUIRED_CONDITIONS = ("clean", "downsample", "random", "segment", "destructive")


def run(run_id: str | None = None) -> Path:
    run = canonical_run("tju_ncm_readiness", run_id)
    dataset = "tju_ncm_45c"
    inventory = {"dataset": dataset, "population_result_present": False,
                 "reconstruction_archives": {}, "predictor_checkpoints": {},
                 "data_splits": {}, "verdict": "INSUFFICIENT"}
    result_root = RESULTS_ROOT / "PopulationIndividual" / dataset
    summaries = list(result_root.glob("*/summary.json"))
    complete = [path for path in summaries
                if json.loads(path.read_text()).get("status") == "COMPLETE"]
    inventory["population_result_present"] = bool(complete)

    for seed in (2021, 2022, 2023):
        setting = f"Reconstruction_seed{seed}_batterydit_{dataset}_0"
        checkpoint = REPOSITORY_ROOT / "source_code/checkpoints/Reconstruction" / dataset / setting / "checkpoint.pth"
        inventory["reconstruction_archives"][seed] = {
            "checkpoint_exists": checkpoint.is_file(),
            "archives": {
                condition: (REPOSITORY_ROOT / "results/multi_dataset" / dataset / setting / f"reconstructions_{condition}.npz").is_file()
                for condition in REQUIRED_CONDITIONS
            },
        }
    for seed in (2021, 2022, 2023):
        setting = f"PredictDownstream_seed{seed}_batterydit_{dataset}_0"
        checkpoint = REPOSITORY_ROOT / "source_code/checkpoints/PredictDownstream" / dataset / setting / "best.pth"
        inventory["predictor_checkpoints"][seed] = {
            "checkpoint_exists": checkpoint.is_file(),
        }
    for split in ("train", "vali", "test"):
        root = REPOSITORY_ROOT / "source_code/data" / dataset / split
        inventory["data_splits"][split] = {
            "clean_exists": (root / "clean").is_dir(),
            "n_clean_files": len(list((root / "clean").rglob("*.npz"))) if (root / "clean").is_dir() else 0,
        }
    archives_ready = all(
        value["checkpoint_exists"] and all(value["archives"].values())
        for value in inventory["reconstruction_archives"].values())
    predictors_ready = all(value["checkpoint_exists"] for value in inventory["predictor_checkpoints"].values())
    data_ready = all(value["clean_exists"] and value["n_clean_files"] > 0 for value in inventory["data_splits"].values())
    if not complete and archives_ready and predictors_ready and data_ready:
        inventory["verdict"] = "READY_FOR_FORMAL_POP_INDIVIDUAL_RUN_NO_NEW_INFERENCE_REQUIRED"
    elif complete:
        inventory["verdict"] = "ALREADY_COMPLETE"
    else:
        inventory["verdict"] = "NEEDS_UPSTREAM"
    (run.provenance / "tju_ncm_readiness.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True, default=str) + "\n")
    run.write_manifest(config={"purpose": "upstream readiness check"}, input_paths={
        "population_result_root": str(result_root),
        "reconstruction_results_root": str(REPOSITORY_ROOT / "results/multi_dataset" / dataset),
        "data_root": str(REPOSITORY_ROOT / "source_code/data" / dataset)})
    print(f"TJU-NCM readiness -> {run.root} ({inventory['verdict']})")
    return run.root


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id")
    return command


def main() -> None:
    args = parser().parse_args()
    run(args.run_id)


if __name__ == "__main__":
    main()
