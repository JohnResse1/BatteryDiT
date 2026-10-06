"""R5-A seven-cohort framework generality provenance workflow.

This workflow consumes completed dataset-specific reconstruction and frozen
downstream-prediction runs.  It materialises the framework-level numbers and
their split/checkpoint/OCV/normalisation provenance into the canonical
``raw/analysis/plotting/provenance`` layout.  It never trains or reconstructs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import multi_dataset
from analysis.utils.results import FORMAL_CONDITIONS, condition_matrix
from .provenance import canonical_run


CONDITIONS = tuple(multi_dataset.DEGRADED_CONDITIONS)
TARGETS = {
    "sdu": "second_life_capacity_normalized",
    "mit": "nominal_capacity_soh",
    "hust": "nominal_capacity_soh",
    "mich": "nominal_capacity_soh",
    "stanford_2": "nominal_capacity_soh",
    "tju_nca_45c": "nominal_capacity_soh",
    "tju_ncm_45c": "nominal_capacity_soh",
}
SOURCE_PROTOCOL_FILES = (
    "source_code/data_preprocessing/datasets.py",
    "source_code/data_preprocessing/README.md",
    "source_code/exp/multi_dataset.py",
    "source_code/models/ocv/extraction.py",
    "source_code/models/ocv/README.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_checkpoint(run) -> Path:
    # Keep the path construction explicit because the result directory may be
    # under results/multi_dataset while checkpoints always live in source_code.
    root = Path(__file__).resolve().parents[2]
    args = run.arguments
    task = str(args.get("task_name", ""))
    group = "Reconstruction" if task == "Reconstruction" else "PredictDownstream"
    filename = "checkpoint.pth" if group == "Reconstruction" else "best.pth"
    return root / "source_code" / "checkpoints" / group / run.dataset / run.setting / filename


def _split_manifest(dataset: str) -> list[dict]:
    # BatteryDataset is the same loader used by the training/evaluation code;
    # no split is inferred from a table of metrics.
    from data_provider.data_loader import BatteryDataset

    rows = []
    for split in ("train", "vali", "test"):
        data = BatteryDataset(dataset, split=split, condition="clean")
        keys = [str(path.relative_to(data.root)) for path in data.paths]
        batteries = sorted({key.rsplit("/", 1)[0] for key in keys})
        rows.append({
            "dataset": dataset,
            "dataset_label": multi_dataset.DATASETS[dataset],
            "split": split,
            "target": TARGETS[dataset],
            "n_trajectories": len(keys),
            "n_batteries": len(batteries),
            "battery_ids": json.dumps(batteries),
        })
    return rows


def _run_inventory(dataset: str, reconstruction_runs, prediction_runs) -> list[dict]:
    root = Path(__file__).resolve().parents[2]
    rows = []
    for stage, runs in (("reconstruction", reconstruction_runs), ("downstream", prediction_runs)):
        for formal_run in runs:
            config_path = formal_run.directory / "config.json"
            metrics_path = formal_run.directory / "test_metrics.json"
            checkpoint = _resolve_checkpoint(formal_run)
            args = formal_run.arguments
            ocv_rel = str(args.get("ocv_table_path", ""))
            ocv_path = root / "source_code" / ocv_rel if ocv_rel else Path("")
            rows.append({
                "dataset": dataset,
                "dataset_label": multi_dataset.DATASETS[dataset],
                "stage": stage,
                "seed": formal_run.seed,
                "result_dir": str(formal_run.directory),
                "config_path": str(config_path),
                "config_sha256": _sha256(config_path) if config_path.is_file() else "MISSING",
                "metrics_path": str(metrics_path),
                "metrics_sha256": _sha256(metrics_path) if metrics_path.is_file() else "MISSING",
                "checkpoint_path": str(checkpoint),
                "checkpoint_exists": checkpoint.is_file(),
                "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else "MISSING",
                "ocv_table_path": str(ocv_path),
                "ocv_table_exists": ocv_path.is_file(),
                "ocv_table_sha256": _sha256(ocv_path) if ocv_path.is_file() else "MISSING",
                "training_scope": "dataset-specific independent training",
                "normalization_policy": (
                    "preprocessing-defined SOH target; no post-split capacity/min-max normalization; "
                    "physical voltage/current bounds are recorded in the run config"
                ),
                "weak_prior_policy": "dataset-specific OCV table built from training data; no test/validation OCV source",
            })
    return rows


def run(run_id: str | None = None) -> Path:
    run = canonical_run("r5_framework_generality", run_id)
    summary_rows, raw_rows, corrupted_rows = [], [], []
    split_rows, inventory_rows = [], []
    representative_payload = {}
    representative_metadata = {}
    input_paths: dict[str, str] = {}
    checkpoint_ids: dict[str, str] = {}

    columns = [FORMAL_CONDITIONS.index(name) for name in CONDITIONS]
    for dataset, label in multi_dataset.DATASETS.items():
        reconstruction_runs, prediction_runs = multi_dataset.runs(dataset)
        split_rows.extend(_split_manifest(dataset))
        inventory_rows.extend(_run_inventory(dataset, reconstruction_runs, prediction_runs))
        for formal_run in reconstruction_runs + prediction_runs:
            input_paths[f"{dataset}_{formal_run.setting}_config"] = str(formal_run.directory / "config.json")
        for row in inventory_rows:
            if row["dataset"] == dataset and row["checkpoint_exists"]:
                checkpoint_ids[f"{dataset}_{row['stage']}_{row['seed']}"] = row["checkpoint_sha256"]

        recon_values = condition_matrix(reconstruction_runs, "battery_balanced_rmse")[:, columns]
        prediction_values = condition_matrix(prediction_runs, "soh_rmse")
        clean_index = FORMAL_CONDITIONS.index("clean")
        deltas = prediction_values[:, columns] - prediction_values[:, [clean_index]]
        for index, condition in enumerate(CONDITIONS):
            for seed_index, formal_run in enumerate(reconstruction_runs):
                raw_rows.append({
                    "dataset": label, "dataset_key": dataset, "target": TARGETS[dataset],
                    "condition": condition, "seed": formal_run.seed,
                    "reconstruction_battery_balanced_rmse": float(recon_values[seed_index, index]),
                    "downstream_soh_rmse": float(prediction_values[seed_index, columns[index]]),
                    "downstream_delta_vs_clean": float(deltas[seed_index, index]),
                })
            summary_rows.append({
                "dataset": label, "dataset_key": dataset, "target": TARGETS[dataset],
                "chemistry": multi_dataset.CHEMISTRIES[dataset], "condition": condition,
                "reconstruction_mean": float(recon_values[:, index].mean()),
                "reconstruction_sd": float(recon_values[:, index].std(ddof=1)),
                "downstream_delta_mean": float(deltas[:, index].mean()),
                "downstream_delta_sd": float(deltas[:, index].std(ddof=1)),
                "n_model_seeds": int(recon_values.shape[0]),
                "training_scope": "dataset-specific independent training",
            })
            corrupted_values = []
            for formal_run in reconstruction_runs:
                archive_path = formal_run.directory / f"reconstructions_{condition}.npz"
                with np.load(archive_path, allow_pickle=False) as archive:
                    clean = np.asarray(archive["clean_voltage"], dtype=np.float32)
                    corrupted = np.asarray(archive["input_voltage"], dtype=np.float32)
                corrupted_values.append(float(np.sqrt(np.mean((corrupted - clean) ** 2, axis=(1, 2))).mean()))
            corrupted_rows.append({
                "dataset": label, "dataset_key": dataset, "target": TARGETS[dataset],
                "condition": condition,
                "corrupted_input_rmse_mean": float(np.mean(corrupted_values)),
                "corrupted_input_rmse_sd": float(np.std(corrupted_values, ddof=1)),
                "n_model_seeds": len(corrupted_values),
            })

        item, metadata = multi_dataset.representative(reconstruction_runs)
        representative_payload.update({f"{dataset}_{name}": value for name, value in item.items()})
        representative_metadata[dataset] = metadata

    raw_frame = pd.DataFrame(raw_rows)
    summary_frame = pd.DataFrame(summary_rows)
    corrupted_frame = pd.DataFrame(corrupted_rows)
    split_frame = pd.DataFrame(split_rows)
    inventory_frame = pd.DataFrame(inventory_rows)

    run.write_table("raw", "framework_formal_metrics.csv", raw_frame)
    run.write_table("raw", "framework_corrupted_input_rmse.csv", corrupted_frame)
    np.savez_compressed(run.raw / "framework_representatives.npz", **representative_payload)
    (run.raw / "framework_representatives.json").write_text(
        json.dumps(representative_metadata, indent=2) + "\n"
    )
    run.write_table("analysis", "framework_generality_summary.csv", summary_frame)
    run.write_table("analysis", "framework_corrupted_input_rmse.csv", corrupted_frame)
    run.write_table("plotting", "framework_generality.csv", summary_frame)
    run.write_table("provenance", "cohort_split_manifest.csv", split_frame)
    run.write_table("provenance", "formal_run_inventory.csv", inventory_frame)

    source_files = []
    for relative in SOURCE_PROTOCOL_FILES:
        path = Path(__file__).resolve().parents[2] / relative
        source_files.append({
            "path": str(path), "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else "MISSING",
        })
    run.write_json("protocol_provenance.json", {
        "framework_scope": list(multi_dataset.DATASETS),
        "population_individual_scope": ["mit", "hust", "mich", "stanford_2", "tju_nca_45c", "tju_ncm_45c"],
        "sdu_exclusion": "SDU target is dataset-defined second-life capacity normalization",
        "training_scope": "BatteryDiT reconstruction and downstream predictor independently trained per dataset and seed",
        "normalization_policy": "preprocessing-defined targets; no additional post-split capacity/min-max normalization",
        "weak_prior_policy": "dataset-specific train-only OCV tables selected by formal source_code/models/ocv/extraction.py",
        "source_protocol_files": source_files,
    })
    config = {
        "datasets": list(multi_dataset.DATASETS),
        "conditions": list(CONDITIONS),
        "formal_seeds": [2021, 2022, 2023],
        "target_policy": TARGETS,
        "representative_rule": "analysis.multi_dataset.representative",
        "training_scope": "dataset-specific independent reconstruction and downstream predictor training",
    }
    run.write_json("framework_generality_config.json", config)
    run.write_manifest(
        config=config,
        input_paths=input_paths,
        checkpoint_ids=checkpoint_ids,
        seeds=[2021, 2022, 2023],
        split_manifest=json.dumps(
            {dataset: [row for row in split_rows if row["dataset"] == dataset]
             for dataset in multi_dataset.DATASETS},
            sort_keys=True,
        ),
    )
    print(f"R5 framework generality outputs -> {run.root}")
    return run.root


def parser():
    import argparse

    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id")
    return command


def main() -> None:
    args = parser().parse_args()
    run(args.run_id)


if __name__ == "__main__":
    main()
