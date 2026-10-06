"""R4-10/R4-11 infrastructure using frozen MIT L5 stress waveforms.

This is a smoke/schema workflow only. It maps the frozen MIT L5 waveform
assets into the canonical representation source and exports the
required raw schema. It does not compute correlations, select batteries, or
perform a formal full prognosis run before the future target is confirmed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import _certified as C
from .provenance import canonical_run
from analysis.utils.paths import REPOSITORY_ROOT

L5_ASSET_ROOT = REPOSITORY_ROOT / "analysis/assets/reconstruction/mit_l5"
ASSET_CONDITIONS = {
    "downsample": "downsample",
    "random_missing": "random",
    "part_missing": "segment",
    "destructive": "destructive",
    "hybrid": "hybrid",
}
SEVERITY = "L5"


def smoke(run_id: str | None = None) -> Path:
    run = canonical_run("r4_reconstruction_prognosis_infrastructure", run_id)
    inventory = []
    schema_rows = []
    for asset_condition, canonical_condition in ASSET_CONDITIONS.items():
        waveform_path = L5_ASSET_ROOT / asset_condition / "level_5" / "waveforms.npz"
        complete_path = L5_ASSET_ROOT / asset_condition / "level_5" / "complete.json"
        if not waveform_path.is_file():
            inventory.append({
                "asset_condition": asset_condition, "canonical_condition": canonical_condition,
                "path": str(waveform_path.relative_to(REPOSITORY_ROOT)), "exists": False})
            continue
        with np.load(waveform_path, allow_pickle=False) as archive:
            shape = tuple(archive["V_reconstructed"].shape)
        complete = json.loads(complete_path.read_text()) if complete_path.is_file() else {}
        inventory.append({
            "asset_condition": asset_condition, "canonical_condition": canonical_condition,
            "path": str(waveform_path.relative_to(REPOSITORY_ROOT)), "exists": True,
            "shape": list(shape), "asset_metadata": complete})
        for representation_source in ("clean", "reconstructed"):
            schema_rows.append({
                "battery_id": "<identity_from_clean_test>",
                "cycle_id": -1,
                "corruption": canonical_condition,
                "severity": SEVERITY,
                "representation_source": representation_source,
                "E_pop": np.nan,
                "E_ind": np.nan,
                "C_ind": np.nan,
                "future_target": np.nan,
                "population_prediction": np.nan,
                "individual_prediction": np.nan,
            })
    schema = pd.DataFrame(schema_rows)
    run.write_table("raw", "reconstruction_prognosis_schema.csv.gz", schema)
    run.write_json("l5_asset_inventory.json", inventory)
    config = {
        "status": "BLOCKED_FOR_SCIENTIFIC_CONFIRMATION",
        "reason": "future prognosis target requires human confirmation",
        "l5_asset_root": str(L5_ASSET_ROOT.relative_to(REPOSITORY_ROOT)),
        "corruptions": list(ASSET_CONDITIONS.values()),
        "severity": SEVERITY,
    }
    run.write_json("reconstruction_prognosis_config.json", config)
    run.write_manifest(config=config, input_paths={
        item["canonical_condition"]: item["path"] for item in inventory if item["exists"]})
    print(f"R4 reconstruction prognosis infrastructure smoke -> {run.root}")
    return run.root


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id")
    return command


def main() -> None:
    args = parser().parse_args()
    smoke(args.run_id)


if __name__ == "__main__":
    main()
