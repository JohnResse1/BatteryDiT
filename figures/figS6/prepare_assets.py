"""Materialise the existing R3 battery-level data for Supplementary Fig. S6.

This script only selects the already-computed MIT battery-level decomposition
rows at L1, L3, and L5 and copies the two retained metrics, E_pop and E_ind.
It performs no model inference, aggregation, normalization, bootstrap, or
phenotype assignment.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
CONDITION_ORDER = [
    ("downsample", "Downsampling"),
    ("random_missing", "Random-point missing"),
    ("segment_missing", "Segment missing"),
    ("destructive", "Destructive noise"),
    ("hybrid", "Hybrid"),
]
SEVERITY_ORDER = ["L1", "L2", "L3", "L4", "L5"]
DIST_LEVELS = ["L1", "L3", "L5"]
MIT_RUN = (
    REPO / "results" / "PopulationIndividual" / "mit"
    / "PopulationIndividual_seed2021_batterydit_trb_mit_0"
)


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO))


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    source = MIT_RUN / "battery_level_decomposition.csv"
    if not source.is_file():
        raise FileNotFoundError(source)

    data = pd.read_csv(source)
    required = {"cell", "condition", "severity", "severity_index", "E_pop", "E_ind", "n_cycles"}
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f"formal battery table is missing: {sorted(missing)}")

    data = data[
        data.condition.isin([name for name, _label in CONDITION_ORDER])
        & data.severity.isin(DIST_LEVELS)
    ].copy()
    data = data.rename(columns={"cell": "battery_id"})
    labels = dict(CONDITION_ORDER)
    data["condition_label"] = data.condition.map(labels)
    data["severity_index"] = data.severity.map(
        {level: index + 1 for index, level in enumerate(SEVERITY_ORDER)}
    )
    data = data.sort_values(
        ["severity_index", "condition", "battery_id"], kind="stable"
    )
    columns = [
        "battery_id", "condition", "condition_label", "severity", "severity_index",
        "E_pop", "E_ind", "n_cycles",
    ]
    data = data[columns]
    expected = len(CONDITION_ORDER) * len(DIST_LEVELS) * data.battery_id.nunique()
    if len(data) != expected or data.battery_id.nunique() != 21:
        raise RuntimeError(
            f"battery-level asset does not have complete 21 x 5 x 3 coverage: {len(data)} rows"
        )
    data.to_csv(DATA / "s6_distribution.csv", index=False)

    provenance = {
        "figure": "Supplementary Fig. S6",
        "title": "Battery-level Population and Individual recovery under progressive observation degradation",
        "scope": "MIT test split; 21 batteries; five corruption topologies; L1, L3, and L5",
        "source": _relative(source),
        "aggregation": "existing battery-level decomposition values; no new aggregation",
        "point_unit": "battery x corruption x severity",
        "metrics": {
            "E_pop": "formal battery-level Population reconstruction error",
            "E_ind": "formal battery-level Individual reconstruction error",
        },
        "severity_levels": DIST_LEVELS,
        "no_new_analysis": True,
        "no_normalization": True,
        "no_phenotype_classification": True,
        "output_table": "figures/figS6/data/s6_distribution.csv",
    }
    (DATA / "s6_asset_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Fig. S6 plotting asset written -> {DATA / 's6_distribution.csv'}")


if __name__ == "__main__":
    main()
