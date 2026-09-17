"""Shared train-only SOC--OCV extraction used during dataset preparation.

Dataset adapters provide training cells as cycles containing current and
voltage. This module selects one cycle-10 source and writes the arrays used by
the physics operator.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OCVPriorPolicy:
    target_cycle: int = 10
    ranking_cycle_count: int = 5
    soc_points: int = 1000
    minimum_charge_points: int = 20
    positive_current_fraction: float = 0.01
    ranking_current_quantile: float = 0.90
    minimum_voltage_span_fraction: float = 0.90


@dataclass(frozen=True)
class OCVCurve:
    cycle: int
    current: np.ndarray
    voltage: np.ndarray


@dataclass(frozen=True)
class OCVCell:
    name: str
    curves: tuple[OCVCurve, ...]
    capacity_ah: float | None = None


@dataclass(frozen=True)
class OCVSelection:
    cell: str
    curve: OCVCurve
    rate_proxy: float
    voltage_span: float
    maximum_voltage_span: float


def _charge_segment(current: np.ndarray, policy: OCVPriorPolicy) -> tuple[np.ndarray, np.ndarray, float]:
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    finite = np.isfinite(current)
    if int(finite.sum()) < policy.minimum_charge_points:
        raise ValueError("too few finite current samples")
    values = current[finite]
    orientation = 1.0 if np.clip(values, 0, None).sum() >= np.clip(-values, 0, None).sum() else -1.0
    oriented = current * orientation
    maximum = float(np.max(oriented[finite]))
    if maximum <= 0.0:
        raise ValueError("no charging current")
    threshold = max(1e-8, policy.positive_current_fraction * maximum)
    charge = finite & (oriented >= threshold)
    if int(charge.sum()) < policy.minimum_charge_points:
        raise ValueError("too few charging samples")
    return oriented, charge, float(np.quantile(oriented[charge], policy.ranking_current_quantile))


def _valid_cycles(cell: OCVCell, policy: OCVPriorPolicy) -> list[tuple[OCVCurve, float]]:
    valid: list[tuple[OCVCurve, float]] = []
    for curve in cell.curves:
        try:
            current = np.asarray(curve.current).reshape(-1)
            voltage = np.asarray(curve.voltage).reshape(-1)
            if current.size != voltage.size or not np.isfinite(voltage).all():
                continue
            _, _, mean_current = _charge_segment(current, policy)
            valid.append((curve, mean_current))
        except ValueError:
            continue
    return sorted(valid, key=lambda item: (abs(item[0].cycle - policy.target_cycle), item[0].cycle))


def _select_ocv_source(cells: Iterable[OCVCell], policy: OCVPriorPolicy) -> OCVSelection:
    """Choose a full-coverage, low-rate training curve nearest cycle 10.

    The upper current quantile represents the main charging stage without
    allowing a long low-current CV tail to make a fast-charge cell appear
    slow.  Voltage coverage is checked relative to the widest candidate so a
    truncated curve cannot silently define SOC=0 at a high voltage.
    """
    cells = tuple(cells)
    capacities = {cell.capacity_ah is not None for cell in cells}
    if len(capacities) > 1:
        raise ValueError("provide capacity_ah for every candidate cell or for none")
    candidates: list[tuple[float, str, OCVCurve, float]] = []
    for cell in cells:
        ranked = _valid_cycles(cell, policy)
        if len(ranked) < policy.ranking_cycle_count:
            continue
        near_target = ranked[:policy.ranking_cycle_count]
        current_score = float(np.mean([value for _, value in near_target]))
        speed = current_score / float(cell.capacity_ah) if cell.capacity_ah is not None else current_score
        source = near_target[0][0]
        source_current = np.asarray(source.current, dtype=np.float64).reshape(-1)
        source_voltage = np.asarray(source.voltage, dtype=np.float64).reshape(-1)
        _, source_charge, _ = _charge_segment(source_current, policy)
        voltage_span = float(np.ptp(source_voltage[source_charge]))
        candidates.append((speed, cell.name, source, voltage_span))
    if not candidates:
        raise RuntimeError("no training cell has enough valid charge cycles near cycle 10")
    maximum_span = max(item[3] for item in candidates)
    minimum_span = policy.minimum_voltage_span_fraction * maximum_span
    eligible = [item for item in candidates if item[3] >= minimum_span]
    if not eligible:
        raise RuntimeError("no OCV candidate satisfies the voltage-coverage policy")
    speed, cell, curve, voltage_span = min(eligible, key=lambda item: (item[0], item[1]))
    return OCVSelection(cell, curve, speed, voltage_span, maximum_span)


def select_ocv_source(cells: Iterable[OCVCell], policy: OCVPriorPolicy = OCVPriorPolicy()) -> OCVCurve:
    """Return the selected train-only source curve."""
    return _select_ocv_source(cells, policy).curve


def _extract_curve(curve: OCVCurve, policy: OCVPriorPolicy) -> tuple[np.ndarray, np.ndarray]:
    """Map one selected charging curve to the fixed SOC grid."""
    current = np.asarray(curve.current, dtype=np.float64).reshape(-1)
    source_voltage = np.asarray(curve.voltage, dtype=np.float64).reshape(-1)
    oriented, charge, _ = _charge_segment(current, policy)
    charge_current, charge_voltage = oriented[charge], source_voltage[charge]
    cumulative = np.zeros(charge_current.size, dtype=np.float64)
    cumulative[1:] = np.cumsum(0.5 * (charge_current[:-1] + charge_current[1:]))
    if cumulative[-1] <= 0.0:
        raise ValueError("selected curve has non-positive integrated charge")
    source_soc, unique = np.unique(cumulative / cumulative[-1], return_index=True)
    if source_soc.size < policy.minimum_charge_points:
        raise ValueError("selected curve has too few unique SOC points")
    soc = np.linspace(0.0, 1.0, policy.soc_points, dtype=np.float32)
    voltage = np.maximum.accumulate(np.interp(soc, source_soc, charge_voltage[unique])).astype(np.float32)
    return soc, voltage


def extract_ocv_prior(cells: Iterable[OCVCell], policy: OCVPriorPolicy = OCVPriorPolicy()) -> tuple[np.ndarray, np.ndarray]:
    """Extract one monotone OCV curve from train cells using the shared protocol."""
    return _extract_curve(select_ocv_source(cells, policy), policy)


def training_cells_from_directory(
    train_root: str | Path,
    policy: OCVPriorPolicy = OCVPriorPolicy(),
) -> tuple[OCVCell, ...]:
    """Adapter for ``train/<cell>/<cycle>.npz`` files containing ``I`` and ``V``."""
    root = Path(train_root)
    cells: list[OCVCell] = []
    for cell_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        curves: list[OCVCurve] = []
        paths = sorted(
            cell_dir.glob("*.npz"),
            key=lambda path: (abs(int(path.stem) - policy.target_cycle), int(path.stem)),
        )
        # Continue past malformed files until the requested number of valid
        # cycles has been collected.
        for path in paths:
            try:
                cycle = int(path.stem)
                with np.load(path) as data:
                    curves.append(OCVCurve(cycle, data["I"], data["V"]))
            except (KeyError, ValueError):
                continue
            if len(curves) == policy.ranking_cycle_count:
                break
        cells.append(OCVCell(cell_dir.name, tuple(curves)))
    return tuple(cells)


def save_training_ocv(train_root: str | Path, output_path: str | Path, policy: OCVPriorPolicy = OCVPriorPolicy()) -> Path:
    """Write the fixed table and an auditable sidecar describing its source."""
    cells = training_cells_from_directory(train_root, policy)
    selection = _select_ocv_source(cells, policy)
    soc, voltage = _extract_curve(selection.curve, policy)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, soc=soc, voltage=voltage)
    metadata = {
        "schema_version": 1,
        "source": "train/clean only",
        "train_root": str(Path(train_root)),
        "selected_cell": selection.cell,
        "selected_cycle": int(selection.curve.cycle),
        "rate_proxy": float(selection.rate_proxy),
        "rate_proxy_definition": (
            f"mean charging-current q{policy.ranking_current_quantile:g} over the "
            f"nearest {policy.ranking_cycle_count} valid cycles"
        ),
        "selected_voltage_min": float(voltage.min()),
        "selected_voltage_max": float(voltage.max()),
        "selected_voltage_span": float(selection.voltage_span),
        "maximum_candidate_voltage_span": float(selection.maximum_voltage_span),
        "policy": asdict(policy),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    command = argparse.ArgumentParser(description="Build one fixed train-only charging-voltage prior")
    command.add_argument("--train-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    options = command.parse_args()
    path = save_training_ocv(options.train_root, options.output)
    print(path)
    print(path.with_suffix(".json"))


if __name__ == "__main__":
    main()
