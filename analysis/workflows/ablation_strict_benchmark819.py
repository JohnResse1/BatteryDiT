"""Run the retained MIT benchmark-819 component ablation.

This workflow is deliberately self-contained in the rebuilt repository.  The
Full row reuses the rebuilt formal BatteryDiT checkpoints; the other three
rows are trained independently with the same reconstruction protocol.  Test
views, seed-matched TRB predictors, and the Population--Individual manifold
are all fixed inputs.  Only artifacts under this rebuilt repository are used.

The four rows are:

``all``
    Full BatteryDiT.
``sensor+mask``
    Without the weak charging prior; the confidence channel is retained.
``sensor+po``
    Without the physics-consistency gate; sensor and prior are equally fused.
``sensor``
    Sensor raw; both prior and gate are removed.

The formal commands are intended to be launched by
``analysis/formal_runs/run_ablation_strict_benchmark819_gpu.sbatch``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.interpolate import BSpline
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CODE_ROOT = REPOSITORY_ROOT / "source_code"
if str(SOURCE_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE_ROOT))

from data_provider.data_loader import BatteryDataset, PairedBatteryDataset  # noqa: E402
from models.BatteryDiT_ablation_aligned import Model as AblationBatteryDiT  # noqa: E402
from models.checkpoint_loading import (  # noqa: E402
    load_reconstruction_checkpoint,
    state_dict_from_checkpoint,
)
from models.predictors.trb.model import Model as TRB  # noqa: E402
from exp import population_individual as certified_decomposition  # noqa: E402


SEEDS = (2021, 2022, 2023)
CONDITIONS = (
    ("downsample", "Downsampling"),
    ("random", "Random missing"),
    ("segment", "Segment missing"),
    ("destructive", "Destructive noise"),
)
VARIANTS = (
    {
        "slug": "full",
        "name": "Full BatteryDiT",
        "mode": "all",
        "prior": "ON",
        "gate": "ON",
        "independently_trained": False,
    },
    {
        "slug": "without_po",
        "name": "w/o weak charging prior",
        "mode": "sensor+mask",
        "prior": "OFF",
        "gate": "ON (confidence channel retained)",
        "independently_trained": True,
    },
    {
        "slug": "without_gate",
        "name": "w/o physics-consistency gate",
        "mode": "sensor+po",
        "prior": "ON",
        "gate": "OFF (equal sensor/PO fusion)",
        "independently_trained": True,
    },
    {
        "slug": "sensor_raw",
        "name": "Sensor raw (w/o prior and gate)",
        "mode": "sensor",
        "prior": "OFF",
        "gate": "OFF",
        "independently_trained": True,
    },
)
PRIMARY_METRICS = ("waveform_rmse", "soh_rmse", "E_pop", "E_ind")
SECONDARY_METRICS = ("Q", "abs_Q_minus_1", "C_ind")
ALL_METRICS = PRIMARY_METRICS + SECONDARY_METRICS
# ``soh_rmse`` is a condition-level aggregate, computed from
# ``soh_squared_error`` in ``summarize_condition``.  It is therefore not a
# per-cycle column in the raw evaluation table and must not be requested by
# the per-cycle finite-value check below.
ROW_FINITE_METRICS = (
    "waveform_rmse",
    "soh_squared_error",
    "E_pop",
    "E_ind",
    "Q",
    "abs_Q_minus_1",
    "C_ind",
)

TRAIN_EPOCHS = 200
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
GRADIENT_CLIP = 1.0
EVAL_BATCH_SIZE = 128
NUM_WORKERS = int(os.environ.get("ABLATION_NUM_WORKERS", "4"))
EXPECTED_TEST_ROWS = 16729
EXPECTED_ELIGIBLE_ROWS = 16624
V_MIN = 2.0
V_MAX = 3.6
SAMPLING_STEPS = 1

FULL_CHECKPOINTS = SOURCE_CODE_ROOT / "checkpoints" / "Reconstruction" / "mit"
TRB_CHECKPOINTS = SOURCE_CODE_ROOT / "checkpoints" / "PredictDownstream" / "mit"
MANIFOLD_ROOT = (
    REPOSITORY_ROOT
    / "analysis"
    / "outputs"
    / "canonical"
    / "population_individual_provenance"
    / "20260926T070134Z"
    / "provenance"
)
MANIFOLD_PARAMETERS = MANIFOLD_ROOT / "population_manifold_parameters.npz"
MANIFOLD_CONFIG = MANIFOLD_ROOT / "population_manifold_config.json"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def run_root(value: str | Path) -> Path:
    root = Path(value)
    if not root.is_absolute():
        root = REPOSITORY_ROOT / root
    return root.resolve()


def output_dirs(root: Path) -> dict[str, Path]:
    directories = {
        "root": root,
        "raw": root / "raw",
        "analysis": root / "analysis",
        "plotting": root / "plotting",
        "provenance": root / "provenance",
        "checkpoints": root / "checkpoints",
        "task_results": root / "task_results",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def variant_by_slug(slug: str) -> dict[str, Any]:
    for variant in VARIANTS:
        if variant["slug"] == slug:
            return variant
    raise KeyError(f"unknown variant {slug!r}")


def decode_train_task(task_index: int) -> tuple[dict[str, Any], int]:
    total = len(VARIANTS[1:]) * len(SEEDS)
    if not 0 <= task_index < total:
        raise ValueError(f"training task index must be in [0, {total - 1}]")
    variant = VARIANTS[1 + task_index // len(SEEDS)]
    seed = SEEDS[task_index % len(SEEDS)]
    return variant, seed


def decode_eval_task(task_index: int) -> tuple[dict[str, Any], int]:
    total = len(VARIANTS) * len(SEEDS)
    if not 0 <= task_index < total:
        raise ValueError(f"evaluation task index must be in [0, {total - 1}]")
    variant = VARIANTS[task_index // len(SEEDS)]
    seed = SEEDS[task_index % len(SEEDS)]
    return variant, seed


def checkpoint_for(root: Path, variant: dict[str, Any], seed: int) -> Path:
    if variant["slug"] == "full":
        return (
            FULL_CHECKPOINTS
            / f"Reconstruction_seed{seed}_batterydit_mit_0"
            / "checkpoint.pth"
        )
    return (
        root
        / "checkpoints"
        / "reconstruction"
        / variant["slug"]
        / f"seed{seed}"
        / "checkpoint.pth"
    )


def predictor_for(seed: int) -> Path:
    return (
        TRB_CHECKPOINTS
        / f"PredictDownstream_seed{seed}_batterydit_trb_mit_0"
        / "best.pth"
    )


def model_args() -> SimpleNamespace:
    return SimpleNamespace(
        data="mit",
        v_min=V_MIN,
        v_max=V_MAX,
        i_min=0.0,
        i_max=6.2,
        po_type="MIT",
        ocv_table_path=str(SOURCE_CODE_ROOT / "models" / "ocv" / "mit.npz"),
        seq_len=1000,
        hidden_size=256,
        depth=6,
        num_heads=4,
        timestep_embed_dim=256,
        timestep_freq_embed_size=64,
        soc_points=1000,
        voltage_points=1000,
        dropout=0.1,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def voltage_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    voltage_range: float,
) -> torch.Tensor:
    return F.mse_loss(
        (prediction[:, 1:] - prediction[:, :-1]) * voltage_range,
        (target[:, 1:] - target[:, :-1]) * voltage_range,
    )


def load_reconstruction(
    path: Path,
    device: torch.device,
) -> AblationBatteryDiT:
    if not path.is_file():
        raise FileNotFoundError(f"reconstruction checkpoint not found: {path}")
    model = AblationBatteryDiT(model_args()).to(device)
    load_reconstruction_checkpoint(model, path, map_location=device, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_predictor(path: Path, device: torch.device) -> TRB:
    if not path.is_file():
        raise FileNotFoundError(f"TRB checkpoint not found: {path}")
    predictor = TRB(model_args(), dropout=0.1).to(device)
    payload = torch.load(path, map_location=device)
    state = state_dict_from_checkpoint(payload) if isinstance(payload, dict) else payload
    predictor.load_state_dict(state, strict=True)
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    return predictor


class FrozenPopulation:
    """Read-only reconstruction of the serialized train-only manifold."""

    def __init__(self, parameter_path: Path, config_path: Path):
        if not parameter_path.is_file() or not config_path.is_file():
            raise FileNotFoundError("frozen Population manifold assets are incomplete")
        with np.load(parameter_path, allow_pickle=True) as archive:
            full_knots = np.asarray(archive["spline_knots"], dtype=np.float64).reshape(-1)
            self.degree = int(np.asarray(archive["spline_degree"]).item())
            self.coefficients = np.asarray(archive["ridge_coefficients"], dtype=np.float32)
            self.intercept = np.asarray(archive["ridge_intercept"], dtype=np.float32)
            self.s_ind = float(np.asarray(archive["S_ind"]).item())
            self.train_soh_min = float(np.asarray(archive["train_soh_min"]).item())
            self.train_soh_max = float(np.asarray(archive["train_soh_max"]).item())
            self.n_knots = int(np.asarray(archive["spline_n_knots"]).item())
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.grid_size = int(config.get("projection_grid", 500))
        self.alpha = float(config["manifold"]["alpha"])
        self.status = str(config["manifold"].get("manifold_status", "UNKNOWN"))
        if self.n_knots != 6 or self.degree != 3:
            raise RuntimeError("unexpected frozen MIT manifold basis configuration")
        if self.coefficients.shape != (1000, 7) or self.intercept.shape != (1000,):
            raise RuntimeError("unexpected frozen MIT manifold coefficient shape")
        if not np.isfinite(self.s_ind) or self.s_ind <= 0:
            raise RuntimeError("invalid frozen Individual scale")
        if not np.isfinite(self.coefficients).all() or not np.isfinite(self.intercept).all():
            raise RuntimeError("non-finite frozen Population manifold parameters")

        self.full_knots = full_knots
        n_splines = len(full_knots) - 2 * self.degree + self.degree - 1
        if n_splines != 8:
            raise RuntimeError(f"unexpected frozen spline basis size: {n_splines}")
        self._basis = BSpline(
            full_knots,
            np.eye(n_splines, dtype=np.float64),
            self.degree,
            extrapolate=False,
        )
        self.x_min = float(full_knots[self.degree])
        self.x_max = float(full_knots[-self.degree - 1])
        self.grid_h = np.linspace(
            self.train_soh_min, self.train_soh_max, self.grid_size, dtype=np.float32
        )
        self.grid_population = self.predict(self.grid_h)

    def design(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        design = np.zeros((len(values), 8), dtype=np.float64)
        inside = (values >= self.x_min) & (values <= self.x_max)
        if np.any(inside):
            design[inside] = self._basis(values[inside])
        lower = values < self.x_min
        if np.any(lower):
            design[lower, : self.degree] = self._basis(self.x_min)[: self.degree]
        upper = values > self.x_max
        if np.any(upper):
            design[upper, -self.degree :] = self._basis(self.x_max)[-self.degree :]
        # The serialized Ridge coefficient matrix came from include_bias=False.
        return design[:, :-1]

    def predict(self, soh: np.ndarray) -> np.ndarray:
        design = self.design(np.asarray(soh))
        prediction = design @ self.coefficients.T.astype(np.float64)
        prediction += self.intercept.astype(np.float64)
        return prediction.astype(np.float32)[:, :, None]


def load_clean_decomposition(frozen: FrozenPopulation) -> dict[str, Any]:
    """Build clean R arrays from clean data and the frozen manifold only."""
    test = certified_decomposition.load_clean("mit", "test")
    residual, references, keep = certified_decomposition.residualize(
        test, require_five=False
    )
    finite = np.isfinite(residual).all(axis=(1, 2))
    eligible = np.asarray(keep, dtype=bool) & finite
    if len(test["voltage"]) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"MIT test clean rows changed: {len(test['voltage'])} != {EXPECTED_TEST_ROWS}"
        )
    if int(eligible.sum()) != EXPECTED_ELIGIBLE_ROWS:
        raise RuntimeError(
            f"MIT eligible rows changed: {int(eligible.sum())} != {EXPECTED_ELIGIBLE_ROWS}"
        )
    cells = np.asarray(test["cells"]).astype(str)
    reference = np.stack([references[str(cell)] for cell in cells[eligible]])
    clean_population = frozen.predict(np.asarray(test["soh"])[eligible])
    clean_individual = np.asarray(residual[eligible], dtype=np.float32) - clean_population
    clean_norm = np.sum(clean_individual.astype(np.float64) ** 2, axis=(1, 2))
    if np.any(clean_norm <= 0) or not np.isfinite(clean_norm).all():
        raise RuntimeError("clean Individual residual has an invalid norm")
    keys = np.asarray(test["keys"]).astype(str)
    clean_index = {key: index for index, key in enumerate(keys)}
    eligible_indices = np.flatnonzero(eligible)
    eligible_row = {int(index): row for row, index in enumerate(eligible_indices)}
    return {
        "test": test,
        "keys": keys,
        "clean_index": clean_index,
        "eligible": eligible,
        "eligible_indices": eligible_indices,
        "eligible_row": eligible_row,
        "reference": reference,
        "clean_individual": clean_individual,
        "clean_norm": clean_norm,
        "s_ind": frozen.s_ind,
    }


def initialize(root: Path) -> None:
    directories = output_dirs(root)
    data_root = SOURCE_CODE_ROOT / "data" / "mit"
    required_paths = [MANIFOLD_PARAMETERS, MANIFOLD_CONFIG]
    required_paths += [
        checkpoint_for(root, VARIANTS[0], seed) for seed in SEEDS
    ]
    required_paths += [predictor_for(seed) for seed in SEEDS]
    required_paths += [
        data_root / "train" / "clean",
        data_root / "vali" / "clean",
        data_root / "test" / "clean",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing rebuilt ablation asset: {missing[0]}")

    frozen = FrozenPopulation(MANIFOLD_PARAMETERS, MANIFOLD_CONFIG)
    clean = load_clean_decomposition(frozen)
    condition_counts = {}
    for condition, _label in CONDITIONS:
        dataset = PairedBatteryDataset("mit", "test", condition)
        condition_counts[condition] = len(dataset)
        if len(dataset) != EXPECTED_TEST_ROWS:
            raise RuntimeError(
                f"MIT test condition {condition} rows changed: {len(dataset)}"
            )

    design = pd.DataFrame(
        [
            {
                "variant_slug": item["slug"],
                "variant": item["name"],
                "ablation_mode": item["mode"],
                "weak_charging_prior": item["prior"],
                "physics_consistency_gate": item["gate"],
                "training_status": "reused frozen rebuilt checkpoint"
                if item["slug"] == "full"
                else "independently trained in this run",
            }
            for item in VARIANTS
        ]
    )
    design.to_csv(directories["provenance"] / "four_variant_design.csv", index=False)

    inventory_rows = []
    for path in required_paths:
        if path.is_dir():
            file_count = sum(1 for item in path.rglob("*") if item.is_file())
            inventory_rows.append(
                {"path": str(path), "kind": "directory", "exists": True, "file_count": file_count}
            )
        else:
            inventory_rows.append(
                {
                    "path": str(path),
                    "kind": "file",
                    "exists": path.is_file(),
                    "file_count": 1 if path.is_file() else 0,
                }
            )
    pd.DataFrame(inventory_rows).to_csv(
        directories["provenance"] / "input_inventory.csv", index=False
    )

    write_json(
        directories["provenance"] / "run_manifest.json",
        {
            "status": "INITIALIZED",
            "protocol": "benchmark_819 strict MIT reconstruction component ablation",
            "repository": str(REPOSITORY_ROOT),
            "dataset": "mit",
            "seeds": list(SEEDS),
            "variants": [dict(item) for item in VARIANTS],
            "conditions": [
                {"internal": internal, "label": label} for internal, label in CONDITIONS
            ],
            "training": {
                "independent_variants": 3,
                "epochs": TRAIN_EPOCHS,
                "batch_size": TRAIN_BATCH_SIZE,
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip": GRADIENT_CLIP,
                "loss": "pointwise velocity MSE + physical-voltage gradient MSE",
                "gradient_loss_weight": 1.0,
                "train_view": "train/clean",
            },
            "evaluation": {
                "test_view": "four frozen paired corruption views",
                "sampling_steps": SAMPLING_STEPS,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "waveform_aggregation": "mean cycle RMSE, then mean across seeds and conditions",
                "soh_aggregation": "sqrt of mean cycle squared error, then mean across seeds and conditions",
                "E_pop_E_ind_aggregation": "cycle median, then mean across seeds and conditions",
                "eligible_test_cycles": int(clean["eligible"].sum()),
            },
            "frozen_assets": {
                "reconstruction_control": [str(checkpoint_for(root, VARIANTS[0], seed)) for seed in SEEDS],
                "seed_matched_trb": [str(predictor_for(seed)) for seed in SEEDS],
                "population_manifold_parameters": str(MANIFOLD_PARAMETERS),
                "population_manifold_config": str(MANIFOLD_CONFIG),
            },
            "new_model_training": 0,
            "new_model_inference": "evaluation only for this ablation; no Full retraining",
            "new_corruption_generation": 0,
            "asset_counts": {
                "clean_test": int(len(clean["keys"])),
                "eligible_test": int(clean["eligible"].sum()),
                "condition_counts": condition_counts,
            },
            "manifold": {
                "status": frozen.status,
                "n_knots": frozen.n_knots,
                "alpha": frozen.alpha,
                "S_ind": frozen.s_ind,
                "projection_grid": frozen.grid_size,
                "fit_performed_by_this_workflow": False,
            },
            "environment": {
                "host": socket.gethostname(),
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        },
    )
    write_json(
        directories["provenance"] / "training_manifest.json",
        {
            "status": "PENDING",
            "tasks": [
                {
                    "task_index": index,
                    "variant_slug": variant["slug"],
                    "seed": seed,
                }
                for index, (variant, seed) in enumerate(
                    (decode_train_task(i) for i in range(9))
                )
            ],
            "full_control": "existing rebuilt checkpoint; not retrained",
        },
    )
    print(f"initialized {root}", flush=True)


def train_one_epoch(
    model: AblationBatteryDiT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ablation_mode: str,
) -> float:
    model.train()
    losses = []
    for current, voltage, _soh, _rul in loader:
        current = current.to(device, non_blocking=True)
        voltage = voltage.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        predicted, target, _task_id = model(
            voltage,
            current,
            mode="recon_only_train",
            steps=SAMPLING_STEPS,
            ablation_mode=ablation_mode,
        )
        pointwise = F.mse_loss(predicted, target)
        gradient = voltage_gradient_loss(predicted, target, V_MAX - V_MIN)
        loss = pointwise + gradient
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("empty reconstruction training loader")
    return float(np.mean(losses))


def train(root: Path, task_index: int, requested_device: str) -> None:
    variant, seed = decode_train_task(task_index)
    directories = output_dirs(root)
    checkpoint = checkpoint_for(root, variant, seed)
    task_dir = checkpoint.parent
    complete = task_dir / "complete.json"
    last = task_dir / "last.pth"
    if complete.is_file() and checkpoint.is_file():
        print(f"already complete: {checkpoint}", flush=True)
        return
    if checkpoint.exists() and not complete.exists():
        raise RuntimeError(f"incomplete checkpoint exists; inspect before retrying: {checkpoint}")

    device = choose_device(requested_device)
    if device.type != "cuda":
        raise RuntimeError("formal ablation training must run on a GPU")
    torch.cuda.set_device(device)
    seed_everything(seed)
    args = model_args()
    model = AblationBatteryDiT(args).to(device)
    optimizer = torch.optim.Adam(model.recon.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    start_epoch = 1
    if last.is_file():
        payload = torch.load(last, map_location=device)
        load_reconstruction_checkpoint(model, last, map_location=device, strict=True)
        if isinstance(payload, dict) and isinstance(payload.get("optimizer"), dict):
            optimizer.load_state_dict(payload["optimizer"])
            start_epoch = int(payload.get("epoch", 0)) + 1

    dataset = BatteryDataset("mit", split="train", condition="clean")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
        generator=generator,
    )
    task_dir.mkdir(parents=True, exist_ok=True)
    train_log = task_dir / "train_metrics.jsonl"
    started = time.time()
    for epoch in range(start_epoch, TRAIN_EPOCHS + 1):
        t0 = time.time()
        loss = train_one_epoch(model, loader, optimizer, device, variant["mode"])
        record = {
            "epoch": epoch,
            "loss": loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - t0,
        }
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "variant_slug": variant["slug"],
                "ablation_mode": variant["mode"],
                "seed": seed,
            },
            last,
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == TRAIN_EPOCHS:
            print(json.dumps(record, sort_keys=True), flush=True)

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": TRAIN_EPOCHS,
            "variant_slug": variant["slug"],
            "variant": variant["name"],
            "ablation_mode": variant["mode"],
            "seed": seed,
            "protocol": "benchmark_819 strict MIT reconstruction component ablation",
        },
        checkpoint,
    )
    write_json(
        complete,
        {
            "status": "COMPLETE",
            "checkpoint": str(checkpoint),
            "variant": variant,
            "seed": seed,
            "epochs": TRAIN_EPOCHS,
            "n_train_samples": len(dataset),
            "elapsed_seconds": time.time() - started,
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(device),
        },
    )
    print(f"completed training: {variant['slug']} seed={seed}", flush=True)


def nearest_population(
    deviation: torch.Tensor,
    grid_population: torch.Tensor,
    grid_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = deviation.squeeze(-1)
    grid = grid_population.squeeze(-1)
    distance = (
        flat.square().sum(dim=1, keepdim=True)
        + grid.square().sum(dim=1).unsqueeze(0)
        - 2.0 * flat @ grid.transpose(0, 1)
    )
    selected = distance.argmin(dim=1)
    return grid_h[selected], grid[selected].unsqueeze(-1)


def result_paths(root: Path, seed: int, variant: dict[str, Any], condition: str) -> tuple[Path, Path]:
    directory = root / "task_results" / f"seed{seed}" / variant["slug"]
    return directory / f"{condition}.csv.gz", directory / f"{condition}.json"


def evaluate_condition(
    root: Path,
    seed: int,
    variant: dict[str, Any],
    condition: str,
    model: AblationBatteryDiT,
    predictor: TRB,
    frozen: FrozenPopulation,
    clean: dict[str, Any],
    device: torch.device,
) -> None:
    output, marker_path = result_paths(root, seed, variant, condition)
    if output.is_file() and marker_path.is_file():
        print(f"already complete: seed={seed} {variant['slug']} {condition}", flush=True)
        return
    dataset = PairedBatteryDataset("mit", "test", condition)
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    grid_population = torch.from_numpy(frozen.grid_population).to(device)
    grid_h = torch.from_numpy(frozen.grid_h).to(device)
    rows: list[pd.DataFrame] = []
    model.eval()
    predictor.eval()
    start_time = time.time()
    for current, voltage, clean_voltage, soh, _rul, keys in loader:
        key_list = [str(key) for key in keys]
        clean_indices = np.asarray([clean["clean_index"][key] for key in key_list], dtype=np.int64)
        valid = np.asarray([index in clean["eligible_row"] for index in clean_indices], dtype=bool)
        if not np.any(valid):
            continue
        row_indices = np.asarray(
            [clean["eligible_row"][int(index)] for index in clean_indices[valid]],
            dtype=np.int64,
        )
        current = current[valid].to(device, non_blocking=True)
        voltage = voltage[valid].to(device, non_blocking=True)
        clean_voltage = clean_voltage[valid].to(device, non_blocking=True)
        true_soh = soh[valid].reshape(-1).to(device, non_blocking=True)
        with torch.inference_mode():
            normalized = model(
                V=voltage,
                I=current,
                mode="recon_only_sampling",
                steps=SAMPLING_STEPS,
                ablation_mode=variant["mode"],
            )
            reconstruction = model._denormalize_phy(normalized)
            predicted_soh = predictor(reconstruction)[:, 0]
            reference = torch.from_numpy(clean["reference"][row_indices]).to(device)
            clean_individual = torch.from_numpy(clean["clean_individual"][row_indices]).to(device)
            clean_norm = torch.from_numpy(clean["clean_norm"][row_indices]).to(device)
            reconstructed_deviation = reconstruction - reference
            h_rec, rec_population = nearest_population(
                reconstructed_deviation, grid_population, grid_h
            )
            rec_individual = reconstructed_deviation - rec_population
            e_ind = (
                (rec_individual - clean_individual).square().mean(dim=(1, 2)).sqrt()
                / frozen.s_ind
            )
            rec_norm = rec_individual.square().sum(dim=(1, 2))
            a_ind = (rec_individual * clean_individual).sum(dim=(1, 2)) / clean_norm
            q = (rec_norm / clean_norm).sqrt()
            c_ind = a_ind / q
            waveform_rmse = (reconstruction - clean_voltage).square().mean(dim=(1, 2)).sqrt()
            soh_error = predicted_soh - true_soh
            e_pop = (h_rec - true_soh).abs()
        if not torch.isfinite(reconstruction).all():
            raise RuntimeError(f"non-finite reconstruction: {variant['slug']} {condition}")
        frame = pd.DataFrame(
            {
                "seed": seed,
                "variant_slug": variant["slug"],
                "variant": variant["name"],
                "mode": variant["mode"],
                "prior": variant["prior"],
                "gate": variant["gate"],
                "sample_index": clean_indices[valid],
                "battery": np.asarray([key.rsplit("/", 1)[0] for key in np.asarray(key_list)[valid]], dtype=str),
                "cycle": np.asarray([int(key.rsplit("/", 1)[1].removesuffix(".npz")) for key in np.asarray(key_list)[valid]], dtype=np.int32),
                "corruption": CONDITIONS[[item[0] for item in CONDITIONS].index(condition)][1],
                "corruption_internal": condition,
                "difficulty": "standard benchmark_819",
                "waveform_rmse": waveform_rmse.detach().cpu().numpy(),
                "soh_predicted": predicted_soh.detach().cpu().numpy(),
                "soh_true": true_soh.detach().cpu().numpy(),
                "soh_abs_error": soh_error.abs().detach().cpu().numpy(),
                "soh_squared_error": soh_error.square().detach().cpu().numpy(),
                "E_pop": e_pop.detach().cpu().numpy(),
                "E_ind": e_ind.detach().cpu().numpy(),
                "Q": q.detach().cpu().numpy(),
                "abs_Q_minus_1": (q - 1.0).abs().detach().cpu().numpy(),
                "C_ind": c_ind.detach().cpu().numpy(),
            }
        )
        rows.append(frame)
    if not rows:
        raise RuntimeError(f"no eligible evaluation rows: {variant['slug']} {condition}")
    result = pd.concat(rows, ignore_index=True)
    if len(result) != EXPECTED_ELIGIBLE_ROWS:
        raise RuntimeError(
            f"evaluation rows changed for {variant['slug']} {condition}: {len(result)}"
        )
    if not np.isfinite(result[list(ROW_FINITE_METRICS)].to_numpy()).all():
        raise RuntimeError(f"non-finite metrics: {variant['slug']} {condition}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, compression="gzip")
    write_json(
        marker_path,
        {
            "status": "COMPLETE",
            "seed": seed,
            "variant_slug": variant["slug"],
            "variant": variant["name"],
            "ablation_mode": variant["mode"],
            "corruption": condition,
            "n_cycles": len(result),
            "reconstruction_checkpoint": str(checkpoint_for(root, variant, seed)),
            "predictor_checkpoint": str(predictor_for(seed)),
            "frozen_manifold_parameters": str(MANIFOLD_PARAMETERS),
            "model_eval": not model.training,
            "predictor_eval": not predictor.training,
            "new_corruption_generated": False,
            "new_model_training": False,
            "host": socket.gethostname(),
            "device": str(device),
            "seconds": time.time() - start_time,
            "result": str(output),
        },
    )
    print(
        f"completed evaluation: seed={seed} {variant['slug']} {condition} rows={len(result)}",
        flush=True,
    )


def evaluate(root: Path, task_index: int, requested_device: str) -> None:
    variant, seed = decode_eval_task(task_index)
    device = choose_device(requested_device)
    if device.type != "cuda":
        raise RuntimeError("formal ablation evaluation must run on a GPU")
    torch.cuda.set_device(device)
    frozen = FrozenPopulation(MANIFOLD_PARAMETERS, MANIFOLD_CONFIG)
    clean = load_clean_decomposition(frozen)
    model = load_reconstruction(checkpoint_for(root, variant, seed), device)
    predictor = load_predictor(predictor_for(seed), device)
    for condition, _label in CONDITIONS:
        evaluate_condition(root, seed, variant, condition, model, predictor, frozen, clean, device)


def summarize_condition(group: pd.DataFrame) -> dict[str, float | int]:
    return {
        "waveform_rmse": float(group.waveform_rmse.mean()),
        "soh_rmse": float(np.sqrt(group.soh_squared_error.mean())),
        "E_pop": float(group.E_pop.median()),
        "E_ind": float(group.E_ind.median()),
        "Q": float(group.Q.median()),
        "abs_Q_minus_1": float(group.abs_Q_minus_1.median()),
        "C_ind": float(group.C_ind.median()),
        "n_cycles": int(len(group)),
    }


def aggregate(root: Path) -> None:
    directories = output_dirs(root)
    frames = []
    markers = []
    missing = []
    for variant in VARIANTS:
        for seed in SEEDS:
            for condition, _label in CONDITIONS:
                result_path, marker_path = result_paths(root, seed, variant, condition)
                if not result_path.is_file() or not marker_path.is_file():
                    missing.append(str(result_path))
                    continue
                frames.append(pd.read_csv(result_path, compression="gzip"))
                markers.append(json.loads(marker_path.read_text(encoding="utf-8")))
    if missing:
        raise RuntimeError(f"missing {len(missing)} evaluation tasks; first={missing[0]}")
    cycle = pd.concat(frames, ignore_index=True)
    expected = len(VARIANTS) * len(SEEDS) * len(CONDITIONS) * EXPECTED_ELIGIBLE_ROWS
    if len(cycle) != expected:
        raise RuntimeError(f"cycle rows {len(cycle)} != {expected}")

    identity_columns = ["sample_index", "battery", "cycle", "corruption_internal"]
    reference_identity = None
    for key, group in cycle.groupby(["seed", "variant_slug"], sort=False):
        identity = pd.MultiIndex.from_frame(group[identity_columns])
        if identity.has_duplicates:
            raise RuntimeError(f"duplicate test identity: {key}")
        ordered = identity.sort_values()
        if reference_identity is None:
            reference_identity = ordered
        elif not ordered.equals(reference_identity):
            raise RuntimeError(f"test identity mismatch: {key}")
    cycle.to_csv(directories["raw"] / "ablation_cycle_level.csv.gz", index=False, compression="gzip")

    grouping = [
        "seed", "variant_slug", "variant", "mode", "prior", "gate",
        "corruption", "corruption_internal", "difficulty",
    ]
    condition_seed_rows = []
    for values, group in cycle.groupby(grouping, sort=False):
        row = dict(zip(grouping, values))
        row.update(summarize_condition(group))
        condition_seed_rows.append(row)
    condition_seed = pd.DataFrame(condition_seed_rows)
    condition_seed.to_csv(
        directories["analysis"] / "ablation_condition_level_by_seed.csv", index=False
    )

    variant_grouping = ["variant_slug", "variant", "mode", "prior", "gate"]
    condition = (
        condition_seed.groupby(
            variant_grouping + ["corruption", "corruption_internal", "difficulty"],
            sort=False,
            as_index=False,
        )[list(ALL_METRICS)]
        .mean()
    )
    condition["n_seeds"] = len(SEEDS)
    condition["n_cycles_per_seed"] = EXPECTED_ELIGIBLE_ROWS
    condition.to_csv(directories["analysis"] / "ablation_condition_level.csv", index=False)

    summary = condition.groupby(variant_grouping, sort=False, as_index=False)[list(ALL_METRICS)].mean()
    summary["n_seeds"] = len(SEEDS)
    summary["n_conditions"] = len(CONDITIONS)
    order = [item["slug"] for item in VARIANTS]
    summary["_order"] = pd.Categorical(summary.variant_slug, order, ordered=True)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    summary.to_csv(directories["analysis"] / "ablation_summary.csv", index=False)

    markers_frame = pd.DataFrame(markers)
    qc_rows = []
    qc_rows.append(
        (
            "Full control checkpoints",
            "PASS",
            "three rebuilt seed-matched reconstruction checkpoints found; Full was not retrained",
        )
    )
    completed_training = all(
        (checkpoint_for(root, variant, seed).is_file()
         and checkpoint_for(root, variant, seed).parent.joinpath("complete.json").is_file())
        for variant in VARIANTS[1:]
        for seed in SEEDS
    )
    qc_rows.append(
        (
            "independently trained ablation checkpoints",
            "PASS" if completed_training else "FAIL",
            "nine variant-seed checkpoints completed at 200 epochs" if completed_training else "one or more training markers are missing",
        )
    )
    qc_rows.append(
        (
            "same formal test identities",
            "PASS",
            f"{len(CONDITIONS) * EXPECTED_ELIGIBLE_ROWS:,} eligible identities per seed-variant",
        )
    )
    qc_rows.append(
        (
            "same frozen seed-matched TRB",
            "PASS" if markers_frame.predictor_checkpoint.nunique() == len(SEEDS) else "FAIL",
            "one fixed predictor path per seed across all variants",
        )
    )
    qc_rows.append(
        (
            "same frozen Population manifold",
            "PASS" if markers_frame.frozen_manifold_parameters.nunique() == 1 else "FAIL",
            "serialized train-only parameters reloaded; no manifold fitting in this workflow",
        )
    )
    qc_rows.append(
        (
            "no new corruption generation",
            "PASS" if not markers_frame.new_corruption_generated.any() else "FAIL",
            "all four paired test views were read from source_code/data/mit/test",
        )
    )
    qc_rows.append(
        (
            "all formal rows retained",
            "PASS",
            "no condition filtering or variant-specific retuning",
        )
    )
    qc = pd.DataFrame(qc_rows, columns=["check", "status", "detail"])
    qc.to_csv(directories["analysis"] / "ablation_qc.csv", index=False)

    table = summary[
        ["variant_slug", "variant", "waveform_rmse", "soh_rmse", "E_pop", "E_ind"]
    ].rename(
        columns={
            "variant_slug": "Variant ID",
            "variant": "Variant",
            "waveform_rmse": "Waveform RMSE",
            "soh_rmse": "SOH RMSE",
            "E_pop": "E_pop",
            "E_ind": "E_ind",
        }
    )
    table.to_csv(directories["plotting"] / "ablation_four_metric_table.csv", index=False)

    report_lines = [
        "# Strict benchmark-819 ablation report",
        "",
        "This report contains only the retained MIT component ablation. The Full row uses the rebuilt formal BatteryDiT checkpoint for each seed; the other three rows were independently trained under the same 200-epoch reconstruction protocol.",
        "",
        "## Four-metric summary",
        "",
        "| Variant | Waveform RMSE | SOH RMSE | E_pop | E_ind |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        report_lines.append(
            f"| {row.Variant} | {row._2:.8g} | {row._3:.8g} | {row.E_pop:.8g} | {row.E_ind:.8g} |"
        )
    report_lines.extend(
        [
            "",
            "## Aggregation",
            "",
            "Waveform RMSE is the mean cycle-level RMSE. SOH RMSE is the square root of the mean cycle-level squared SOH error. E_pop and E_ind are cycle medians. Condition summaries first average the three seeds, and the four-condition summary then averages the fixed corruption views.",
            "",
            "## Guardrails",
            "",
            "The test views, test identities, seed-matched TRB checkpoints, and serialized Population manifold are shared across variants. No new model is trained for Full, no corruption is generated, and the manifold is not refit.",
            "",
        ]
    )
    report_lines.extend(
        f"- {row.check}: **{row.status}** — {row.detail}." for row in qc.itertuples(index=False)
    )
    (directories["analysis"] / "ablation_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    manifest_path = directories["provenance"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "COMPLETE"
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["outputs"] = {
        "raw_cycle_level": str(directories["raw"] / "ablation_cycle_level.csv.gz"),
        "summary": str(directories["analysis"] / "ablation_summary.csv"),
        "four_metric_table": str(directories["plotting"] / "ablation_four_metric_table.csv"),
        "qc": str(directories["analysis"] / "ablation_qc.csv"),
        "report": str(directories["analysis"] / "ablation_report.md"),
    }
    write_json(manifest_path, manifest)
    training_manifest_path = directories["provenance"] / "training_manifest.json"
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    training_manifest["status"] = "COMPLETE"
    write_json(training_manifest_path, training_manifest)
    print(f"aggregate complete: {root}", flush=True)


def smoke(root: Path, requested_device: str) -> None:
    """Load one rebuilt control path and exercise both aligned branches."""
    device = choose_device(requested_device)
    seed = SEEDS[0]
    frozen = FrozenPopulation(MANIFOLD_PARAMETERS, MANIFOLD_CONFIG)
    model = load_reconstruction(checkpoint_for(root, VARIANTS[0], seed), device)
    predictor = load_predictor(predictor_for(seed), device)
    dataset = PairedBatteryDataset("mit", "test", "clean")
    current, voltage, _clean_voltage, _soh, _rul, _key = dataset[5]
    current = current.unsqueeze(0).to(device)
    voltage = voltage.unsqueeze(0).to(device)
    with torch.inference_mode():
        full = model(voltage, current, mode="recon_only_sampling", steps=1, ablation_mode="all")
        branch = model(
            voltage,
            current,
            mode="recon_only_sampling",
            steps=1,
            ablation_mode="sensor+mask",
        )
        soh = predictor(model._denormalize_phy(full))
    if not torch.isfinite(full).all() or not torch.isfinite(branch).all() or not torch.isfinite(soh).all():
        raise RuntimeError("aligned ablation smoke produced a non-finite output")
    write_json(
        root / "provenance" / "smoke.json",
        {
            "status": "PASS",
            "seed": seed,
            "device": str(device),
            "full_branch_shape": list(full.shape),
            "ablation_branch_shape": list(branch.shape),
            "predictor_shape": list(soh.shape),
            "frozen_S_ind": frozen.s_ind,
        },
    )
    print(f"smoke PASS: {root}", flush=True)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-root", required=True)
    command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    sub = command.add_subparsers(dest="command", required=True)
    sub.add_parser("initialize")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--task-index", type=int, required=True)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--task-index", type=int, required=True)
    sub.add_parser("aggregate")
    sub.add_parser("smoke")
    return command


def main() -> None:
    args = parser().parse_args()
    root = run_root(args.run_root)
    if args.command == "initialize":
        initialize(root)
    elif args.command == "train":
        train(root, args.task_index, args.device)
    elif args.command == "evaluate":
        evaluate(root, args.task_index, args.device)
    elif args.command == "aggregate":
        aggregate(root)
    elif args.command == "smoke":
        smoke(root, args.device)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
