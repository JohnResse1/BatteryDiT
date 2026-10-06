"""Profile the new repository's frozen reconstruction methods.

This is an analysis-only cost audit.  It loads the local formal checkpoints,
uses one fixed MIT test trajectory for every method, and reports parameter
count, FP32 state size, latency, throughput, and peak memory.  It does not
train a model, alter a checkpoint, or estimate FLOPs when no reliable profiler
is part of the repository protocol.

The four reconstruction methods are reported separately.  The frozen
BatteryDiT-to-TRB pipeline is included as an additional downstream row because
it was part of the earlier computational-cost comparison, not as a fifth
reconstruction method.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from analysis.utils.paths import REPOSITORY_ROOT, RESULTS_ROOT

from .provenance import canonical_run


METHODS = ("kalman", "transformer", "csdi", "batterydit", "batterydit_trb")
RECONSTRUCTION_METHODS = {"kalman", "transformer", "csdi", "batterydit"}
DISPLAY_NAMES = {
    "kalman": "Kalman",
    "transformer": "Transformer",
    "csdi": "CSDI",
    "batterydit": "BatteryDiT",
    "batterydit_trb": "BatteryDiT + TRB",
}
DEFAULT_BATCH_SIZES = (1, 64)


def reconstruction_result_directory(method: str) -> Path:
    return RESULTS_ROOT / "Reconstruction" / "mit" / (
        f"Reconstruction_seed2021_{method}_mit_0"
    )


def reconstruction_checkpoint(method: str) -> Path:
    return (
        REPOSITORY_ROOT / "source_code" / "checkpoints" / "Reconstruction" / "mit"
        / f"Reconstruction_seed2021_{method}_mit_0" / "checkpoint.pth"
    )


def trb_checkpoint() -> Path:
    return (
        REPOSITORY_ROOT / "source_code" / "checkpoints" / "PredictDownstream" / "mit"
        / "PredictDownstream_seed2021_batterydit_trb_mit_0" / "best.pth"
    )


def load_fixed_input() -> tuple[np.ndarray, np.ndarray, str]:
    """Use the same frozen corrupted MIT trajectory for every method."""
    source = reconstruction_result_directory("batterydit") / "reconstructions_random.npz"
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        keys = np.asarray(archive["keys"]).astype(str)
        target = "b1c1/1.npz"
        index = int(np.where(keys == target)[0][0]) if target in set(keys) else 0
        current = np.asarray(archive["input_current"][index], dtype=np.float32)
        voltage = np.asarray(archive["input_voltage"][index], dtype=np.float32)
        key = str(keys[index])
    if current.shape != (1000, 1) or voltage.shape != (1000, 1):
        raise ValueError(f"expected one (1000, 1) trajectory, got I={current.shape}, V={voltage.shape}")
    return current, voltage, key


def load_args(method: str) -> SimpleNamespace:
    config = reconstruction_result_directory(method) / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    arguments = json.loads(config.read_text())["arguments"]
    args = SimpleNamespace(**arguments)
    args.data = "mit"
    args.gpu = 0
    args.use_gpu = True
    args.ocv_table_path = str(REPOSITORY_ROOT / "source_code" / "models" / "ocv" / "mit.npz")
    args.sampling_steps = int(getattr(args, "sampling_steps", 1))
    return args


def load_state_dict(model: torch.nn.Module, path: Path) -> None:
    from models.checkpoint_loading import load_reconstruction_checkpoint

    if not path.is_file():
        raise FileNotFoundError(path)
    load_reconstruction_checkpoint(model, path, map_location="cpu", strict=True)


def active_named_parameters(model: torch.nn.Module):
    return [
        (name, value) for name, value in model.named_parameters()
        if not name.startswith("pred.")
    ]


def parameter_count(model: torch.nn.Module) -> tuple[int | None, int | None]:
    values = active_named_parameters(model)
    if not values:
        return None, None
    total = sum(value.numel() for _, value in values)
    trainable = sum(value.numel() for _, value in values if value.requires_grad)
    return int(total), int(trainable)


def fp32_state_bytes(model: torch.nn.Module) -> int | None:
    state = {
        name: value for name, value in model.state_dict().items()
        if not name.startswith("pred.")
    }
    if not state:
        return None
    # The table is explicitly an FP32 model-size comparison.  Integer buffers
    # retain their actual element size; all learned floating tensors are FP32.
    return int(sum(value.numel() * (4 if value.is_floating_point() else value.element_size())
                   for value in state.values()))


def load_model(method: str, device: torch.device):
    from exp.exp_basic import Exp_Basic
    from models.predictors.trb.model import Model as TRB

    args = load_args("batterydit" if method == "batterydit_trb" else method)
    checkpoints = []
    if method == "batterydit_trb":
        reconstruction = Exp_Basic.model_dict["batterydit"](args).float()
        load_state_dict(reconstruction, reconstruction_checkpoint("batterydit"))
        predictor = TRB(args, dropout=float(getattr(args, "dropout", 0.1))).float()
        predictor_path = trb_checkpoint()
        payload = torch.load(predictor_path, map_location="cpu")
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        predictor.load_state_dict(state, strict=True)
        model = torch.nn.ModuleDict({"reconstruction": reconstruction, "predictor": predictor})
        checkpoints = [reconstruction_checkpoint("batterydit"), predictor_path]
    elif method == "kalman":
        model = Exp_Basic.model_dict["kalman"](args).float()
    else:
        model = Exp_Basic.model_dict[method](args).float()
        checkpoint = reconstruction_checkpoint(method)
        load_state_dict(model, checkpoint)
        checkpoints = [checkpoint]
    model = model.to(device).eval()
    total, trainable = parameter_count(model)
    return model, args, checkpoints, {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "model_state_size_bytes_fp32": fp32_state_bytes(model),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "checkpoint_sizes_bytes": [int(path.stat().st_size) for path in checkpoints],
    }


def inference_function(method: str, model: torch.nn.Module, args: SimpleNamespace):
    if method == "kalman":
        def infer(voltage, current):
            return model(V=voltage, I=current, mode="recon_only_sampling", steps=1)
        return infer, "sequential filtering over L=1000 observations", None

    if method == "batterydit_trb":
        reconstruction = model["reconstruction"]
        predictor = model["predictor"]

        def infer(voltage, current):
            normalized = reconstruction(
                V=voltage, I=current, mode="recon_only_sampling", steps=1
            )
            physical = reconstruction._denormalize_phy(normalized).squeeze(-1)
            return predictor(physical)

        return infer, "1 BatteryDiT evaluation + 1 TRB evaluation", 1

    def infer(voltage, current):
        normalized = model(
            V=voltage, I=current, mode="recon_only_sampling",
            steps=int(getattr(args, "sampling_steps", 1)),
        )
        return model._denormalize_phy(normalized)

    if method == "csdi":
        steps = int(getattr(args, "diffusion_steps", 50))
        return infer, f"{steps} reverse diffusion evaluations", steps
    return infer, "1 network evaluation", 1


def validate_output(method: str, output: torch.Tensor, batch_size: int) -> None:
    expected = (batch_size, 2) if method == "batterydit_trb" else (batch_size, 1000, 1)
    if tuple(output.shape) != expected:
        raise ValueError(f"{method}: output shape {tuple(output.shape)} != {expected}")
    if not torch.isfinite(output).all():
        raise FloatingPointError(f"{method}: non-finite output")


def environment(device: torch.device) -> dict:
    record = {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "precision": "FP32",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record.update({
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        })
    return record


def profile_method(
    method: str,
    device: torch.device,
    current_np: np.ndarray,
    voltage_np: np.ndarray,
    batch_sizes: tuple[int, ...],
    warmup: int,
    repeats: int,
) -> tuple[list[dict], list[dict]]:
    model, args, checkpoints, metadata = load_model(method, device)
    infer, inference_steps, sampling_steps = inference_function(method, model, args)
    raw_rows, summary_rows = [], []
    for batch_size in batch_sizes:
        if method == "kalman" and batch_size != 1:
            continue
        current = torch.from_numpy(np.repeat(current_np[None], batch_size, axis=0)).to(device)
        voltage = torch.from_numpy(np.repeat(voltage_np[None], batch_size, axis=0)).to(device)
        with torch.inference_mode():
            first = infer(voltage, current)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            validate_output(method, first, batch_size)
            del first
            for _ in range(int(warmup)):
                output = infer(voltage, current)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                validate_output(method, output, batch_size)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                baseline_memory = int(torch.cuda.memory_allocated(device))
                torch.cuda.reset_peak_memory_stats(device)
                memory_output = infer(voltage, current)
                torch.cuda.synchronize(device)
                validate_output(method, memory_output, batch_size)
                peak_memory = int(torch.cuda.max_memory_allocated(device))
                incremental_memory = peak_memory - baseline_memory
                del memory_output
            else:
                baseline_memory = peak_memory = incremental_memory = None

            elapsed = []
            for run_index in range(1, int(repeats) + 1):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter_ns()
                output = infer(voltage, current)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                stop = time.perf_counter_ns()
                validate_output(method, output, batch_size)
                elapsed_ms = (stop - start) / 1_000_000.0
                elapsed.append(elapsed_ms)
                raw_rows.append({
                    "method": method,
                    "method_display": DISPLAY_NAMES[method],
                    "batch_size": batch_size,
                    "run_index": run_index,
                    "elapsed_ms_batch": elapsed_ms,
                    "latency_ms_per_trajectory": elapsed_ms / batch_size,
                })

        per_trajectory = np.asarray(elapsed, dtype=np.float64) / batch_size
        summary_rows.append({
            "method": method,
            "method_display": DISPLAY_NAMES[method],
            "batch_size": batch_size,
            "total_parameters": metadata["total_parameters"],
            "trainable_parameters": metadata["trainable_parameters"],
            "model_state_size_bytes_fp32": metadata["model_state_size_bytes_fp32"],
            "model_state_size_mb_fp32": (
                metadata["model_state_size_bytes_fp32"] / 1e6
                if metadata["model_state_size_bytes_fp32"] is not None else None
            ),
            "inference_steps": inference_steps,
            "sampling_steps": sampling_steps,
            "latency_median_ms_batch": float(statistics.median(elapsed)),
            "latency_mean_ms_batch": float(statistics.mean(elapsed)),
            "latency_sd_ms_batch": float(statistics.stdev(elapsed)) if len(elapsed) > 1 else 0.0,
            "latency_p95_ms_batch": float(np.percentile(elapsed, 95)),
            "latency_median_ms_per_trajectory": float(np.median(per_trajectory)),
            "latency_mean_ms_per_trajectory": float(np.mean(per_trajectory)),
            "latency_p95_ms_per_trajectory": float(np.percentile(per_trajectory, 95)),
            "throughput_trajectories_per_second": float(batch_size / (np.median(elapsed) / 1000.0)),
            "peak_memory_bytes": peak_memory,
            "incremental_peak_memory_bytes": incremental_memory,
            "device": str(device),
            "checkpoint_paths": " | ".join(metadata["checkpoint_paths"]) or "N/A",
            "checkpoint_sizes_bytes": json.dumps(metadata["checkpoint_sizes_bytes"]),
            "flops": "NOT_AVAILABLE",
            "status": "PASS",
        })
        del voltage, current
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return raw_rows, summary_rows


def smoke(run_id: str | None = None) -> Path:
    run = canonical_run("computational_efficiency", run_id or "debug_smoke")
    summary = pd.DataFrame([
        {
            "method": method, "method_display": DISPLAY_NAMES[method],
            "status": "SMOKE_ONLY", "flops": "NOT_AVAILABLE",
        }
        for method in METHODS
    ])
    run.write_table("analysis", "computational_efficiency_summary.csv", summary)
    run.write_json("computational_efficiency_protocol.json", {
        "status": "SMOKE_PASS", "methods": list(METHODS),
        "new_model_training": 0, "new_model_inference": 0,
    })
    run.write_manifest(
        config={"status": "SMOKE_PASS", "new_model_training": 0, "new_model_inference": 0},
        input_paths={"synthetic": "generated_in_memory"}, seeds=[],
    )
    print(f"Computational-efficiency smoke outputs -> {run.root}")
    return run.root


def run(
    run_id: str | None = None,
    method: str = "all",
    device_name: str = "cuda",
    warmup: int = 5,
    repeats: int = 30,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    smoke_flag: bool = False,
) -> Path:
    if smoke_flag:
        return smoke(run_id)
    if method != "all" and method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose all or {METHODS}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for neural computational profiling")
    selected = METHODS if method == "all" else (method,)
    run = canonical_run("computational_efficiency", run_id)
    current, voltage, input_key = load_fixed_input()
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    raw_rows, summary_rows = [], []
    for item in selected:
        item_device = torch.device("cpu") if item == "kalman" else device
        print(f"[{item}] profiling on {item_device}", flush=True)
        raw, summary = profile_method(
            item, item_device, current, voltage, batch_sizes,
            warmup, repeats,
        )
        raw_rows.extend(raw); summary_rows.extend(summary)
        print(f"[{item}] complete", flush=True)
    raw_frame = pd.DataFrame(raw_rows)
    summary_frame = pd.DataFrame(summary_rows)
    run.write_table("raw", "computational_efficiency_raw.csv", raw_frame)
    run.write_table("analysis", "computational_efficiency_summary.csv", summary_frame)
    run.write_json("computational_efficiency_environment.json", environment(device))
    config = {
        "methods": list(selected),
        "reconstruction_methods": sorted(RECONSTRUCTION_METHODS & set(selected)),
        "additional_pipeline": "batterydit_trb" if "batterydit_trb" in selected else None,
        "dataset": "mit",
        "input_key": input_key,
        "input_source": str(reconstruction_result_directory("batterydit") / "reconstructions_random.npz"),
        "batch_sizes": list(batch_sizes),
        "kalman_batch_sizes": [1],
        "warmup_iterations": int(warmup),
        "timing_iterations": int(repeats),
        "precision": "FP32",
        "latency_scope": "model forward plus physical-unit output; excludes input loading and model construction",
        "flops": "NOT_AVAILABLE; no accepted FLOP profiler is part of the new-repository protocol",
        "new_model_training": 0,
        "new_model_inference": 1,
        "checkpoint_mutation": 0,
    }
    run.write_json("computational_efficiency_protocol.json", config)
    checkpoint_paths = {
        method_name: " | ".join(row["checkpoint_paths"] for row in summary_rows
                                 if row["method"] == method_name and row["batch_size"] == 1)
        for method_name in selected
    }
    run.write_manifest(
        config=config,
        input_paths={"input_archive": config["input_source"], **checkpoint_paths},
        checkpoint_ids=checkpoint_paths,
        seeds=[2021],
    )
    print(f"Computational-efficiency outputs -> {run.root}")
    return run.root


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id")
    command.add_argument("--method", choices=("all", *METHODS), default="all")
    command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    command.add_argument("--warmup", type=int, default=5)
    command.add_argument("--repeats", type=int, default=30)
    command.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES))
    command.add_argument("--smoke", action="store_true")
    return command


def main() -> None:
    args = parser().parse_args()
    run(
        args.run_id, args.method, args.device, args.warmup, args.repeats,
        tuple(args.batch_sizes), args.smoke,
    )


if __name__ == "__main__":
    main()
