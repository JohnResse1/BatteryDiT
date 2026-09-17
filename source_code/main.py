""" Main entry point for all formal BatteryDiT experiments."""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from exp.downstream_prediction import Exp_Predict_Downstream
from exp.direct_prediction import Exp_Direct_Prediction
from exp.prediction_stress_test import Exp_Predict_Stress_Test
from exp.degradation_signature import Exp_Degradation_Signature
from exp.multi_dataset import Exp_MultiDataset
from exp.reconstruction import Exp_Reconstruction
from exp.reconstruction_stress_test import Exp_Reconstruction_Stress_Test
from models.dataset_physics import dataset_physics
from data_provider.data_loader import dataset_name
from exp.population_individual import Exp_Population_Individual

# All experiments classes
TASK_REGISTRY = {
    "Reconstruction": Exp_Reconstruction,
    "ReconstructionStressTest": Exp_Reconstruction_Stress_Test,
    "PredictDownstream": Exp_Predict_Downstream,
    "DirectPrediction": Exp_Direct_Prediction,
    "PredictStressTest": Exp_Predict_Stress_Test,
    "DegradationSignature": Exp_Degradation_Signature,
    "MultiDataset": Exp_MultiDataset,
    "PopulationIndividual": Exp_Population_Individual,
}

# str2bool for argparse, from https://stackoverflow.com/a/43357954
def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()

    command.add_argument("--task_name", required=True, choices=sorted(TASK_REGISTRY))
    command.add_argument("--is_training", type=int, required=True, choices=(0, 1))
    command.add_argument("--model_id", default="")
    command.add_argument("--recon_model", default="", choices=("", "batterydit", "csdi", "transformer", "kalman"))
    command.add_argument("--direct_model", default="", choices=("", "lstm", "gru", "transformer", "tcn", "trb"))
    command.add_argument("--predictor", default="trb", choices=("trb", "tcn"))
    command.add_argument("--data", default="")
    command.add_argument("--itr", type=int, default=1)
    command.add_argument("--seed", type=int, default=2021)

    command.add_argument("--checkpoints", default="checkpoints")
    command.add_argument("--results_dir", default="../results")
    command.add_argument("--recn_chkpt", default="")
    command.add_argument("--pred_chkpt", default="")
    command.add_argument("--multi_stage", choices=("reconstruction", "downstream", "all"), default="reconstruction")
    command.add_argument("--multi_datasets", nargs="*", default=[], help="multi-dataset evaluation datasets, or all")
    command.add_argument("--multi_seeds", nargs="*", type=int, default=[], help="multi-dataset evaluation seeds")

    command.add_argument("--epochs", type=int, default=200)
    command.add_argument("--batch_size", type=int, default=64)
    command.add_argument("--eval_batch_size", type=int, default=None)
    command.add_argument("--learning_rate", type=float, default=1e-4)
    command.add_argument("--weight_decay", type=float, default=0.0)
    command.add_argument("--num_workers", type=int, default=4)
    command.add_argument("--use_gpu", type=str2bool, default=True)
    command.add_argument("--gpu", type=int, default=0)

    command.add_argument("--ocv_table_path", default="")
    command.add_argument("--po_type", default="MIT")
    command.add_argument("--v_min", type=float, default=2.0)
    command.add_argument("--v_max", type=float, default=3.6)
    command.add_argument("--i_min", type=float, default=0.0)
    command.add_argument("--i_max", type=float, default=6.2)
    command.add_argument("--soc_points", type=int, default=1000)
    command.add_argument("--voltage_points", type=int, default=1000)

    command.add_argument("--seq_len", type=int, default=1000)
    command.add_argument("--hidden_size", type=int, default=256)
    command.add_argument("--depth", type=int, default=6)
    command.add_argument("--num_heads", type=int, default=4)
    command.add_argument("--timestep_embed_dim", type=int, default=256)
    command.add_argument("--timestep_freq_embed_size", type=int, default=64)
    command.add_argument("--sampling_steps", type=int, default=1)
    command.add_argument("--dropout", type=float, default=0.1)

    command.add_argument("--d_model", type=int, default=256)
    command.add_argument("--n_heads", type=int, default=4)
    command.add_argument("--d_ff", type=int, default=512)
    command.add_argument("--e_layers", type=int, default=4)
    command.add_argument("--activation", default="gelu")
    command.add_argument("--diffusion_steps", type=int, default=50)
    command.add_argument("--csdi_channels", type=int, default=64)
    command.add_argument("--csdi_layers", type=int, default=4)
    command.add_argument("--csdi_official_heads", type=int, default=8)
    command.add_argument("--csdi_diffusion_dim", type=int, default=128)
    command.add_argument("--csdi_n_samples", type=int, default=1)
    command.add_argument("--csdi_sample_clip", type=float, default=10.0)
    command.add_argument("--eval_reconstruction_seed", type=int, default=3407)
    command.add_argument("--stress_seed", type=int, default=3407)
    command.add_argument("--cycles_per_cell", type=int, default=64)
    command.add_argument("--kf_q", type=float, default=1e-5)
    command.add_argument("--kf_r", type=float, default=1e-2)
    command.add_argument("--kf_threshold", type=float, default=0.1)

    command.add_argument("--p_clean", type=float, default=0.10)
    command.add_argument("--p_downsample", type=float, default=0.15)
    command.add_argument("--p_segment", type=float, default=0.45)
    command.add_argument("--p_random", type=float, default=0.15)
    command.add_argument("--segment_ratio_min", type=float, default=0.10)
    command.add_argument("--segment_ratio_max", type=float, default=0.70)
    command.add_argument("--validation_interval", type=int, default=10)
    command.add_argument("--validation_per_cell", type=int, default=20)
    command.add_argument("--patience", type=int, default=10)
    command.add_argument("--worst_weight", type=float, default=0.5)
    command.add_argument(
        "--save_reconstructions", type=str2bool, default=False,
        help="save full paired test trajectories for a later scientific study",
    )
    return command


# Set the fixed PO and voltage configuration for one formal dataset.
def configure_dataset(args: argparse.Namespace) -> None:
    """Fill the fixed PO and voltage configuration for one formal dataset."""
    args.data = dataset_name(args.data)
    physics = dataset_physics(args.data)
    args.po_type = physics.po_type
    args.v_min, args.v_max = physics.v_min, physics.v_max
    if not args.ocv_table_path:
        args.ocv_table_path = f"models/ocv/{args.data}.npz"

# Set the random seed for reproducibility.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Run the experiment based on the provided arguments.
def run(args: argparse.Namespace) -> None:
    # Multi-dataset generalization evaluation
    if args.task_name == "MultiDataset":
        experiment = Exp_MultiDataset(args)
        experiment.train("MultiDataset")
        return
    # Model-based reconstruction or downstream prediction tasks
    is_direct = args.task_name == "DirectPrediction"
    if not args.model_id or not args.data or (is_direct and not args.direct_model) or (not is_direct and not args.recon_model):
        required = "--model_id, --direct_model, and --data" if is_direct else "--model_id, --recon_model, and --data"
        raise ValueError(f"model tasks require {required}")
    if args.task_name == "PredictDownstream" and args.predictor == "tcn" and args.recon_model != "batterydit":
        raise ValueError("the formal reconstruction→TCN comparator is defined only for --recon_model batterydit")


    configure_dataset(args)
    set_seed(args.seed)
    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    experiment_class = TASK_REGISTRY[args.task_name]
    for index in range(args.itr):
        model_name = args.direct_model if is_direct else args.recon_model
        predictor_name = f"_{args.predictor}" if args.task_name in {"PredictDownstream", "DegradationSignature", "PopulationIndividual"} else ""
        setting = f"{args.task_name}_{args.model_id}_{model_name}{predictor_name}_{args.data}_{index}"
        experiment = experiment_class(args)
        if args.is_training:
            experiment.train(setting)
            experiment.vali(setting)
        experiment.test(setting)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run(parser().parse_args())
