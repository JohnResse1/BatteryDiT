"""Single access point to the certified decomposition and formal artifacts.

The new repository's certified implementation lives in
`source_code/exp/population_individual.py`.  This module wraps it so every
workflow reads the *same* objects through one interface:

    R_i,c = (V_clean,i,c - V_ref,i) - M(SOH_i,c)

It also centralises the formal-artifact layout (`results/<task>/<dataset>/...`)
so no workflow hard-codes a path.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np

from analysis.utils.paths import REPOSITORY_ROOT, RESULTS_ROOT

SOURCE_CODE = REPOSITORY_ROOT / "source_code"

# Fixed, pre-registered analysis constants (inherited unchanged from the
# certified analyses; nothing here is chosen after seeing a result).
ANCHORS = (0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84)
ANCHOR_HALF_WIDTH = 0.005
MIN_CYCLES_PER_ANCHOR = 5

PAST_WINDOWS = (25, 50, 100)
FUTURE_HORIZONS = (25, 50, 75, 100)
FIXED_SOH_LOSSES = (0.02, 0.03)

PCA_K = (3, 5, 10, 20)
PRIMARY_K = 10
PCA_VARIANCE_TARGETS = (0.80, 0.90)

RIDGE_ALPHA = 1.0
KNN_NEIGHBOURS = 25
MLP_HIDDEN = (32, 16)
MLP_EPOCHS = 300
MLP_LEARNING_RATE = 1e-3

N_PERM = 2000
N_BOOT = 2000
SEED = 20260926

BATCHES = ("b1", "b2", "b3")

# Formal six nominal-SOH systems (SDU excluded upstream: no compatible
# nominal-capacity SOH target).
SYSTEMS = {
    "mit": ("MIT", 2021),
    "hust": ("HUST", 2023),
    "mich": ("MICH", 2021),
    "stanford_2": ("Stanford", 2021),
    "tju_nca_45c": ("TJU-NCA", 2021),
    "tju_ncm_45c": ("TJU-NCM", 2021),
}

CORRUPTIONS = ("downsample", "random", "segment", "destructive", "hybrid")
SEVERITIES = ("L1", "L2", "L3", "L4", "L5")

# which formal NPZ condition string backs each corruption label
CONDITION_FILE = {
    "downsample": "downsample",
    "random": "random",
    "segment": "segment",
    "destructive": "destructive",
}


def log(message: str) -> None:
    print(message, flush=True)


@functools.lru_cache(maxsize=1)
def module():
    """Import the certified decomposition module from `source_code`."""
    if str(SOURCE_CODE) not in sys.path:
        sys.path.insert(0, str(SOURCE_CODE))
    from exp import population_individual  # noqa: PLC0415
    return population_individual


# --------------------------------------------------------------------------
# certified decomposition state
# --------------------------------------------------------------------------
_MANIFOLD_SPLITS = ("train", "vali")


def load_state(dataset: str = "mit", splits=("train", "vali", "test")) -> dict:
    """Frozen manifold plus the certified individual residual for each split.

    Returns, per split, `residual` = R (manifold-subtracted), `population` =
    M(SOH), plus cell/cycle/soh identity arrays.  The manifold is selected with
    the certified train/validation rule and never reselected here.

    The train and validation blocks are always loaded because the certified
    manifold selection needs them; `splits` only controls which splits are
    returned.
    """
    certified = module()
    needed = tuple(dict.fromkeys(tuple(_MANIFOLD_SPLITS) + tuple(splits)))
    blocks = {name: certified.load_clean(dataset, name) for name in needed}
    spline, model, s_ind, validation_table, manifold = certified.select_manifold(
        blocks["train"], blocks["vali"])

    def individual(block, name):
        residual, references, _ = certified.residualize(
            block, require_five=(name != "test"))
        population = np.asarray(certified.predict(model, spline, block["soh"]),
                                dtype=np.float64).reshape(len(block["soh"]), -1)
        finite = np.isfinite(residual).all(axis=(1, 2))
        index = np.where(finite)[0]
        return {
            "residual": np.asarray(residual, dtype=np.float64)[index].reshape(len(index), -1),
            "population": population[index],
            "cell": np.asarray(block["cells"]).astype(str)[index],
            "cycle": np.asarray(block["cycles"], dtype=np.int64)[index],
            "soh": np.asarray(block["soh"], dtype=np.float64)[index],
            "references": references,
        }

    state = {name: individual(blocks[name], name) for name in splits}
    state["s_ind"] = float(s_ind)
    state["manifold"] = manifold
    state["spline"], state["model"] = spline, model
    state["certified"] = certified
    return state


def load_sequences(dataset: str = "mit", splits=("train", "vali", "test")) -> dict:
    """Per-battery ordered residual sequences (the longitudinal unit)."""
    state = load_state(dataset, splits)
    out = {name: [] for name in splits}
    for name in splits:
        block = state[name]
        for cell in sorted(set(block["cell"])):
            mask = block["cell"] == cell
            order = np.argsort(block["cycle"][mask], kind="stable")
            out[name].append({
                "cell": str(cell), "batch": str(cell)[:2],
                "cycle": block["cycle"][mask][order].astype(np.int64),
                "soh": block["soh"][mask][order].astype(np.float64),
                "residual": block["residual"][mask][order],
            })
    out["s_ind"] = state["s_ind"]
    out["manifold"] = state["manifold"]
    out["spline"], out["model"] = state["spline"], state["model"]
    out["state"] = state
    return out


def split_manifest(dataset: str) -> dict[str, dict[str, int]]:
    """Return deterministic train/validation/test counts for provenance.

    The manifest is derived from the same clean loader used by the certified
    decomposition.  It is intentionally a small JSON-serialisable record,
    rather than a second split implementation, so canonical R5 workflows can
    prove which batteries/trajectories were consumed without changing the
    experiment outputs.
    """
    # Smoke tests intentionally use the injected synthetic loader.  For real
    # datasets inspect paths only.  Calling ``load_clean`` here would reread every
    # 1000-point waveform after the analysis has already loaded the split,
    # making provenance export needlessly expensive on the formal cohorts.
    if dataset == "smoke":
        certified = module()
        manifest: dict[str, dict[str, int]] = {}
        for split in ("train", "vali", "test"):
            block = certified.load_clean(dataset, split)
            cells = np.asarray(block["cells"]).astype(str)
            manifest[split] = {
                "n_trajectories": int(len(cells)),
                "n_batteries": int(len(set(cells))),
                "n_cycles": int(len(cells)),
                "battery_ids": sorted(set(cells.tolist())),
            }
        return manifest

    from data_provider.data_loader import BatteryDataset

    manifest: dict[str, dict[str, int]] = {}
    for split in ("train", "vali", "test"):
        data = BatteryDataset(dataset, split=split, condition="clean")
        keys = [str(path.relative_to(data.root)) for path in data.paths]
        cells = np.asarray([key.rsplit("/", 1)[0] for key in keys]).astype(str)
        manifest[split] = {
            "n_trajectories": int(len(cells)),
            "n_batteries": int(len(set(cells))),
            "n_cycles": int(len(keys)),
            "battery_ids": sorted(set(cells.tolist())),
        }
    return manifest


def clean_validation_metrics(dataset: str = "mit") -> tuple[pd.DataFrame, dict]:
    """Evaluate clean validation trajectories with the certified decomposition.

    ``E_pop`` is intentionally obtained by passing the clean validation
    waveform through :func:`evaluate_reconstruction`, not by substituting a
    waveform RMSE or a per-battery manifold score.  This is the frozen R5
    threshold definition: P90/P95/P99 of trajectory-level clean-validation
    ``E_pop`` after the deterministic five-cycle reference exclusion.
    """
    certified = module()
    train = certified.load_clean(dataset, "train")
    validation = certified.load_clean(dataset, "vali")
    spline, model, s_ind, selection, manifold = certified.select_manifold(
        train, validation
    )
    state = certified.prepare_test_decomposition(validation, spline, model)
    metrics, _ = certified.evaluate_reconstruction(
        validation["voltage"], validation, state, spline, model, s_ind,
        train["soh"],
    )
    metrics["dataset"] = dataset
    return metrics, {
        "spline": spline,
        "model": model,
        "s_ind": float(s_ind),
        "selection": selection,
        "manifold": manifold,
        "train": train,
        "validation": validation,
        "state": state,
    }


# --------------------------------------------------------------------------
# formal artifact locations
# --------------------------------------------------------------------------
def reconstruction_archive(dataset: str, condition: str, model: str = "batterydit",
                           seed: int = 2021, results_root: Path | None = None) -> Path:
    """Frozen `reconstructions_<condition>.npz` written by the formal run."""
    root = Path(results_root) if results_root else RESULTS_ROOT
    setting = f"Reconstruction_seed{seed}_{model}_{dataset}_0"
    path = root / "Reconstruction" / dataset / setting / f"reconstructions_{condition}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_reconstruction(dataset: str, condition: str, model: str = "batterydit",
                        seed: int = 2021, results_root: Path | None = None) -> dict:
    """Paired clean / corrupted / reconstructed voltages for one condition."""
    with np.load(reconstruction_archive(dataset, condition, model, seed, results_root),
                 allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.keys()}


def population_individual_run(dataset: str, seed: int, results_root: Path | None = None) -> Path:
    """Directory of the formal PopulationIndividual result for one system."""
    root = Path(results_root) if results_root else RESULTS_ROOT
    directory = root / "PopulationIndividual" / dataset / \
        f"PopulationIndividual_seed{seed}_batterydit_trb_{dataset}_0"
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return directory


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"DATA_PROBLEM: {message}")
