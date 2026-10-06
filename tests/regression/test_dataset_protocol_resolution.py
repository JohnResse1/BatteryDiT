"""Regression contract for dataset-specific physical-voltage settings."""
from __future__ import annotations

from types import SimpleNamespace

from main import configure_dataset, configure_model_protocol


def main() -> None:
    mit = SimpleNamespace(
        data="mit", po_type="HUST", v_min=1.5, v_max=4.2,
        ocv_table_path="", recon_model="csdi", diffusion_steps=50,
        sampling_steps=1,
    )
    configure_dataset(mit)
    configure_model_protocol(mit)
    assert (mit.po_type, mit.v_min, mit.v_max) == ("MIT", 2.0, 3.6)
    assert mit.ocv_table_path == "models/ocv/mit.npz"
    assert mit.sampling_steps == mit.diffusion_steps == 50

    hust = SimpleNamespace(
        data="hust", po_type="MIT", v_min=2.0, v_max=3.6,
        ocv_table_path="", recon_model="batterydit", diffusion_steps=50,
        sampling_steps=1,
    )
    configure_dataset(hust)
    configure_model_protocol(hust)
    assert (hust.po_type, hust.v_min, hust.v_max) == ("HUST", 2.0, 3.6)
    assert hust.sampling_steps == 1

    expected = {
        "mich": ("NCM", 3.0, 4.2),
        "sdu": ("NCM", 2.5, 4.2),
        "stanford_2": ("NCM", 3.0, 4.4),
        "tju_nca_45c": ("NCA", 2.5, 4.2),
        "tju_ncm_45c": ("NCM", 2.5, 4.2),
        "tju_ncm_nca_25c": ("NCM_NCA", 2.5, 4.2),
    }
    for dataset, physical_settings in expected.items():
        args = SimpleNamespace(
            data=dataset, po_type="MIT", v_min=0.0, v_max=1.0,
            ocv_table_path="", recon_model="batterydit", diffusion_steps=50,
            sampling_steps=1,
        )
        configure_dataset(args)
        configure_model_protocol(args)
        assert (args.po_type, args.v_min, args.v_max) == physical_settings
        assert args.ocv_table_path == f"models/ocv/{dataset}.npz"
    print("dataset-specific voltage protocol passed")


if __name__ == "__main__":
    main()
