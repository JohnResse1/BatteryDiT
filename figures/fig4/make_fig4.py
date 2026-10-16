"""Reproduce the current formal BatteryDiT Fig. 4.

This is the public entry point for the figure.  The implementation remains in
``plot_fig4.py`` so the redraw is an incremental revision of the existing
figure code and continues to read only the co-located canonical assets.
"""
from __future__ import annotations

from figures.fig4.plot_fig4 import draw


if __name__ == "__main__":
    outputs = draw()
    print("Fig.4 written:")
    for key in ("png", "pdf", "svg"):
        print(f"{key}: {outputs[key]}")
    print("qc: figures/fig4/fig4_qc.md")
