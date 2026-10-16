"""Reproduce Supplementary Fig. S7 from the frozen R4 plotting assets."""
from __future__ import annotations

from figures.figS7.plot import draw


if __name__ == "__main__":
    outputs = draw()
    print("FigS7 written:")
    for key in ("png", "pdf", "svg"):
        print(f"{key}: {outputs[key]}")
    print("qc: figures/figS7/figS7_qc.md")
