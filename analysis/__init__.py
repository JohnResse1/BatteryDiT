"""Analysis layer: formal results -> statistics tables in `outputs/analysis/`.

Modules here must not import matplotlib.  They read `results/`, compute
battery-balanced statistics with whole-battery uncertainty, and write
machine-readable tables for the figure layer to consume.
"""
