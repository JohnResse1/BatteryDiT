"""Shared plotting style for the reconstructed manuscript figures.

The palette and rcParams are kept in this module so every figure uses the
same publication settings.
"""
from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MM = 1 / 25.4
NC_WIDTH = 179.0 * MM      # Nature Communications double-column width
CD = 183 * MM              # full/double-column width used in the notebooks
SW = 89 * MM               # single-column width
FONT_FAMILY = 'Arial'

# ---------------------------------------------------------------------------
# rcParams — verbatim from Fig_main.ipynb cell 0
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 6,
    'axes.labelsize': 6,
    'axes.titlesize': 6.5,
    'xtick.labelsize': 5.5,
    'ytick.labelsize': 5.5,
    'legend.fontsize': 5.5,
    'axes.linewidth': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'legend.frameon': False,
})

# ---------------------------------------------------------------------------
# Unified color system — all from CPLATE (verbatim)
# ---------------------------------------------------------------------------
CPLATE = ['#2166AC', '#3275B4', '#4483BB', '#5693C4', '#67A1CB',
          '#7AB1D3', '#8BBFDB', '#9CC2D4', '#ABBDC6', '#BCB7B7',
          '#CAB3A9', '#DBAD9A', '#EAA88C', '#F2A17F', '#EE9677',
          '#E98B6E', '#E48066', '#DF755D', '#DB6B55', '#D6604D']

# Reconstruction-model colours (RC in Fig_main cell 0)
RC = OrderedDict([
    ('BatteryDiT',  CPLATE[0]),
    ('CSDI',        CPLATE[5]),
    ('Transformer', CPLATE[10]),
    ('Kalman',      CPLATE[15]),
])
# Bar-chart method colours (BMC in Fig_main cell 0)
BMC = OrderedDict([
    ('batterydit_trb',   CPLATE[0]),
    ('kalman_trb',       CPLATE[2]),
    ('csdi_trb',         CPLATE[4]),
    ('transformer_trb',  CPLATE[6]),
    ('raw_tcn',          CPLATE[9]),
    ('raw_trb',          CPLATE[14]),
])
# Legacy methods for the violin panel (BMC_OLD in Fig_main cell 0)
BMC_OLD = OrderedDict([
    ('batterydit_trb_novelocity',  CPLATE[0]),
    ('transformer_trb_novelocity', CPLATE[4]),
    ('tcn_raw',                    CPLATE[8]),
    ('trb_novelocity_raw',         CPLATE[12]),
    ('lstm_raw',                   CPLATE[16]),
])
# SI models (MC in Fig_SI cell 0)
MC = {m: CPLATE[i * 5] for i, m in
      enumerate(['BatteryDiT', 'CSDI', 'Transformer', 'Kalman'])}

# Semantic colours (verbatim from the notebooks)
C_CLEAN = CPLATE[0]
C_CORRUPTED = CPLATE[19]
C_RECONSTRUCTED = CPLATE[4]
C_REFERENCE = '#111111'
C_INPUT = '#999999'
C_BLUE = '#2878B5'
C_SHRINK = '#3B82B8'
C_MISALIGN = '#E59A36'
C_AMPLIFY = '#C94C4C'
C_GREY = '#C9CDD2'
C_DARK = '#222222'

# Colormaps (verbatim)
PCMAP = LinearSegmentedColormap.from_list(
    'nc_blue', ['#1A4778', '#3A6798', '#5A87B8', '#8AB0D8', '#BBD0E8'], N=256)
SOH_CMAP = LinearSegmentedColormap.from_list(
    'cplate_soh', [CPLATE[19], CPLATE[15], CPLATE[10], CPLATE[5], CPLATE[0]], N=256)
CMAP_POP = LinearSegmentedColormap.from_list(
    'population_error', ['#F7FBFF', '#BED7E9', '#4E91C2', '#174A78'])
CMAP_IND = LinearSegmentedColormap.from_list(
    'individual_error', ['#FFF8F0', '#F7CFAC', '#E78A56', '#A83032'])

SEV_COLORS = ['#DCE7F5', '#9DC3E6', '#5693C4', '#2166AC', '#1A4778']

# Backwards-compatible aliases (do not use in new code)
COLORS = RC
FIGURE_WIDTH = CD


def style_axis(ax, grid_axis: str = 'both'):
    """Reference axis treatment (verbatim from Fig_main cell 0)."""
    ax.tick_params(pad=1)
    ax.grid(True, axis=grid_axis, alpha=0.10, lw=0.3)
    ax.set_axisbelow(True)
    return ax


def apply_nc_style() -> None:
    """Redundant alias: the module-level `plt.rcParams.update` above already
    applies the manuscript style at import time.  Kept so figure modules can
    call `style.apply_nc_style()` explicitly before drawing."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 6,
        'axes.labelsize': 6,
        'axes.titlesize': 6.5,
        'xtick.labelsize': 5.5,
        'ytick.labelsize': 5.5,
        'legend.fontsize': 5.5,
        'axes.linewidth': 0.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
        'legend.frameon': False,
    })


def panel_label(ax, label: str, x: float = -0.08, y: float = 1.05):
    """Bold parenthesised subfigure label (verbatim convention)."""
    text = label if str(label).startswith('(') else f'({label})'
    return ax.text(x, y, text, transform=ax.transAxes,
                   fontsize=8, fontweight='bold', va='bottom')


def apply_paper_style() -> None:
    """Backwards-compatible alias for `apply_nc_style`."""
    apply_nc_style()


class _CaseInsensitiveColorMap(OrderedDict):
    """OrderedDict with case-insensitive lookup for model colour access."""

    def __getitem__(self, key):
        if key in self:
            return OrderedDict.__getitem__(self, key)
        if isinstance(key, str):
            for existing, value in self.items():
                if existing.lower() == key.lower():
                    return value
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


COLORS = _CaseInsensitiveColorMap(RC)
