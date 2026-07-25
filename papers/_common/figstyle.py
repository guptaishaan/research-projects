"""Shared matplotlib styling for every figure in the paper set.

Keeps figures typographically consistent with the LaTeX body (Times-family serif,
STIX maths) and embeds TrueType rather than Type 3 outlines so the resulting PDFs
pass publisher font checks.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIF = ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes",
         "Liberation Serif", "DejaVu Serif"]

PARAMS = {
    "font.family": "serif",
    "font.serif": SERIF,
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # embed TrueType (42) instead of Type 3 outlines
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def use():
    """Apply the shared style to the global matplotlib state."""
    plt.rcParams.update(PARAMS)
    return plt
