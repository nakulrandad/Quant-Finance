import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cycler import cycler
from matplotlib.ticker import FuncFormatter


def set_plot_options(version="bold"):
    if version == "bold":
        # Define a consistent color cycle for plots
        plt.rcParams["axes.prop_cycle"] = cycler(
            color=[
                "blue",  # Primary blue
                "red",  # Primary red
                "green",  # Primary green
                "purple",  # Primary purple
                "orange",  # Primary orange
                "brown",  # Primary brown
                "pink",  # Primary pink
                "gray",  # Primary gray
                "olive",  # Olive green
                "cyan",  # Primary cyan
                "magenta",  # Primary magenta
                "yellow",  # Primary yellow
                "navy",  # Dark blue
                "maroon",  # Dark red
                "lime",  # Light green
                "teal",  # Blue-green
                "aqua",  # Light blue
            ]
        )

        # Set line properties
        plt.rcParams.update({
            "lines.linewidth": 1,
            "lines.color": "black",
            "lines.markersize": 4,
            "lines.markeredgewidth": 1,
        })

        # Configure legend appearance
        plt.rcParams.update({
            "legend.edgecolor": "black",
            "legend.fancybox": False,
            "legend.framealpha": 1,
            "legend.loc": "best",
            "legend.fontsize": 10,
        })

        # Enable and style grid
        plt.rcParams.update({
            "axes.grid": True,
            "grid.color": "gray",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "grid.linestyle": "--",
        })

        # Additional time series specific settings
        plt.rcParams.update({
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        })

        # Force matplotlib to use the settings
        plt.style.use('default')
        plt.rcParams.update(plt.rcParamsDefault)
    return None


def heatmap(corr, **kwargs):
    mask = np.diag(np.ones(corr.shape[0], dtype=bool))
    ax = sns.heatmap(corr, annot=True, fmt=".0%", cmap="coolwarm", mask=mask, **kwargs)
    for i in range(len(corr.columns)):
        ax.add_patch(
            patches.Rectangle(
                (i, i), 1, 1, fill=True, edgecolor="none", lw=0, facecolor="lightgrey"
            )
        )
        ax.text(i + 0.5, i + 0.5, "100%", color="grey", ha="center", va="center")
    ax.grid(False)
    return ax


def set_yaxis_percent(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
    return ax
