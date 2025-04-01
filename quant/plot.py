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
        mpl.rcParams["axes.prop_cycle"] = cycler(
            color=[
                "blue",
                "red",
                "green",
                "purple",
                "orange",
                "brown",
                "cyan",
                "magenta",
                "gold",
                "navy",
                "darkorange",
                "olive",
                "royalblue",
                "crimson",
                "darkgreen",
            ]
        )

        # Set line properties
        plt.rcParams.update({"lines.linewidth": 1, "lines.color": "black"})

        # Configure legend appearance
        plt.rcParams.update(
            {
                "legend.edgecolor": "black",
                "legend.fancybox": False,
                "legend.framealpha": 1,
            }
        )

        # Enable and style grid
        plt.rcParams.update(
            {
                "axes.grid": True,
                "grid.color": "gray",
                "grid.alpha": 0.3,
                "grid.linewidth": 1,
                "grid.linestyle": "--",
            }
        )
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
