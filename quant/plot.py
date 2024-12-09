import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cycler import cycler


def set_plot_options(version="bold"):
    if version == "bold":
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
        plt.rcParams["lines.linewidth"] = 1
        plt.rcParams["lines.color"] = "black"
        plt.rcParams["legend.edgecolor"] = "black"
        plt.rcParams["legend.fancybox"] = False
        plt.rcParams["legend.framealpha"] = 1
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.color"] = "black"
        plt.rcParams["grid.alpha"] = "0.2"
        plt.rcParams["grid.linewidth"] = 1
        plt.rcParams["grid.linestyle"] = "--"


def corr(df, **kwargs):
    corr = df.corr()
    ax = sns.heatmap(
        corr, annot=True, fmt=".0%", cmap="coolwarm", mask=np.triu(corr), **kwargs
    )
    ax.grid(False)
    return ax
