import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt

def set_plot_options(version="bold"):
    if version == "bold":
        mpl.rcParams['axes.prop_cycle'] = cycler(color=[
            'blue', 'red', 'green', 'purple',
            'orange', 'brown', 'cyan', 'magenta',
            'gold', 'navy', 'darkorange', 'olive',
            'royalblue', 'crimson', 'darkgreen'
        ])
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['lines.color'] = 'black'
        plt.rcParams['legend.edgecolor'] = 'black'
        plt.rcParams['legend.fancybox'] = False
        plt.rcParams['legend.framealpha'] = 1
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.color'] = 'black'
        plt.rcParams['grid.alpha'] = '0.2'
        plt.rcParams['grid.linewidth'] = 1
        plt.rcParams['grid.linestyle'] = "--"