"""
Plotting utilities for redback-jax.
"""

import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import jax.numpy as jnp
from typing import Union, Optional


def setup_plot_style():
    """Set up a nice default plotting style."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def create_subplot_grid(n_plots: int, ncols: int = 2) -> tuple:
    """
    Create a subplot grid for multiple plots.

    Parameters
    ----------
    n_plots : int
        Number of plots needed
    ncols : int, optional
        Number of columns in the grid

    Returns
    -------
    tuple
        (fig, axes) where axes is array of matplotlib.axes.Axes
    """
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

    # Make axes always iterable
    if n_plots == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    return fig, axes


def plot_errorbar_jax(
    axes: matplotlib.axes.Axes,
    x: Union[np.ndarray, jnp.ndarray],
    y: Union[np.ndarray, jnp.ndarray],
    xerr: Union[np.ndarray, jnp.ndarray] = None,
    yerr: Union[np.ndarray, jnp.ndarray] = None,
    **kwargs,
) -> None:
    """
    Plot errorbar with JAX array support.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Axes to plot on
    x, y : array_like
        Data coordinates
    xerr, yerr : array_like, optional
        Error bar sizes
    **kwargs
        Additional arguments passed to errorbar
    """
    # Convert JAX arrays to numpy
    x = np.asarray(x)
    y = np.asarray(y)

    if xerr is not None:
        xerr = np.asarray(xerr)
    if yerr is not None:
        yerr = np.asarray(yerr)

    axes.errorbar(x, y, xerr=xerr, yerr=yerr, **kwargs)


def plot_bands_separately(transient, model_func=None, model_params=None, **kwargs):
    """
    Plot lightcurve data separated by photometric bands.

    Parameters
    ----------
    transient : Transient
        Transient object with band information
    model_func : callable, optional
        Model function to overplot
    model_params : dict, optional
        Model parameters
    **kwargs
        Additional plotting arguments

    Returns
    -------
    tuple
        (fig, axes) of the created plots
    """
    if transient.bands is None:
        raise ValueError("Transient object must have band information")

    unique_bands = np.unique(transient.bands)
    n_bands = len(unique_bands)

    fig, axes = create_subplot_grid(n_bands, ncols=min(3, n_bands))

    colors = plt.cm.tab10(np.linspace(0, 1, n_bands))

    for i, band in enumerate(unique_bands):
        ax = axes[i] if n_bands > 1 else axes

        # Filter data for this band
        mask = transient.bands == band
        time_band = transient.time[mask]
        y_band = transient.y[mask]
        y_err_band = transient.y_err[mask] if transient.y_err is not None else None

        # Plot data
        if y_err_band is not None:
            plot_errorbar_jax(
                ax,
                time_band,
                y_band,
                yerr=y_err_band,
                fmt="o",
                color=colors[i],
                label=f"{band} data",
            )
        else:
            ax.scatter(
                np.asarray(time_band),
                np.asarray(y_band),
                color=colors[i],
                label=f"{band} data",
            )

        # Plot model if provided
        if model_func is not None and model_params is not None:
            t_min, t_max = float(jnp.min(time_band)), float(jnp.max(time_band))
            dt = t_max - t_min
            t_model = jnp.linspace(t_min - 0.1 * dt, t_max + 0.1 * dt, 200)

            # Add band information to model parameters if needed
            model_params_band = model_params.copy()
            if "band" in model_func.__code__.co_varnames:
                model_params_band["band"] = band

            y_model = model_func(t_model, **model_params_band)
            ax.plot(
                np.asarray(t_model),
                np.asarray(y_model),
                color=colors[i],
                alpha=0.8,
                label=f"{band} model",
            )

        ax.set_xlabel(transient.xlabel)
        ax.set_ylabel(transient.ylabel)
        ax.set_title(f"{transient.name} - {band}")
        ax.legend()

        if transient.data_mode == "magnitude":
            ax.invert_yaxis()

    plt.tight_layout()
    return fig, axes
