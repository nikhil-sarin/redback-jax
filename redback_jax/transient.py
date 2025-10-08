"""
JAX-friendly transient classes for electromagnetic transient analysis.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Union

import jax.numpy as jnp
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Spectrum:
    """
    A JAX-friendly class to store and analyze spectral data.
    """

    def __init__(
        self,
        wavelength: Union[np.ndarray, jnp.ndarray],
        flux_density: Union[np.ndarray, jnp.ndarray],
        flux_density_err: Union[np.ndarray, jnp.ndarray] = None,
        time: Optional[str] = None,
        name: str = "",
        **kwargs,
    ) -> None:
        """
        Initialize a Spectrum object.

        Parameters
        ----------
        wavelength : array_like
            Wavelength array (preferably in Angstroms)
        flux_density : array_like
            Flux density values (e.g., erg/s/cm²/Å)
        flux_density_err : array_like, optional
            Flux density uncertainties
        time : str, optional
            Time label for the spectrum
        name : str, optional
            Name identifier for the spectrum
        """
        self.wavelength = jnp.asarray(wavelength)
        self.flux_density = jnp.asarray(flux_density)
        self.flux_density_err = (
            jnp.asarray(flux_density_err) if flux_density_err is not None else None
        )
        self.time = time
        self.name = name
        self.data_mode = "spectrum"

        # Validate data
        if len(self.wavelength) != len(self.flux_density):
            raise ValueError(
                "Wavelength and flux_density arrays must have the same length"
            )

        if self.flux_density_err is not None and len(self.flux_density_err) != len(
            self.flux_density
        ):
            raise ValueError(
                "flux_density_err must have the same length as flux_density"
            )

    @property
    def xlabel(self) -> str:
        """X-axis label for plotting."""
        return r"Wavelength [$\mathrm{\AA}$]"

    @property
    def ylabel(self) -> str:
        """Y-axis label for plotting."""
        return r"Flux density [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]"

    def plot_data(
        self,
        axes: Optional[matplotlib.axes.Axes] = None,
        color: str = "blue",
        alpha: float = 0.7,
        show_errors: bool = True,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """
        Plot the spectrum data.

        Parameters
        ----------
        axes : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure
        color : str, optional
            Color for the data points
        alpha : float, optional
            Transparency level
        show_errors : bool, optional
            Whether to show error bars
        **kwargs
            Additional plotting arguments

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the plot
        """
        if axes is None:
            fig, axes = plt.subplots(figsize=(10, 6))

        # Convert JAX arrays to numpy for plotting
        wavelength = np.asarray(self.wavelength)
        flux_density = np.asarray(self.flux_density)

        if show_errors and self.flux_density_err is not None:
            flux_err = np.asarray(self.flux_density_err)
            axes.errorbar(
                wavelength,
                flux_density,
                yerr=flux_err,
                color=color,
                alpha=alpha,
                **kwargs,
            )
        else:
            axes.plot(wavelength, flux_density, color=color, alpha=alpha, **kwargs)

        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)

        if self.time:
            axes.set_title(f"{self.name} - {self.time}")
        elif self.name:
            axes.set_title(self.name)

        return axes


class Transient:
    """
    A JAX-friendly transient class for electromagnetic transient data analysis.

    This is a simplified version of the redback Transient class, optimized for
    JAX arrays and focused on data loading and plotting functionality.
    """

    DATA_MODES = ["luminosity", "flux", "flux_density", "magnitude", "counts"]

    def __init__(
        self,
        time: Union[np.ndarray, jnp.ndarray] = None,
        time_err: Union[np.ndarray, jnp.ndarray] = None,
        y: Union[np.ndarray, jnp.ndarray] = None,
        y_err: Union[np.ndarray, jnp.ndarray] = None,
        data_mode: str = "flux",
        name: str = "",
        redshift: float = 0.0,
        bands: Union[np.ndarray, list] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a Transient object.

        Parameters
        ----------
        time : array_like
            Time values (observer frame by default)
        time_err : array_like, optional
            Time uncertainties
        y : array_like
            Dependent variable values (flux, magnitude, etc.)
        y_err : array_like, optional
            Uncertainties in dependent variable
        data_mode : str, optional
            Type of data ('flux', 'magnitude', 'luminosity', etc.)
        name : str, optional
            Name of the transient
        redshift : float, optional
            Redshift of the transient
        bands : array_like, optional
            Photometric bands or filters
        **kwargs
            Additional attributes
        """

        # Validate data_mode
        if data_mode not in self.DATA_MODES:
            raise ValueError(f"data_mode must be one of {self.DATA_MODES}")

        self.data_mode = data_mode
        self.name = name
        self.redshift = redshift

        # Convert inputs to JAX arrays
        if time is not None:
            self.time = jnp.asarray(time)
            self.y = jnp.asarray(y)

            # Validate data consistency
            if len(self.time) != len(self.y):
                raise ValueError("Time and y arrays must have the same length")

            self.time_err = jnp.asarray(time_err) if time_err is not None else None
            self.y_err = jnp.asarray(y_err) if y_err is not None else None

            if self.time_err is not None and len(self.time_err) != len(self.time):
                raise ValueError("time_err must have the same length as time")
            if self.y_err is not None and len(self.y_err) != len(self.y):
                raise ValueError("y_err must have the same length as y")
        else:
            self.time = None
            self.y = None
            self.time_err = None
            self.y_err = None

        self.bands = bands

        # Store additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_data_file(
        cls,
        filename: str,
        data_mode: str = "flux",
        time_col: str = "time",
        y_col: str = None,
        time_err_col: str = None,
        y_err_col: str = None,
        **kwargs,
    ) -> "Transient":
        """
        Create a Transient object from a data file.

        Parameters
        ----------
        filename : str
            Path to the data file (CSV, ASCII, etc.)
        data_mode : str, optional
            Type of data being loaded
        time_col : str, optional
            Name of time column
        y_col : str, optional
            Name of y-data column. If None, inferred from data_mode
        time_err_col : str, optional
            Name of time error column
        y_err_col : str, optional
            Name of y-data error column
        **kwargs
            Additional arguments passed to Transient constructor

        Returns
        -------
        Transient
            Transient object with loaded data
        """

        # Load data based on file extension
        if filename.endswith(".csv"):
            data = pd.read_csv(filename)
        else:
            # Try to read as whitespace-separated
            try:
                data = pd.read_csv(filename, delim_whitespace=True)
            except Exception as e:
                raise ValueError(f"Could not read file {filename}: {e}")

        # Infer y column name if not provided
        if y_col is None:
            y_col = data_mode
            if y_col not in data.columns:
                # Try common alternatives
                alternatives = {
                    "flux": ["flux", "Flux", "f", "F"],
                    "magnitude": ["mag", "magnitude", "m", "Magnitude"],
                    "luminosity": ["lum", "luminosity", "L", "Luminosity"],
                    "flux_density": ["flux_density", "f_nu", "flux_dens"],
                }

                found = False
                for alt in alternatives.get(data_mode, [data_mode]):
                    if alt in data.columns:
                        y_col = alt
                        found = True
                        break

                if not found:
                    raise ValueError(
                        f"Could not find {data_mode} column in data. "
                        f"Available columns: {list(data.columns)}"
                    )

        # Extract data
        time = data[time_col].values
        y = data[y_col].values

        time_err = (
            data[time_err_col].values
            if time_err_col and time_err_col in data.columns
            else None
        )
        y_err = (
            data[y_err_col].values if y_err_col and y_err_col in data.columns else None
        )

        # Get bands if available
        bands = data["band"].values if "band" in data.columns else None

        return cls(
            time=time,
            time_err=time_err,
            y=y,
            y_err=y_err,
            data_mode=data_mode,
            bands=bands,
            **kwargs,
        )

    @property
    def xlabel(self) -> str:
        """X-axis label for plotting."""
        return "Time"

    @property
    def ylabel(self) -> str:
        """Y-axis label for plotting."""
        ylabel_dict = {
            "luminosity": r"Luminosity [erg s$^{-1}$]",
            "magnitude": r"Magnitude",
            "flux": r"Flux [erg cm$^{-2}$ s$^{-1}$]",
            "flux_density": r"Flux density [mJy]",
            "counts": r"Counts",
        }
        return ylabel_dict.get(self.data_mode, "Y")

    def plot_data(
        self,
        axes: Optional[matplotlib.axes.Axes] = None,
        color: str = "blue",
        alpha: float = 0.7,
        show_errors: bool = True,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """
        Plot the transient lightcurve data.

        Parameters
        ----------
        axes : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure
        color : str, optional
            Color for the data points
        alpha : float, optional
            Transparency level
        show_errors : bool, optional
            Whether to show error bars
        **kwargs
            Additional plotting arguments

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the plot
        """
        if self.time is None:
            raise ValueError(
                "No data loaded. Use from_data_file() or provide data in constructor."
            )

        if axes is None:
            fig, axes = plt.subplots(figsize=(10, 6))

        # Convert JAX arrays to numpy for plotting
        time = np.asarray(self.time)
        y = np.asarray(self.y)

        if show_errors and self.y_err is not None:
            y_err = np.asarray(self.y_err)
            axes.errorbar(
                time, y, yerr=y_err, fmt="o", color=color, alpha=alpha, **kwargs
            )
        else:
            axes.scatter(time, y, color=color, alpha=alpha, **kwargs)

        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)

        if self.name:
            axes.set_title(self.name)

        # Invert y-axis for magnitude data
        if self.data_mode == "magnitude":
            axes.invert_yaxis()

        return axes

    def plot_model(
        self,
        model_func: callable,
        model_params: Dict[str, Any],
        time_range: tuple = None,
        axes: Optional[matplotlib.axes.Axes] = None,
        color: str = "red",
        alpha: float = 0.8,
        label: str = "Model",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """
        Plot a model over the data.

        Parameters
        ----------
        model_func : callable
            Model function that takes time and parameters
        model_params : dict
            Dictionary of model parameters
        time_range : tuple, optional
            (min_time, max_time) for model evaluation
        axes : matplotlib.axes.Axes, optional
            Axes to plot on
        color : str, optional
            Color for model curve
        alpha : float, optional
            Transparency level
        label : str, optional
            Label for model curve
        **kwargs
            Additional plotting arguments

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the plot
        """
        if axes is None:
            axes = self.plot_data()

        # Determine time range
        if time_range is None:
            if self.time is not None:
                t_min, t_max = float(jnp.min(self.time)), float(jnp.max(self.time))
                # Extend range slightly
                dt = t_max - t_min
                t_min -= 0.1 * dt
                t_max += 0.1 * dt
            else:
                t_min, t_max = 0, 100
        else:
            t_min, t_max = time_range

        # Create model time array
        t_model = jnp.linspace(t_min, t_max, 1000)

        # Evaluate model
        y_model = model_func(t_model, **model_params)

        # Plot model
        axes.plot(
            np.asarray(t_model),
            np.asarray(y_model),
            color=color,
            alpha=alpha,
            label=label,
            **kwargs,
        )

        axes.legend()
        return axes

    def __repr__(self) -> str:
        """String representation of the Transient object."""
        n_points = len(self.time) if self.time is not None else 0
        return (
            f"Transient(name='{self.name}', data_mode='{self.data_mode}', "
            f"n_points={n_points}, redshift={self.redshift})"
        )
