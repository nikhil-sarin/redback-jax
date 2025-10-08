#!/usr/bin/env python3
"""
Basic usage examples for redback-jax Transient class.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from redback_jax import Transient, Spectrum


def example_basic_transient():
    """Example of creating and plotting a basic transient."""
    print("Example 1: Basic Transient Creation")
    print("=" * 40)

    # Generate synthetic lightcurve data
    time = np.linspace(0, 20, 30)
    # Exponential decay with some noise
    flux = 5 * np.exp(-time / 8) + 0.2 * np.random.randn(30)
    flux_err = 0.1 * np.abs(flux) + 0.05

    # Create transient object
    transient = Transient(
        time=time,
        y=flux,
        y_err=flux_err,
        data_mode="flux",
        name="Example Transient",
        redshift=0.05,
    )

    print(f"Created transient: {transient}")
    print(f"Peak flux: {float(jnp.max(transient.y)):.3f}")
    print(f"Mean time: {float(jnp.mean(transient.time)):.3f}")

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    transient.plot_data(axes=ax, color="blue", alpha=0.7)
    plt.title("Basic Transient Example")
    plt.tight_layout()
    plt.show()

    return transient


def example_multi_band_transient():
    """Example of multi-band photometric data."""
    print("\nExample 2: Multi-band Transient")
    print("=" * 40)

    # Create multi-band data
    time_points = np.linspace(0, 15, 12)
    bands = ["g", "r", "i"]

    time = []
    magnitude = []
    mag_err = []
    band_list = []

    for band in bands:
        for t in time_points:
            time.append(t)
            band_list.append(band)

            # Different evolution for different bands
            if band == "g":
                mag = 18 + 0.3 * t + 0.1 * np.random.randn()
            elif band == "r":
                mag = 17.5 + 0.25 * t + 0.1 * np.random.randn()
            else:  # i band
                mag = 17 + 0.2 * t + 0.1 * np.random.randn()

            magnitude.append(mag)
            mag_err.append(0.05 + 0.01 * np.random.rand())

    # Create transient
    transient = Transient(
        time=np.array(time),
        y=np.array(magnitude),
        y_err=np.array(mag_err),
        data_mode="magnitude",
        bands=np.array(band_list),
        name="Multi-band Supernova",
        redshift=0.03,
    )

    print(f"Created multi-band transient: {transient}")
    print(f"Unique bands: {np.unique(transient.bands)}")

    # Plot all data together
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"g": "green", "r": "red", "i": "orange"}

    for band in bands:
        mask = transient.bands == band
        time_band = transient.time[mask]
        mag_band = transient.y[mask]
        err_band = transient.y_err[mask]

        ax.errorbar(
            np.asarray(time_band),
            np.asarray(mag_band),
            yerr=np.asarray(err_band),
            fmt="o",
            color=colors[band],
            label=f"{band} band",
            alpha=0.7,
        )

    ax.set_xlabel(transient.xlabel)
    ax.set_ylabel(transient.ylabel)
    ax.set_title(transient.name)
    ax.invert_yaxis()  # Magnitudes increase downward
    ax.legend()
    plt.tight_layout()
    plt.show()

    return transient


def example_spectrum():
    """Example of creating and plotting a spectrum."""
    print("\nExample 3: Spectrum Analysis")
    print("=" * 40)

    # Generate synthetic spectrum data
    wavelength = np.linspace(4000, 7000, 200)  # Angstroms

    # Simple blackbody-like spectrum with emission lines
    temp = 8000  # Kelvin
    flux_bb = (wavelength / 5500) ** (-4) * np.exp(-1.44e8 / (wavelength * temp))

    # Add some emission lines
    for line_wave in [4861, 5007, 6563]:  # H-beta, [OIII], H-alpha
        line_strength = 0.3 * np.exp(-0.5 * ((wavelength - line_wave) / 10) ** 2)
        flux_bb += line_strength

    # Add noise
    flux = flux_bb + 0.02 * np.random.randn(200)
    flux_err = 0.05 * np.abs(flux) + 0.01

    # Create spectrum
    spectrum = Spectrum(
        wavelength=wavelength,
        flux_density=flux,
        flux_density_err=flux_err,
        name="Example Spectrum",
        time="Day 5",
    )

    print(f"Created spectrum: {spectrum.name}")
    print(
        f"Wavelength range: {float(jnp.min(spectrum.wavelength)):.0f} - {float(jnp.max(spectrum.wavelength)):.0f} Å"
    )
    print(f"Peak flux: {float(jnp.max(spectrum.flux_density)):.3f}")

    # Plot the spectrum
    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum.plot_data(axes=ax, color="purple", alpha=0.8)
    plt.title("Example Spectrum with Emission Lines")
    plt.tight_layout()
    plt.show()

    return spectrum


def example_model_fitting():
    """Example of plotting a model over data."""
    print("\nExample 4: Model Fitting")
    print("=" * 40)

    # Create synthetic data with known model
    time = np.linspace(0, 25, 40)

    # True model parameters
    true_params = {"amplitude": 3.0, "decay_time": 8.0, "offset": 0.5}

    # Generate data with the model
    def exponential_model(t, amplitude, decay_time, offset):
        return amplitude * jnp.exp(-t / decay_time) + offset

    true_flux = exponential_model(time, **true_params)
    noisy_flux = true_flux + 0.2 * np.random.randn(len(time))
    flux_err = 0.15 * np.ones_like(noisy_flux)

    # Create transient
    transient = Transient(
        time=time,
        y=noisy_flux,
        y_err=flux_err,
        data_mode="flux",
        name="Model Fitting Example",
    )

    # Plot data with true model
    fig, ax = plt.subplots(figsize=(10, 6))
    transient.plot_data(axes=ax, color="blue", alpha=0.7, label="Data")

    # Overplot the true model
    transient.plot_model(
        model_func=exponential_model,
        model_params=true_params,
        axes=ax,
        color="red",
        label="True Model",
        linewidth=2,
    )

    plt.title("Transient Data with Exponential Model")
    plt.tight_layout()
    plt.show()

    print(f"True parameters: {true_params}")
    return transient


def example_data_from_file():
    """Example of loading data from a file."""
    print("\nExample 5: Loading Data from File")
    print("=" * 40)

    # Create a temporary data file
    import tempfile
    import os

    data_content = """time,flux,flux_err,band
0.5,4.2,0.2,g
1.0,3.8,0.18,g
1.5,3.1,0.15,g
2.0,2.5,0.12,g
0.5,3.9,0.19,r
1.0,3.6,0.17,r
1.5,3.0,0.14,r
2.0,2.4,0.11,r
0.5,3.5,0.17,i
1.0,3.2,0.15,i
1.5,2.8,0.13,i
2.0,2.2,0.10,i"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(data_content)
        temp_filename = f.name

    try:
        # Load the data
        transient = Transient.from_data_file(
            temp_filename, data_mode="flux", name="File Example"
        )

        print(f"Loaded transient: {transient}")
        print(f"Data points: {len(transient.time)}")
        print(f"Bands: {np.unique(transient.bands)}")

        # Plot the loaded data
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"g": "green", "r": "red", "i": "orange"}

        for band in np.unique(transient.bands):
            mask = transient.bands == band
            time_band = transient.time[mask]
            flux_band = transient.y[mask]
            err_band = transient.y_err[mask]

            ax.errorbar(
                np.asarray(time_band),
                np.asarray(flux_band),
                yerr=np.asarray(err_band),
                fmt="o",
                color=colors[band],
                label=f"{band} band",
                alpha=0.7,
            )

        ax.set_xlabel(transient.xlabel)
        ax.set_ylabel(transient.ylabel)
        ax.set_title("Data Loaded from File")
        ax.legend()
        plt.tight_layout()
        plt.show()

    finally:
        os.unlink(temp_filename)

    return transient


if __name__ == "__main__":
    print("Redback-JAX Transient Examples")
    print("=" * 50)

    # Run all examples
    transient1 = example_basic_transient()
    transient2 = example_multi_band_transient()
    spectrum1 = example_spectrum()
    transient3 = example_model_fitting()
    transient4 = example_data_from_file()

    print("\nAll examples completed successfully!")
    print(f"Final example transient: {transient4}")
