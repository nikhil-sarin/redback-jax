"""
Example: Converting Arnett model spectra to multi-band magnitudes

This example demonstrates the full workflow:
1. Generate spectra using the arnett_model
2. Convert spectra to multi-band light curves using JAX-bandflux filters
3. Plot the results

Requires:
    pip install jax-bandflux matplotlib
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from redback_jax.models.supernova_models import arnett_model
from redback_jax.bandflux_integration import spectra_to_lightcurves


def main():
    print("=" * 60)
    print("Arnett Model to Multi-band Magnitudes Example")
    print("=" * 60)
    print()

    # Step 1: Generate spectra using arnett_model
    print("Step 1: Generating spectra with arnett_model...")

    # Define model parameters
    f_nickel = 0.1  # Nickel mass fraction
    mej = 1.4  # Ejecta mass in solar masses
    vej = 5000  # Ejecta velocity in km/s
    kappa = 0.07  # Optical opacity
    kappa_gamma = 0.1  # Gamma-ray opacity
    temperature_floor = 5000  # Minimum temperature in K
    redshift = 0.01  # Redshift

    # Create time array
    times = jnp.linspace(0.1, 50, 50)  # days

    # Generate spectra
    output = arnett_model(
        time=times,
        f_nickel=f_nickel,
        mej=mej,
        vej=vej,
        kappa=kappa,
        kappa_gamma=kappa_gamma,
        temperature_floor=temperature_floor,
        redshift=redshift,
        output_format='spectra'
    )

    print(f"  Generated {len(output.time)} spectra")
    print(f"  Time range: {output.time[0]:.1f} - {output.time[-1]:.1f} days")
    print(f"  Wavelength range: {output.lambdas[0]:.1f} - {output.lambdas[-1]:.1f} Å")
    print(f"  Spectra shape: {output.spectra.shape}")
    print()

    # Step 2: Convert spectra to multi-band light curves
    print("Step 2: Converting spectra to multi-band magnitudes...")

    bands = ['bessellb', 'bessellv', 'bessellr', 'besselli']
    print(f"  Using bands: {bands}")

    lightcurves = spectra_to_lightcurves(
        times=output.time,
        wavelengths=output.lambdas,
        spectra=output.spectra,
        bands=bands,
        output_format='magnitude',
        zp=0.0  # AB magnitude system
    )

    print(f"  Generated light curves for {len(lightcurves)} bands")
    print()

    # Step 3: Plot the results
    print("Step 3: Plotting results...")

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Example spectrum at peak
    ax1 = axes[0]
    peak_idx = jnp.argmax(-lightcurves['bessellv'])  # Find brightest time
    peak_time = output.time[peak_idx]

    ax1.plot(output.lambdas, output.spectra[peak_idx, :], 'k-', linewidth=2)
    ax1.set_xlabel('Wavelength (Å)', fontsize=12)
    ax1.set_ylabel(r'$F_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=12)
    ax1.set_title(f'Spectrum at t = {peak_time:.1f} days (near peak)', fontsize=14)
    ax1.set_xlim(3000, 10000)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Multi-band light curves
    ax2 = axes[1]
    colors = {'bessellb': 'blue', 'bessellv': 'green', 'bessellr': 'red', 'besselli': 'darkred'}

    for band, mags in lightcurves.items():
        color = colors.get(band, 'black')
        ax2.plot(output.time, mags, 'o-', color=color, label=band.upper(),
                markersize=4, linewidth=2)

    ax2.set_xlabel('Observer Frame Time (days)', fontsize=12)
    ax2.set_ylabel('AB Magnitude', fontsize=12)
    ax2.set_title(f'Arnett Model Light Curves (z={redshift})', fontsize=14)
    ax2.invert_yaxis()
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('arnett_magnitudes.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to: arnett_magnitudes.png")
    print()

    # Print some statistics
    print("=" * 60)
    print("Light Curve Statistics:")
    print("=" * 60)
    for band, mags in lightcurves.items():
        peak_mag = jnp.min(mags)
        peak_time = output.time[jnp.argmin(mags)]
        print(f"{band.upper():10s}: Peak magnitude = {peak_mag:.2f} at t = {peak_time:.1f} days")
    print()

    print("Done!")
    print()


if __name__ == "__main__":
    main()
