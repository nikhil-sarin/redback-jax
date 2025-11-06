"""
Example: Converting Arnett model spectra to multi-band magnitudes

This example demonstrates the full workflow:
1. Generate spectra using the arnett_model
2. Convert spectra to multi-band light curves using the PrecomputedSpectraSource
3. Plot the results

No additional integration code needed - everything uses JAX-bandflux directly!
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from redback_jax.sources import PrecomputedSpectraSource


def main():
    print("=" * 60)
    print("Arnett Model to Multi-band Magnitudes Example")
    print("=" * 60)
    print()

    # Create source directly from Arnett model (all in one step!)
    # Note: The new interface automatically generates time/wavelength grids
    print("Creating source from Arnett model...")
    source = PrecomputedSpectraSource.from_arnett_model(
        f_nickel=0.1,
        mej=1.4,
        vej=5000,
        kappa=0.07,
        kappa_gamma=0.1,
        temperature_floor=2000,
        redshift=0.01
    )
    print(f"  Source created with {len(source.phases)} time points")
    print(f"  Time range: {source.phases[0]:.1f} - {source.phases[-1]:.1f} days")
    print(f"  Wavelength range: {source.wavelengths[0]:.1f} - {source.wavelengths[-1]:.1f} Å")
    print()

    # Calculate multi-band light curves using new functional API
    print("Calculating multi-band magnitudes...")
    phases = jnp.linspace(5, 50, 50)
    bands = ['bessellb', 'bessellv', 'bessellr', 'besselli']

    # Define amplitude parameter (1.0 = use model as-is)
    params = {'amplitude': 1.0}

    # Calculate magnitudes for each band
    lightcurves = {}
    for band in bands:
        lightcurves[band] = source.bandmag(params, band, phases, magsys='ab')

    print(f"  Generated light curves for {len(bands)} bands")
    print()

    # Plot the results
    print("Plotting results...")

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Example spectrum at peak
    ax1 = axes[0]
    peak_idx = jnp.argmax(-lightcurves['bessellv'])  # Find brightest time
    peak_phase = float(phases[peak_idx])

    # Find closest phase index in the model grid
    phase_idx = jnp.argmin(jnp.abs(source.phases - peak_phase))
    spectrum = source.flux_grid[phase_idx, :]

    ax1.plot(source.wavelengths, spectrum, 'k-', linewidth=2)
    ax1.set_xlabel('Wavelength (Å)', fontsize=12)
    ax1.set_ylabel(r'$F_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=12)
    ax1.set_title(f'Spectrum at t = {float(source.phases[phase_idx]):.1f} days (near peak)', fontsize=14)
    ax1.set_xlim(3000, 10000)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-16, 1e-14)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Multi-band light curves
    ax2 = axes[1]
    colors = {'bessellb': 'blue', 'bessellv': 'green', 'bessellr': 'red', 'besselli': 'darkred'}

    for band, mags in lightcurves.items():
        color = colors.get(band, 'black')
        ax2.plot(phases, mags, 'o-', color=color, label=band.upper(),
                markersize=4, linewidth=2)

    ax2.set_xlabel('Observer Frame Time (days)', fontsize=12)
    ax2.set_ylabel('AB Magnitude', fontsize=12)
    ax2.set_title('Arnett Model Light Curves (z=0.01)', fontsize=14)
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
        peak_time = phases[jnp.argmin(mags)]
        print(f"{band.upper():10s}: Peak magnitude = {peak_mag:.2f} at t = {peak_time:.1f} days")
    print()

    print("Done!")
    print()


if __name__ == "__main__":
    main()
