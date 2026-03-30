# Redback-JAX

[![Documentation Status](https://readthedocs.org/projects/redback-jax/badge/?version=latest)](https://redback-jax.readthedocs.io/en/latest/)
[![Tests](https://github.com/nikhil-sarin/redback-jax/workflows/Tests/badge.svg)](https://github.com/nikhil-sarin/redback-jax/actions)
[![codecov](https://codecov.io/gh/nikhil-sarin/redback-jax/branch/main/graph/badge.svg)](https://codecov.io/gh/nikhil-sarin/redback-jax)

A lightweight JAX-only rewrite of [redback](https://github.com/nikhil-sarin/redback) for electromagnetic transient modeling and Bayesian inference, designed to run efficiently on GPUs and TPUs in float32.

## Overview

Redback-JAX reimplements redback's analytical transient models in JAX, using log10-space arithmetic throughout to stay float32-safe on GPU hardware. All bolometric functions return `log10(L)` rather than linear luminosities (which exceed the float32 maximum of ~3.4×10³⁸ erg/s). The full spectra pipeline — photosphere, blackbody SED, and bandflux integration — also operates in log10 space end-to-end.

## Features

- **Float32-safe physics**: All models operate in log10 space; no overflow on GPU even for luminosities ~10⁴⁵ erg/s
- **JIT-compiled and differentiable**: Every model is decorated with `@jax.jit`; gradients flow through the full pipeline via `jax.grad`
- **`vmap`-based diffusion integrals**: Arnett-style diffusion uses `jax.vmap` over time points with log-mirror quadrature nodes
- **Spectra pipeline**: `make_spectra_model(bolometric_fn)` wraps any bolometric model to produce time × wavelength spectra for bandflux/magnitude comparison
- **Inference-ready**: Compatible with BlackJAX (NUTS, SMC) and flowMC (MALA/NF) samplers

## Models

### Bolometric models (return `log10_lbol` in erg/s)

| Function | Physics | Reference |
|---|---|---|
| `arnett_bolometric` | Ni/Co decay + Arnett diffusion | Arnett 1982 |
| `magnetar_powered_bolometric` | Dipole spin-down + Arnett diffusion | Nicholl+ 2017 |
| `csm_interaction_bolometric` | Forward/reverse shocks + CSM diffusion | Chatzopoulos+ 2013 |
| `tde_analytical_bolometric` | t⁻⁵/³ fallback + Arnett diffusion | — |
| `shock_cooling_bolometric` | Shock-cooling envelope (n=10) | Piro 2021 |
| `shocked_cocoon_bolometric` | Shocked jet cocoon | Piro & Kollmeier 2018 |
| `metzger_kilonova_bolometric` | r-process ODE, 200 shells | Metzger 2017 |
| `magnetar_boosted_kilonova_bolometric` | r-process ODE + magnetar injection | Yu+ 2013 |

All bolometric functions return **`log10_lbol`** (log base-10 of luminosity in erg/s). This is the natural unit for GPU inference — float32 can represent log10 values for any physically realistic luminosity.

### Spectra pipeline

`make_spectra_model(bolometric_fn)` wraps any bolometric model into a full SED pipeline:

1. Calls `bolometric_fn(time, **kwargs)` → `log10_lbol`
2. Computes photospheric temperature and radius in log10 space (with temperature floor)
3. Evaluates blackbody flux density in log10 space
4. Returns `(time, lambdas, spectra)` in observer frame

## Fitting bolometric data

Since models return `log10_lbol`, fit observed bolometric luminosities in log10 space:

```python
import jax.numpy as jnp
from redback_jax.models.supernova_models import arnett_bolometric

# Observed data
log10_lbol_obs = jnp.log10(observed_lbol)   # convert once
log10_lbol_err = sigma_lbol / (observed_lbol * jnp.log(10.0))  # propagate errors

# Model prediction
log10_lbol_model = arnett_bolometric(time, f_nickel=0.5, mej=1.0,
                                      vej=10000.0, kappa=0.1, kappa_gamma=10.0)

# Gaussian log-likelihood in log10 space
log_like = -0.5 * jnp.sum(((log10_lbol_obs - log10_lbol_model) / log10_lbol_err)**2)
```

## Fitting spectra / photometry

```python
import jax; jax.config.update("jax_enable_x64", False)  # stay in float32
import jax.numpy as jnp
from redback_jax.models.supernova_models import arnett_bolometric
from redback_jax.models.spectra_model import make_spectra_model
from redback_jax.sources import PrecomputedSpectraSource

# Build spectra model
arnett_spectra = make_spectra_model(arnett_bolometric)

# Compute spectra grid once (at fixed parameters for PrecomputedSpectraSource)
output = arnett_spectra(
    redshift=0.05,
    lum_dist=7e26,          # cm (~230 Mpc)
    vej_kms=10000.0,
    temperature_floor=3000.0,
    f_nickel=0.5, mej=1.0,
    vej=10000.0, kappa=0.1, kappa_gamma=10.0,
)

# Create bandflux source
source = PrecomputedSpectraSource(
    phases=output.time,
    wavelengths=output.lambdas,
    flux_grid=output.spectra,
)

# Prepare bridges for fast multi-band fitting
bridges, band_to_idx = source.prepare_bridges(['ztfg', 'ztfr', 'ztfi'])
band_indices = jnp.array([band_to_idx[b] for b in observed_bands])

@jax.jit
def log_likelihood(params):
    out = arnett_spectra(
        redshift=0.05, lum_dist=7e26, vej_kms=10000.0, temperature_floor=3000.0,
        **params,
    )
    src = PrecomputedSpectraSource(phases=out.time, wavelengths=out.lambdas, flux_grid=out.spectra)
    model_mag = src.bandmag({'amplitude': 1.0}, None, obs_times,
                             band_indices=band_indices, bridges=bridges,
                             unique_bands=['ztfg', 'ztfr', 'ztfi'])
    return -0.5 * jnp.sum(((obs_mags - model_mag) / obs_errs)**2)
```

## Parameter conventions

Some parameters changed from the original redback package for float32 safety:

| Model | Old parameter | New parameter | Reason |
|---|---|---|---|
| `tde_analytical_bolometric` | `l0` (erg/s, ~10⁴³) | `log10_l0` | Linear value overflows float32 |
| `shock_cooling_bolometric` | `mass` (Msun), `radius` (cm), `energy` (erg) | `log10_mass`, `log10_radius`, `log10_energy` | Intermediate products overflow float32 |

All other parameter names match redback exactly.

## Float32 design

Physical luminosities of transients (~10³⁸–10⁴⁵ erg/s) exceed float32 max (~3.4×10³⁸). Redback-JAX solves this by:

- Storing all engine luminosities as `log10(L)` throughout
- Using log-sum-exp for combining decay terms (Ni/Co engine)
- Normalising ODE state variables by a scale factor (`E_scale`) in the kilonova scan
- Computing prefactors in log10 before any exponentiation
- Keeping the blackbody SED, temperature, and photospheric radius all in log10 space

The only step that materialises linear values is the final bandflux integral over the SED — where the flux densities (~10⁻²⁰ erg/s/cm²/Å) are comfortably within float32 range.

## Installation

```bash
git clone https://github.com/nikhil-sarin/redback-jax.git
cd redback-jax
pip install -e .
```

### Dependencies

**Python 3.12+** required.

Core: `jax`, `numpy`, `scipy`, `pandas`, `matplotlib`, `astropy`, `wcosmo`

Optional (inference): `blackjax`, `flowmc`, `optax`

Optional (bandflux): `jax-bandflux` (`jax_supernovae`)

## Related Projects

- [redback](https://github.com/nikhil-sarin/redback) — the original full-featured package
- [fiestaEM](https://github.com/ThibeauWouters/fiestaEM) — similar JAX-based transient inference framework
- [JAX](https://github.com/google/jax) — the underlying numerical computing library

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

## Acknowledgments

Based on the original [redback](https://github.com/nikhil-sarin/redback) package. Please cite the redback paper if you use this software.
