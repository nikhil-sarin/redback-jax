# Redback-JAX

[![Documentation Status](https://readthedocs.org/projects/redback-jax/badge/?version=latest)](https://redback-jax.readthedocs.io/en/latest/)
[![Tests](https://github.com/nikhil-sarin/redback-jax/workflows/Tests/badge.svg)](https://github.com/nikhil-sarin/redback-jax/actions)
[![codecov](https://codecov.io/gh/nikhil-sarin/redback-jax/branch/main/graph/badge.svg)](https://codecov.io/gh/nikhil-sarin/redback-jax)

A JAX-native companion to [Redback](https://github.com/nikhil-sarin/redback),
built for rapid, differentiable electromagnetic-transient analysis on CPUs,
GPUs, and TPUs.

## Overview

Redback-JAX sits alongside the broader Redback stack. It provides JAX-native
implementations of selected Redback models and composable modeling tools where
JIT compilation, automatic differentiation, vectorization, and accelerator
execution materially speed up analysis and inference. It is not intended to be
a complete rewrite or drop-in replacement for Redback: use Redback for its
full modeling and analysis ecosystem, and Redback-JAX when a supported workflow
benefits from rapid repeated evaluation or gradients.

Numerically demanding quantities are represented in log10 space where needed
to remain float32-safe. Bolometric functions return `log10(L)`, while afterglow
models return flux density or magnitude. The spectra pipeline carries
photospheric and blackbody quantities safely through to band-integrated
observables.

## Features

- **Designed for rapid analyses**: JIT-compiled kernels, vectorization, and reusable compiled likelihoods reduce repeated-evaluation cost
- **Automatic differentiation**: Supported model paths expose gradients for optimization and gradient-based inference
- **Accelerator-aware numerics**: Log-space and scaled-state implementations avoid float32 overflow where physical values require it
- **Composable afterglows**: Native Redback jet families, arbitrary angular structures and CSM profiles, pluggable radiation, continuous injection, and reverse shocks
- **`vmap`-based diffusion integrals**: Arnett-style diffusion uses `jax.vmap` over time points with log-mirror quadrature nodes
- **Spectra pipeline**: `make_spectra_model(bolometric_fn)` wraps any bolometric model to produce time × wavelength spectra for bandflux/magnitude comparison
- **Clean inference API**: `Prior`, `Likelihood`, `NestedSampler`, and `MCMCSampler` — compose a full Bayesian fit in ~15 lines
- **Multi-sampler support**: BlackJAX NUTS (MCMC) and nested sampling via `blackjax.nss`

## Models

### Bolometric models (return `log10_lbol` in erg/s)

| Function | Physics | Reference |
|---|---|---|
| `arnett_bolometric` | Ni/Co decay + Arnett diffusion | Arnett 1982 |
| `magnetar_powered_bolometric` | Dipole spin-down + Arnett diffusion | Nicholl+ 2017 |
| `csm_interaction_bolometric` | Forward/reverse shocks + CSM diffusion | Chatzopoulos+ 2013 |
| `tde_analytical_bolometric` | t⁻⁵/³ fallback + Arnett diffusion | — |
| `tde_fallback_bolometric` | Guillochon fallback tables + viscous processing | Guillochon+ 2013, Mockler+ 2019 |
| `shock_cooling_bolometric` | Shock-cooling envelope (n=10) | Piro 2021 |
| `shocked_cocoon_bolometric` | Shocked jet cocoon | Piro & Kollmeier 2018 |
| `metzger_kilonova_bolometric` | r-process ODE, 200 shells | Metzger 2017 |
| `magnetar_boosted_kilonova_bolometric` | r-process ODE + magnetar injection | Yu+ 2013 |
| `general_magnetar_driven_supernova_bolometric` | Relativistic spin-down ODE (float64) | Sarin+ 2022, Omand & Sarin 2024 |

All bolometric functions return **`log10_lbol`** (log base-10 of luminosity in erg/s). 
This is the natural unit for GPU inference — float32 can represent log10 values for any physically realistic luminosity.

### Native Redback afterglows (return mJy or AB magnitude)

The six non-refreshed native Redback jet structures are available as
`tophat_redback`, `gaussian_redback`, `twocomponent_redback`,
`powerlaw_redback`, `alternativepowerlaw_redback`, and
`doublegaussian_redback`. Their six `_refreshed` counterparts reproduce the
native refreshed-shell energy prescription. Supply observer-frame `frequency` in Hz and choose
`output_format="flux_density"` or `output_format="magnitude"`.

Angular `res` and radial `steps` are static compilation settings. Their
defaults are 50 and 250; reducing them is useful for exploratory inference,
while convergence studies should increase both explicitly.

The same wrappers accept arbitrary two-dimensional jet-structure callables,
pluggable synchrotron prescriptions, and arbitrary radial CSM profiles. Set
`reverse_shock=True` with a source-frame `engine_duration` to use the coupled
unmagnetized finite-width forward/reverse-shock backend; reverse-shock
microphysics can be set independently. See the afterglow documentation for the
callable contracts and current limitations.

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

## Bayesian inference — photometric fitting

The `Prior` / `Likelihood` / `NestedSampler` / `MCMCSampler` API handles the full pipeline: model evaluation, bandflux integration, and sampling.

```python
import jax
from redback_jax.inference import Prior, Uniform, Likelihood, NestedSampler, MCMCSampler
from redback_jax.utils import luminosity_distance_cm
from redback_jax.transient import Transient

REDSHIFT = 0.01
DL_CM    = luminosity_distance_cm(REDSHIFT)   # ~1.37e26 cm

# Free parameters
prior = Prior([
    Uniform(58580, 58620,  name='t0'),        # MJD explosion epoch
    Uniform(0.05,  0.30,   name='f_nickel'),
    Uniform(0.5,   3.0,    name='mej'),
    Uniform(3000,  12000,  name='vej'),
])

# Similar to how you would load a transient for Redback — but with JAX arrays
transient = Transient(
    time=times_list,
    y=mags_list,
    y_err=y_err,
    bands=bands_list,
    data_mode='magnitude',
    name='SN2019abcde',
    redshift=REDSHIFT,
)
# Likelihood — transient.time (MJD), transient.y (AB mag), transient.y_err, transient.bands
likelihood = Likelihood(
    model='arnett_spectra',
    transient=transient,
    fixed_params={
        'redshift':          REDSHIFT,
        'lum_dist':          DL_CM,
        'temperature_floor': 5000.0,
        'kappa':             0.07,
        'kappa_gamma':       0.1,
    },
    evaluation_mode='direct_photometry',  # opt-in fast path for fitting
)

# Nested sampling (BlackJAX)
ns_result = NestedSampler(likelihood, prior, outdir='results/').run(jax.random.PRNGKey(0))
ns_result.summary()

# Or MCMC with NUTS (BlackJAX)
mcmc_result = MCMCSampler(likelihood, prior, n_warmup=500, n_samples=2000, n_chains=4).run(
    jax.random.PRNGKey(1)
)
mcmc_result.summary()
```

### Fast photometric inference modes

`Likelihood` now has **opt-in** fast evaluation modes for GPU fitting. The
default remains `evaluation_mode="full"`, so existing source-model behaviour is
unchanged.

| Mode | What it does | Keeps `jax_supernovae.timeseries_multiband_flux`? | Best use |
|---|---|---|---|
| `full` | Uses the model's default full source grid | Yes | Backward-compatible default |
| `compact_source` | Uses a dataset-specific source phase grid for the likelihood | Yes | Faster fitting while keeping the source-cube infrastructure |
| `direct_photometry` | Integrates the blackbody model directly through the precomputed bandpasses | No | Fastest photometric fitting path |

Notes:

- `compact_source` is inference-only and does **not** change the model defaults.
- `direct_photometry` is currently available for spectra models built with
  `make_spectra_model(...)`, including `arnett_spectra`.
- These are fitting accelerators; if you need a reusable source or exported
  spectra, keep using the full spectra path.

#### Example: switching between modes

```python
full_like = Likelihood(
    model='arnett_spectra',
    transient=transient,
    fixed_params=FIXED,
    evaluation_mode='full',
)

compact_like = Likelihood(
    model='arnett_spectra',
    transient=transient,
    fixed_params=FIXED,
    evaluation_mode='compact_source',
    compact_time_grid_size=256,
    compact_grid_pad_days=5.0,
)

fast_like = Likelihood(
    model='arnett_spectra',
    transient=transient,
    fixed_params=FIXED,
    evaluation_mode='direct_photometry',
)
```

See `examples/arnett_ns_fast.py` for a complete nested-sampling example using
the fast path.

### Available models

Pass any string from `redback_jax.models.MODELS` as the `model` argument:

```python
from redback_jax.models import MODELS
print(list(MODELS.keys()))
# ['arnett_spectra', 'magnetar_spectra', 'csm_spectra', ...]
```

## Direct spectra / magnitude evaluation

To compute magnitudes outside of inference (e.g. for plotting):

```python
from redback_jax.sources import PrecomputedSpectraSource
from redback_jax.utils import luminosity_distance_cm

source = PrecomputedSpectraSource.from_arnett_model(
    f_nickel=0.15, mej=1.0, vej=8000.0,
    redshift=0.01,
    cosmo_H0=67.66, cosmo_Om0=0.3111,
)

# AB magnitude in ztfr at a set of phases
phases = jnp.linspace(-5, 40, 200)
mags   = source.bandmag({'amplitude': 1.0}, 'ztfr', phases)
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



## Related Projects

- [Redback](https://github.com/nikhil-sarin/redback) — the full transient-modeling and analysis stack that Redback-JAX complements
- [JAX-bandflux](https://github.com/samleeney/JAX-bandflux): `jax-bandflux`
- [JAX](https://github.com/google/jax) — the underlying numerical computing library

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

## Acknowledgments/Citations

Developed alongside and in close connection with the
[Redback](https://github.com/nikhil-sarin/redback) ecosystem.

If you use Redback-JAX, please cite the redback paper. 
Please make sure you also cite all relevant papers for the models. 
These are the same as the papers cited in the original redback package. 

If you use magnitude/flux evaluation please also cite 
- [JAX-bandflux](https://github.com/samleeney/JAX-bandflux): `jax-bandflux` and any other papers recommended by those authors.

If you do any sampling, please cite the relevant sampling papers. 

- [BlackJAX](https://github.com/blackjax-devs/blackjax): `blackjax` and any papers recommended by those authors.

## Redback-JAX paper

A paper describing the Redback-JAX package is in preparation. 
Redback-JAX is still very much in development and the API/etc may not be stable.
