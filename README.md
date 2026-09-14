# Redback-JAX

[![Documentation Status](https://readthedocs.org/projects/redback-jax/badge/?version=latest)](https://redback-jax.readthedocs.io/en/latest/)
[![Tests](https://github.com/nikhil-sarin/redback-jax/workflows/Tests/badge.svg)](https://github.com/nikhil-sarin/redback-jax/actions)
[![codecov](https://codecov.io/gh/nikhil-sarin/redback-jax/branch/main/graph/badge.svg)](https://codecov.io/gh/nikhil-sarin/redback-jax)

A JAX-native companion to [Redback](https://github.com/nikhil-sarin/redback)
for rapid, differentiable electromagnetic-transient analysis on CPUs, GPUs,
and TPUs.

Redback-JAX sits alongside the broader Redback stack. It provides selected
Redback models and newer composable modeling tools in JAX, aimed at workflows
where repeated evaluation, gradients, vectorization, or accelerator execution
matter. It is not a complete rewrite or a drop-in replacement: use Redback for
the full modeling and analysis ecosystem, and Redback-JAX for supported
high-throughput or differentiable analyses.

## Highlights

- JIT compilation, automatic differentiation, and `vmap`-friendly model paths
- Bolometric supernova, shock-cooling, TDE, kilonova, and magnetar models
- Native Redback afterglows with six built-in jet structures and refreshed variants
- Arbitrary 2D jet structures and radial CSM density profiles
- Pluggable forward- and reverse-shock radiation prescriptions
- Continuous energy injection and finite-width forward/reverse-shock dynamics
- Blackbody spectra, SED features, and JAX-bandflux photometry
- BlackJAX MCMC and nested-sampling interfaces
- A registry and entry-point API for external model packages

The package is under active development. APIs and numerical implementations
may change between releases.

## Installation

Redback-JAX requires Python 3.12 or newer. To install the current source:

```bash
git clone https://github.com/nikhil-sarin/redback-jax.git
cd redback-jax
pip install -e .
```

Install optional inference or development dependencies with:

```bash
pip install -e ".[inference]"
pip install -e ".[dev]"
```

Core dependencies include JAX, Diffrax, JAX-bandflux, Astropy, NumPy, SciPy,
Pandas, Matplotlib, `unxt`, and `wcosmo`. The inference extra installs
BlackJAX, NumPyro, and Optax.

## Quick start

Bolometric models accept source-frame time in days and return
`log10(L_bol / (erg s^-1))`:

```python
import jax
import jax.numpy as jnp

from redback_jax.models import arnett_bolometric

time = jnp.geomspace(0.1, 100.0, 200)
log10_lbol = jax.jit(arnett_bolometric)(
    time,
    f_nickel=0.1,
    mej=1.4,
    vej=8_000.0,
    kappa=0.07,
    kappa_gamma=0.1,
)
```

The afterglow wrappers follow the native Redback naming convention. They
accept observer-frame time in days and frequency in Hz, and return mJy by
default:

```python
from redback_jax.models import gaussian_redback

flux_mjy = gaussian_redback(
    time=jnp.geomspace(0.1, 1_000.0, 100),
    frequency=3.0e9,
    redshift=0.01,
    thv=0.05,
    loge0=52.0,
    thc=0.1,
    thj=0.4,
    logn0=0.0,
    p=2.2,
    logepse=-1.0,
    logepsb=-2.0,
    g0=100.0,
    xiN=1.0,
    res=20,
)
```

Set `output_format="magnitude"` for AB magnitudes. The angular resolution
`res` and radial resolution `steps` are static compilation settings; their
defaults are 50 and 250. Use explicit convergence tests when reducing them for
exploratory inference.

## Model coverage

### Bolometric and continuum models

| Family | Public models |
|---|---|
| Radioactive and magnetar supernovae | `arnett_bolometric`, `magnetar_powered_bolometric`, `magnetar_nickel_bolometric` |
| CSM interaction | `csm_interaction_bolometric` |
| Shock powered | `shock_cooling_bolometric`, `shocked_cocoon_bolometric`, `shock_cooling_and_arnett_bolometric` |
| Tidal disruption events | `tde_analytical_bolometric`, `tde_fallback_bolometric` |
| Kilonovae | `metzger_kilonova_bolometric`, `magnetar_boosted_kilonova_bolometric` |
| General magnetar-driven supernovae | Bolometric, bolometric-plus-velocity, batched, Diffrax, flux-density, and spectra variants in the `general_magnetar_driven_supernova*` family |

`arnett_with_features_cosmology` and `blackbody_to_flux_density` provide lower-level
continuum paths. The separate `redback_jax.phenomenological_models` module
contains `smooth_exponential_powerlaw`, `exp_rise_powerlaw_decline`,
`bazin_sne`, and `villar_sne`.

### Native Redback afterglows

The six built-in jet families are:

- `tophat_redback`
- `gaussian_redback`
- `twocomponent_redback`
- `powerlaw_redback`
- `alternativepowerlaw_redback`
- `doublegaussian_redback`

Each also has a `_refreshed` variant implementing the native Redback
refreshed-shell prescription. The default impulsive paths reproduce the
corresponding native Redback calculations to numerical tolerance while
remaining composable.

The same public wrappers support:

- `structure_function` for arbitrary axisymmetric or azimuth-dependent jet structures
- `density_function` for arbitrary differentiable radial CSM profiles
- `radiation_function` for custom forward-shock emission
- `engine_function` for continuous energy injection beyond the impulsive approximation
- `reverse_shock=True` for a coupled finite-width forward/reverse-shock solution
- independent reverse-shock microphysics or `reverse_radiation_function`

Callable contracts, normalization conventions, examples, and limitations are
documented in [the afterglow guide](docs/source/afterglow.rst).

## Spectra and photometry

`make_spectra_model` converts a bolometric function into an observer-frame
blackbody SED model:

```python
from redback_jax.models import arnett_spectra
from redback_jax.utils import luminosity_distance_cm

redshift = 0.01
spectra = arnett_spectra(
    redshift=redshift,
    lum_dist=luminosity_distance_cm(redshift),
    vej=8_000.0,
    temperature_floor=5_000.0,
    f_nickel=0.1,
    mej=1.4,
    kappa=0.07,
    kappa_gamma=0.1,
)

# spectra.time: observer-frame days
# spectra.lambdas: observer-frame Angstrom
# spectra.spectra: erg s^-1 cm^-2 Angstrom^-1
```

Prebuilt spectra models are available for the radioactive/magnetar
supernovae, CSM, shock-powered, TDE, and kilonova bolometric families.
`SEDFeatures` can modify the continuum, and `PrecomputedSpectraSource`
provides JAX-bandflux-compatible `bandflux` and `bandmag` evaluation. See
[`examples/arnett_to_magnitudes.py`](examples/arnett_to_magnitudes.py).

## Bayesian inference

The inference API exports:

- priors: `Prior`, `Uniform`, `LogUniform`, and `Gaussian`
- likelihoods: `Likelihood` for spectra/photometry and
  `FluxDensityLikelihood` for flux-density models such as afterglows
- samplers: `MCMCSampler`/`MCMCResult` and `NestedSampler`/`NSResult`

For photometric fitting, `Likelihood` provides three evaluation modes:

| Mode | Behavior | Intended use |
|---|---|---|
| `full` | Evaluates the model's standard source grid | Reusable spectra and backward-compatible behavior |
| `compact_source` | Builds a dataset-specific source grid | Faster fitting while retaining the source-cube path |
| `direct_photometry` | Integrates supported blackbody models directly through precomputed bandpasses | Fastest supported photometric path |

Models are selected by callable or by name from `MODEL_REGISTRY`:

```python
from redback_jax.models import MODEL_REGISTRY, get_model

print(sorted(MODEL_REGISTRY))
model = get_model("tde_fallback_bolometric")
```

Complete fitting examples are available in
[`examples/arnett_mcmc.py`](examples/arnett_mcmc.py),
[`examples/arnett_ns.py`](examples/arnett_ns.py), and
[`examples/arnett_ns_fast.py`](examples/arnett_ns_fast.py). The clean nested
sampler currently targets the BlackJAX NSS API; consult the example and the
runtime error guidance if the installed BlackJAX build does not expose that
API.

## Numerical and API conventions

Redback-JAX favors numerically scaled representations appropriate to each
model rather than imposing one dtype policy on the entire package.

- Bolometric functions return base-10 logarithmic luminosity in erg/s.
- Afterglow functions return mJy or AB magnitude and use observer-frame time
  and frequency.
- Spectra functions return observer-frame time, wavelength, and flux density.
- Many kernels use logarithmic or scaled internal state so they can run safely
  in float32.
- The general magnetar ODE family uses float64 because its state spans a much
  larger dynamic range.
- JIT compilation adds first-call latency; compare warmed calls for repeated
  analysis and inference workloads.

Some public parameters intentionally differ from native Redback for numerical
safety. For example, `tde_analytical_bolometric` uses `log10_l0`, while the
shock-cooling models use `log10_mass`, `log10_radius`, and `log10_energy`.
Check the selected function's signature and documentation instead of assuming
drop-in parameter parity.

## Performance

Performance depends on backend, array shape, resolution, dtype, and whether a
function has already compiled. The afterglow benchmark reports cold and warm
timings separately and checks the integrated flux difference against native
Redback:

```bash
python benchmarks/afterglow_speed.py \
    --redback-path /path/to/redback \
    --resolution 20
```

Run separate fresh processes for different static resolutions if you want
meaningful cold-compilation measurements.

## Extending the model set

Register local callables directly:

```python
from redback_jax.models import register_model

register_model("my_model", my_model)
```

External packages can publish a registration function through the
`redback_jax.models` entry-point group. Installed plugins are loaded when
`redback_jax.models` is imported. This lets specialized model packages live
alongside the core distribution without expanding its dependency surface.

## Documentation and development

- [Documentation](https://redback-jax.readthedocs.io/)
- [Quick start](docs/source/quickstart.rst)
- [API reference](docs/source/api.rst)
- [Contributing guide](docs/source/contributing.rst)
- [Examples](examples)

Run the test suite with:

```bash
pytest
```

## Related projects

- [Redback](https://github.com/nikhil-sarin/redback) — the full transient-modeling and analysis stack
- [JAX-bandflux](https://github.com/samleeney/JAX-bandflux) — differentiable bandpass integration
- [JAX](https://github.com/jax-ml/jax) — accelerator-oriented array programming and autodiff
- [BlackJAX](https://github.com/blackjax-devs/blackjax) — JAX-based sampling algorithms

## Citation

Redback-JAX was developed alongside the Redback ecosystem. If you use it,
please cite Redback, the papers associated with the selected physical models,
and the relevant numerical packages. Model functions carry citation metadata
where available. For magnitude or flux evaluation, also follow the citation
guidance from JAX-bandflux; for inference, cite the sampler used.

A dedicated Redback-JAX paper is in preparation.

## License

Redback-JAX is distributed under the GNU General Public License v3.0. See
[LICENSE](LICENSE).
