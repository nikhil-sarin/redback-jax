"""
Nested Sampling with Arnett Model

Fits the Arnett supernova model to fake multi-band photometry.

Install requirements:
    pip install git+https://github.com/handley-lab/blackjax@proposal
    pip install anesthetic tqdm wcosmo

Usage:
    python arnett_ns.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from redback_jax.models import arnett_spectra
from redback_jax.utils import luminosity_distance_cm
from redback_jax.sources import PrecomputedSpectraSource
from redback_jax.transient import Transient
from redback_jax.inference import Prior, Uniform, Likelihood, NestedSampler

# ============================================================================
# Setup
# ============================================================================

REDSHIFT  = 0.01
LUM_DIST  = luminosity_distance_cm(REDSHIFT)

TRUE_PARAMS = {'t0': 58600.0, 'f_nickel': 0.1, 'mej': 1.4, 'vej': 5000.0}

FIXED = {
    'redshift':          REDSHIFT,
    'lum_dist':          LUM_DIST,
    'temperature_floor': 5000.0,
    'kappa':             0.07,
    'kappa_gamma':       0.1,
}

BANDS          = ['bessellb', 'bessellv', 'bessellr', 'besselli']
N_OBS_PER_BAND = 15
MAG_ERR        = 0.05

# ============================================================================
# Generate fake observations
# ============================================================================

print("Generating fake data...")

_out = arnett_spectra(**FIXED, vej=TRUE_PARAMS['vej'], f_nickel=TRUE_PARAMS['f_nickel'],
                      mej=TRUE_PARAMS['mej'])
_src = PrecomputedSpectraSource(phases=_out.time, wavelengths=_out.lambdas, flux_grid=_out.spectra)
_bridges, _band_to_idx = _src.prepare_bridges(BANDS)

obs_times_rel = jnp.linspace(5.0, 50.0, N_OBS_PER_BAND)  # source-frame days
obs_times_mjd = obs_times_rel + TRUE_PARAMS['t0']

times_list, bands_list, mags_list = [], [], []
rng = np.random.RandomState(42)

for band in BANDS:
    bi = _band_to_idx[band]
    true_mags = _src.bandmag(
        {'amplitude': 1.0}, None, obs_times_rel,
        band_indices=jnp.array([bi] * N_OBS_PER_BAND),
        bridges=_bridges, unique_bands=BANDS,
    )
    noisy = np.array(true_mags) + rng.normal(0, MAG_ERR, N_OBS_PER_BAND)
    times_list.extend(obs_times_mjd.tolist())
    bands_list.extend([band] * N_OBS_PER_BAND)
    mags_list.extend(noisy.tolist())

transient = Transient(
    time=np.array(times_list),
    y=np.array(mags_list),
    y_err=np.full(len(mags_list), MAG_ERR),
    bands=bands_list,
    data_mode='magnitude',
    name='fake_arnett_SN',
    redshift=REDSHIFT,
)
print(f"  {len(transient.time)} observations ({len(BANDS)} bands × {N_OBS_PER_BAND} epochs)")

# ============================================================================
# Inference
# ============================================================================

prior = Prior([
    Uniform(58580, 58620, name='t0'),
    Uniform(0.05,  0.20,  name='f_nickel'),
    Uniform(0.8,   2.0,   name='mej'),
    Uniform(3000,  8000,  name='vej'),
])

likelihood = Likelihood(
    model        = 'arnett_spectra',
    transient    = transient,
    fixed_params = FIXED,
)

result = NestedSampler(likelihood, prior, outdir='results/').run(jax.random.PRNGKey(42))
result.summary()

print(f"\n{'Param':<12} {'True':>10}")
print("-" * 25)
for p in prior.names:
    print(f"{p:<12} {TRUE_PARAMS[p]:>10.4f}")

NestedSampler(likelihood, prior, outdir='results/').plot_corner(
    result, truth=TRUE_PARAMS, filename='results/corner.png'
)
