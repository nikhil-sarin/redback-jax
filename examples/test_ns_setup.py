"""
Setup verification for nested sampling examples.

Tests the full pipeline before launching a run.
"""

import time as _time

import jax
import jax.numpy as jnp
import numpy as np

import wcosmo

from redback_jax.models import arnett_spectra
from redback_jax.models.supernova_models import PLANCK18_H0, PLANCK18_OM0
from redback_jax.sources import PrecomputedSpectraSource
from redback_jax.transient import Transient
from redback_jax.inference import Prior, Uniform, Likelihood

print("=" * 60)
print("Nested Sampling Setup Test")
print("=" * 60)

REDSHIFT = 0.01
LUM_DIST = wcosmo.luminosity_distance(REDSHIFT, PLANCK18_H0, PLANCK18_OM0).value * 3.085677581e24
BANDS    = ['bessellv', 'bessellr']
FIXED    = {
    'redshift':          REDSHIFT,
    'lum_dist':          LUM_DIST,
    'temperature_floor': 5000.0,
    'kappa':             0.07,
    'kappa_gamma':       0.1,
}

# ============================================================================
# Test 1: spectra model
# ============================================================================
print("\nTest 1: arnett_spectra...")
t0 = _time.time()
out = arnett_spectra(**FIXED, vej=5000.0, f_nickel=0.1, mej=1.4)
print(f"  time {out.time.shape}, spectra {out.spectra.shape}  [{_time.time()-t0:.2f}s]")

# ============================================================================
# Test 2: bandmag
# ============================================================================
print("\nTest 2: bandmag...")
source = PrecomputedSpectraSource(phases=out.time, wavelengths=out.lambdas, flux_grid=out.spectra)
test_times = jnp.linspace(5.0, 40.0, 8)

for band in BANDS:
    mags = source.bandmag({'amplitude': 1.0}, band, test_times)
    assert jnp.all(jnp.isfinite(mags)), f"Non-finite mags in {band}"
    print(f"  {band}: {float(mags.min()):.2f}–{float(mags.max()):.2f} mag")

# ============================================================================
# Test 3: Likelihood + Prior
# ============================================================================
print("\nTest 3: Likelihood + Prior...")

obs_times_rel = jnp.linspace(5.0, 40.0, 8)
obs_times_mjd = obs_times_rel + 58600.0
times_list, bands_list, mags_list = [], [], []
rng = np.random.RandomState(0)

for band in BANDS:
    true_mags = source.bandmag({'amplitude': 1.0}, band, obs_times_rel)
    noisy = np.array(true_mags) + rng.normal(0, 0.05, 8)
    times_list.extend(obs_times_mjd.tolist())
    bands_list.extend([band] * 8)
    mags_list.extend(noisy.tolist())

transient = Transient(
    time=np.array(times_list), y=np.array(mags_list),
    y_err=np.full(len(mags_list), 0.05),
    bands=bands_list, data_mode='magnitude', redshift=REDSHIFT,
)

prior = Prior([
    Uniform(58585.0, 58615.0, name='t0'),
    Uniform(0.05,    0.20,    name='f_nickel'),
    Uniform(0.8,     2.0,     name='mej'),
    Uniform(3000.0,  8000.0,  name='vej'),
])

likelihood = Likelihood(model='arnett_spectra', transient=transient, fixed_params=FIXED)
print(f"  {likelihood}")

# ============================================================================
# Test 4: likelihood evaluation (JIT happens automatically)
# ============================================================================
print("\nTest 4: likelihood evaluation...")

log_like = likelihood._make_log_likelihood(prior)
true_p  = prior.dict_to_params({'t0': 58600.0, 'f_nickel': 0.1,  'mej': 1.4, 'vej': 5000.0})
wrong_p = prior.dict_to_params({'t0': 58600.0, 'f_nickel': 0.15, 'mej': 1.8, 'vej': 6000.0})

t0 = _time.time()
ll_true  = log_like(true_p)
print(f"  first call (JIT compile): {_time.time()-t0:.2f}s")

t0 = _time.time()
ll_wrong = log_like(wrong_p)
print(f"  subsequent call: {(_time.time()-t0)*1000:.1f}ms")
print(f"  log-like (true): {float(ll_true):.2f},  log-like (wrong): {float(ll_wrong):.2f}")

assert ll_true > ll_wrong, "True parameters should have higher likelihood"
print("  ✓ True parameters have higher likelihood")

print("\n" + "=" * 60)
print("All tests passed.  Ready to run:")
print("  python arnett_ns_quick.py")
print("  python arnett_ns.py")
