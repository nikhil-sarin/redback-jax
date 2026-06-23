"""
General Magnetar-Driven Supernova — solver and tolerance tutorial
=================================================================

This script demonstrates how to use the ``general_magnetar_driven_supernova``
family of models in **redback-jax**.  It is structured as a self-contained
tutorial that can be run top-to-bottom; each section prints results and saves
a figure.

Physical model
--------------
The model describes a supernova ejecta shell powered by a magnetar spin-down
engine. A relativistic ODE (Lorentz factor, radius, volume, internal energy)
is integrated forward in source-frame time; the ejecta photosphere and a
CutoffBlackbody SED (Nicholl+2017) then convert the bolometric luminosity to
an observed spectral energy distribution.

Sections
--------
  1. Solver backends  — diffrax (adaptive Tsit5, default) vs euler (fixed-step)
  2. Tolerance control — how rtol/atol trade accuracy for speed

Requirements: diffrax, jax >= 0.4, astropy
Note: float64 precision is required because the ODE state spans >100 dex in
      luminosity.  It is enabled before any JAX operation below.
"""

import sys
import pathlib
import time as _time
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — makes redback_jax importable when running from examples/
# ---------------------------------------------------------------------------
_repo_root = str(pathlib.Path().absolute().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from redback_jax.models import general_magnetar_driven_supernova_bolometric

print(f"JAX {jax.__version__}  |  devices: {jax.devices()}")

# ---------------------------------------------------------------------------
# Reference parameters
# ---------------------------------------------------------------------------
# These values are used as the "ground truth" throughout the script.
# They represent a moderately massive, fast-magnetar-powered supernova.
#
# Free parameters (inferred in a real fit):
#   mej    — ejecta mass; controls the diffusion timescale and peak luminosity
#   E_sn   — kinetic energy of the explosion; sets the initial ejecta velocity
#   l0     — initial magnetar spin-down luminosity
#   tau_sd — magnetar spin-down timescale; ~1 day (millisecond pulsar) to
#             ~100 days (slower rotator)
#   temperature_floor — minimum photosphere temperature; below this the
#             ejecta is optically thin and the SED cuts off
#
# Fixed parameters (held constant during fitting):
#   kappa, nn, kappa_gamma — opacity and braking index; weakly constrained
#   f_nickel, cutoff_wavelength, alpha_uv — SED shape parameters
# ---------------------------------------------------------------------------
TRUTH = dict(
    mej               = 3.5,       # ejecta mass  (M_sun)
    E_sn              = 1.5e51,    # explosion energy  (erg)
    kappa             = 0.1,       # optical opacity  (cm²/g)
    l0                = 1e44,      # initial spin-down luminosity  (erg/s)
    tau_sd            = 3e6,       # spin-down timescale  (s)
    nn                = 3.0,       # magnetic braking index
    kappa_gamma       = 1.0,       # gamma-ray opacity  (cm²/g)
    temperature_floor = 4000.0,    # photosphere temperature floor  (K)
    f_nickel          = 0.0,       # nickel mass fraction
    cutoff_wavelength = 3000.0,    # UV cutoff wavelength  (Å)
    alpha_uv          = 1.0,       # UV power-law suppression index
)

# Keyword arguments common to every model call in this script.
# general_magnetar_driven_supernova_bolometric returns log10(L_bol / erg/s)
# evaluated on a source-frame time grid.
base_kw = dict(
    mej          = TRUTH['mej'],
    E_sn         = TRUTH['E_sn'],
    kappa        = TRUTH['kappa'],
    l0           = TRUTH['l0'],
    tau_sd       = TRUTH['tau_sd'],
    nn           = TRUTH['nn'],
    kappa_gamma  = TRUTH['kappa_gamma'],
)

# Source-frame time grid: 30 points log-spaced from 1 to 100 days
t_src = jnp.geomspace(1.0, 100.0, 30)

# ===========================================================================
# 1. Solver Backends: diffrax vs euler
# ===========================================================================
#
# All three public model functions accept a ``solver`` keyword:
#
#   solver='diffrax'  (default)
#     Adaptive-step Tsit5 integrator from the diffrax library (4th/5th-order
#     Runge-Kutta with error control).  Step sizes are chosen automatically to
#     satisfy rtol and atol tolerances.  Typically ~10× faster than euler for
#     the same accuracy.  Recommended for all production use.
#
#   solver='euler'
#     Fixed-step Euler integrator implemented as a JAX lax.scan loop.
#     Deterministic and easy to reason about, but requires a large n_grid
#     (≥1000) for acceptable accuracy.  Useful as a debugging baseline.
#
# Because ``solver`` is a static_argname, JAX JIT-compiles a separate kernel
# for each value — switching between backends incurs no runtime branching.
#
# Timing methodology
# ------------------
# JAX operations are asynchronous: a model call returns immediately while the
# GPU/CPU kernel runs in the background.  ``block_until_ready()`` forces
# synchronisation so the wall-clock time includes the actual computation.
#
# The first call to a JIT-compiled function triggers compilation (slow).
# Subsequent calls reuse the compiled kernel (fast).  We therefore run a short
# warmup phase before collecting timing samples, and use the median over many
# calls to suppress outliers from OS scheduling jitter.
# ===========================================================================

# --- diffrax (default) -----------------------------------------------------

# First call: JIT compilation + execution
_t0 = _time.perf_counter()
lbol_diffrax = general_magnetar_driven_supernova_bolometric(t_src, **base_kw)
lbol_diffrax.block_until_ready()
print(f"\ndiffrax — compile+run: {(_time.perf_counter() - _t0)*1e3:.0f} ms")

# Warmup: 5 calls to let XLA dispatch overhead stabilise after compilation
for _ in range(5):
    general_magnetar_driven_supernova_bolometric(t_src, **base_kw).block_until_ready()

# Steady-state timing: median of 200 calls
ts_diffrax = []
for _ in range(200):
    _t0 = _time.perf_counter()
    general_magnetar_driven_supernova_bolometric(t_src, **base_kw).block_until_ready()
    ts_diffrax.append(_time.perf_counter() - _t0)
t_diffrax_ms = np.median(ts_diffrax) * 1e3

# --- euler (fixed-step) ----------------------------------------------------

# Warmup: euler has its own JIT kernel; n_grid=2000 is in static_argnames
for _ in range(5):
    general_magnetar_driven_supernova_bolometric(
        t_src, **base_kw, solver='euler', n_grid=2000
    ).block_until_ready()

# Steady-state timing: median of 50 calls (euler is ~10× slower so fewer suffice)
ts_euler = []
for _ in range(50):
    _t0 = _time.perf_counter()
    general_magnetar_driven_supernova_bolometric(
        t_src, **base_kw, solver='euler', n_grid=2000
    ).block_until_ready()
    ts_euler.append(_time.perf_counter() - _t0)
t_euler_ms = np.median(ts_euler) * 1e3

lbol_euler = general_magnetar_driven_supernova_bolometric(
    t_src, **base_kw, solver='euler', n_grid=2000
)

print(f"\nSteady-state timing (median):")
print(f"  diffrax (default, rtol=1e-5):  {t_diffrax_ms:.3f} ms/call")
print(f"  euler   (n_grid=2000):         {t_euler_ms:.3f} ms/call")
print(f"\nMax |Δlog₁₀L| between backends: "
      f"{float(jnp.max(jnp.abs(lbol_diffrax - lbol_euler))):.4f} dex")

# --- Plot ------------------------------------------------------------------

fig, (ax_lc, ax_res) = plt.subplots(1, 2, figsize=(12, 4))

ax_lc.plot(t_src, lbol_diffrax, 'b-',  lw=2, label='diffrax (default)')
ax_lc.plot(t_src, lbol_euler,   'r--', lw=2, label='euler (n_grid=2000)', alpha=0.8)
ax_lc.set_xlabel('Source-frame time (days)')
ax_lc.set_ylabel(r'$\log_{10}(L_{\rm bol}\,/\,{\rm erg\,s^{-1}})$')
ax_lc.set_title('Bolometric light curve — solver comparison')
ax_lc.legend()

ax_res.plot(t_src, lbol_diffrax - lbol_euler, 'k-', lw=1.5)
ax_res.axhline(0, color='gray', lw=0.8, ls='--')
ax_res.set_xlabel('Source-frame time (days)')
ax_res.set_ylabel(r'$\Delta\log_{10}L$ (diffrax − euler)')
ax_res.set_title('Residuals between backends')

plt.tight_layout()
plt.savefig('solver_comparison.pdf', bbox_inches='tight')
plt.show()

# ===========================================================================
# 2. Tolerance Control: rtol and atol
# ===========================================================================
#
# The diffrax backend uses a PID step-size controller that targets a local
# truncation error below max(rtol × |y|, atol) at each ODE step.
#
#   rtol — relative tolerance: scales with the magnitude of the solution
#   atol — absolute tolerance: floor that prevents the controller from taking
#          arbitrarily large steps when y ≈ 0
#
# Practical guidance
# ------------------
#   loose  (1e-3 / 1e-6)  — fastest; fine for exploration and prior predictive
#                           checks where exact values are unimportant
#   default (1e-5 / 1e-8) — recommended for nested sampling and MCMC; error
#                           typically < 1e-4 dex, well below photometric noise
#   tight  (1e-7 / 1e-10) — use for gradient-based optimizers (HMC, L-BFGS)
#                           where inaccurate gradients can destabilise the chain
#
# Each (rtol, atol) pair is a distinct set of static_argnames, so JAX compiles
# a separate kernel per configuration.  The warmup run below triggers that
# compilation before timing begins.
# ===========================================================================

tolerance_configs = [
    dict(rtol=1e-3, atol=1e-6,  label='loose (1e-3/1e-6)',   color='tab:orange'),
    dict(rtol=1e-5, atol=1e-8,  label='default (1e-5/1e-8)', color='tab:blue'),
    dict(rtol=1e-7, atol=1e-10, label='tight (1e-7/1e-10)',  color='tab:green'),
]

# Ultra-tight reference solution (rtol=1e-10) — essentially exact for this model
lbol_ref = general_magnetar_driven_supernova_bolometric(
    t_src, **base_kw, rtol=1e-10, atol=1e-12
)
ref = np.array(lbol_ref)

print(f"\n{'Config':<25} {'Time (ms)':>10} {'Max error (dex)':>18}")
print('-' * 58)

tol_results = []
for cfg in tolerance_configs:
    rtol, atol, label = cfg['rtol'], cfg['atol'], cfg['label']

    # Warmup: compile this (rtol, atol) kernel before timing
    for _ in range(5):
        general_magnetar_driven_supernova_bolometric(
            t_src, **base_kw, rtol=rtol, atol=atol
        ).block_until_ready()

    # Timing: median of 200 calls
    ts = []
    for _ in range(200):
        _t0 = _time.perf_counter()
        out = general_magnetar_driven_supernova_bolometric(
            t_src, **base_kw, rtol=rtol, atol=atol
        )
        out.block_until_ready()
        ts.append(_time.perf_counter() - _t0)

    t_ms = np.median(ts) * 1e3
    err  = float(jnp.max(jnp.abs(out - lbol_ref)))
    tol_results.append({**cfg, 't_ms': t_ms, 'err': err, 'lbol': np.array(out)})
    print(f"  {label:<23} {t_ms:>10.3f} {err:>18.2e}")

# --- Plot ------------------------------------------------------------------

fig, (ax_lc, ax_err) = plt.subplots(1, 2, figsize=(12, 4))

for r in tol_results:
    ax_lc.plot(t_src, r['lbol'], color=r['color'], lw=2, label=r['label'])
ax_lc.set_xlabel('Source-frame time (days)')
ax_lc.set_ylabel(r'$\log_{10}(L_{\rm bol})$')
ax_lc.set_title('Effect of rtol/atol on bolometric light curve')
ax_lc.legend(fontsize=9)

for r in tol_results:
    # Floor at 1e-16 prevents log(0) on the semilogy axis
    ax_err.semilogy(t_src, np.abs(r['lbol'] - ref) + 1e-16,
                    color=r['color'], lw=1.5, label=r['label'])
ax_err.set_xlabel('Source-frame time (days)')
ax_err.set_ylabel(r'$|\Delta\log_{10}L_{\rm bol}|$ vs ultra-tight ref')
ax_err.set_title('Absolute error vs ultra-tight reference (1e-10/1e-12)')
ax_err.legend(fontsize=9)

plt.tight_layout()
plt.savefig('tolerance_comparison.pdf', bbox_inches='tight')
plt.show()

print("\nTiming summary:")
for r in tol_results:
    print(f"  {r['label']:<25}: {r['t_ms']:.3f} ms/call")
