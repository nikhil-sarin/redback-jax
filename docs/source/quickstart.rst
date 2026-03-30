Quick Start
===========

Installation
------------

.. code-block:: bash

   git clone https://github.com/nikhil-sarin/redback-jax.git
   cd redback-jax
   pip install -e .

Float32 mode (recommended for GPU)
-----------------------------------

Redback-JAX is designed to run in float32. Disable x64 before importing any
JAX code::

   import jax
   jax.config.update("jax_enable_x64", False)

Do **not** import ``jax_supernovae`` (jax-bandflux) before this call — it
enables x64 at module level. Redback-JAX lazy-imports bandflux components to
avoid this.

Bolometric light curves
-----------------------

All bolometric functions return ``log10(L)`` in erg/s — not linear luminosity.
This is deliberate: physical luminosities (~10³⁸–10⁴⁵ erg/s) exceed the
float32 maximum of ~3.4×10³⁸, so working in log10 space is the only way to
stay float32-safe on GPU.

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", False)
   import jax.numpy as jnp
   from redback_jax.models.supernova_models import arnett_bolometric

   time = jnp.linspace(1.0, 100.0, 200, dtype=jnp.float32)

   log10_lbol = arnett_bolometric(
       time,
       f_nickel=0.5,
       mej=1.0,        # solar masses
       vej=10000.0,    # km/s
       kappa=0.1,      # cm^2/g
       kappa_gamma=10.0,
   )
   # log10_lbol ~ [41, 43]  (physical range, float32-safe)

Fitting bolometric data
-----------------------

Compare model and data in log10 space:

.. code-block:: python

   import jax.numpy as jnp
   from redback_jax.models.supernova_models import arnett_bolometric

   # Observed bolometric luminosities
   log10_lbol_obs = jnp.log10(observed_lbol)
   # Propagate fractional errors: sigma_{log10 L} = sigma_L / (L * ln10)
   log10_lbol_err = sigma_lbol / (observed_lbol * jnp.log(10.0))

   def log_likelihood(params):
       log10_lbol_model = arnett_bolometric(time, **params)
       return -0.5 * jnp.sum(((log10_lbol_obs - log10_lbol_model) / log10_lbol_err)**2)

Spectra and photometry
----------------------

``make_spectra_model`` wraps any bolometric model into a full SED pipeline
(photosphere → blackbody → observer-frame flux density):

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", False)
   import jax.numpy as jnp
   from redback_jax.models.supernova_models import arnett_bolometric
   from redback_jax.models.spectra_model import make_spectra_model

   arnett_spectra = make_spectra_model(arnett_bolometric)

   output = arnett_spectra(
       redshift=0.05,
       lum_dist=7e26,           # cm (~230 Mpc)
       temperature_floor=3000.0,
       # bolometric kwargs:
       f_nickel=0.5, mej=1.0,
       vej=10000.0, kappa=0.1, kappa_gamma=10.0,
   )

   output.time     # observer-frame times (days)
   output.lambdas  # wavelengths (Angstrom)
   output.spectra  # (n_times, n_lambda)  erg/s/cm^2/Angstrom

For bandflux fitting, pass the spectra grid to ``PrecomputedSpectraSource``:

.. code-block:: python

   from redback_jax.sources import PrecomputedSpectraSource

   source = PrecomputedSpectraSource(
       phases=output.time,
       wavelengths=output.lambdas,
       flux_grid=output.spectra,
   )

   bridges, band_to_idx = source.prepare_bridges(['ztfg', 'ztfr'])
   band_indices = jnp.array([band_to_idx[b] for b in observed_bands])

   model_fluxes = source.bandflux(
       {'amplitude': 1.0}, None, obs_times,
       band_indices=band_indices, bridges=bridges,
       unique_bands=['ztfg', 'ztfr'],
   )

Available models
----------------

**Supernovae**

- ``arnett_bolometric`` — Ni/Co decay + Arnett diffusion (Arnett 1982)
- ``magnetar_powered_bolometric`` — dipole spin-down + Arnett diffusion
- ``csm_interaction_bolometric`` — forward/reverse shocks + CSM diffusion

**TDE**

- ``tde_analytical_bolometric`` — t⁻⁵/³ fallback + Arnett diffusion

  Note: parameter is ``log10_l0`` (not ``l0``), because the linear value
  (~10⁴³ erg/s) overflows float32.

**Shock-powered**

- ``shock_cooling_bolometric`` — Piro 2021

  Parameters ``log10_mass``, ``log10_radius``, ``log10_energy`` (log10 of
  solar masses, cm, erg respectively), plus ``nn`` (outer density slope),
  ``delta`` (inner density slope), ``kappa`` (opacity in cm²/g).

- ``shocked_cocoon_bolometric`` — Piro & Kollmeier 2018

**Kilonova**

- ``metzger_kilonova_bolometric`` — r-process ODE, 200-shell (Metzger 2017)
- ``magnetar_boosted_kilonova_bolometric`` — r-process + magnetar injection

All models are ``@jax.jit`` compiled and support ``jax.grad`` and ``jax.vmap``.

Next steps
----------

* See :doc:`examples` for complete inference examples
* See :doc:`api` for full function signatures
