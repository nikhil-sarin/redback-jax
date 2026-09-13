Native Redback afterglows
=========================

The native ``_redback`` family shares a composable JAX engine for jet
structure, blast-wave dynamics, synchrotron radiation, and observer-time
integration. The public wrappers preserve Redback's parameter names and return
flux density in mJy or AB magnitude.

Arbitrary jet structures
------------------------

Every ``_redback`` wrapper accepts ``structure_function`` and a differentiable
``structure_parameters`` PyTree. The static callable has the contract

``structure_function(theta, phi, gamma_core, theta_core, theta_jet, parameters)``

and returns ``(gamma_initial, energy_fraction)`` on the supplied flattened
angular patches. ``energy_fraction`` multiplies the isotropic-equivalent core
energy. This supports axisymmetric and fully azimuth-dependent jets. For the
latter, ``phi_observer`` sets the observer azimuth in radians.

.. code-block:: python

   import jax.numpy as jnp


   def lopsided_gaussian(theta, phi, gamma_core, theta_core, theta_jet, pars):
       del theta_jet
       asymmetry, phi_peak = pars
       radial = jnp.exp(-0.5 * (theta / theta_core) ** 2)
       energy = radial * (1.0 + asymmetry * jnp.cos(phi - phi_peak))
       gamma = 1.0 + (gamma_core - 1.0) * radial
       return gamma, energy


   flux_mjy = gaussian_redback(
       # Standard arguments omitted here for clarity.
       ...,
       structure_function=lopsided_gaussian,
       structure_parameters=(0.5, 0.0),
       phi_observer=0.3,
   )

Returned Lorentz factors must be at least one and energy fractions must be
non-negative. Built-in structures retain the faster polar-ring path; a general
two-dimensional structure evolves all ``res**2`` patches independently.

Radiation prescriptions
-----------------------

``radiation_function`` replaces the final per-patch spectrum without changing
the dynamics or equal-arrival-time integration. Its contract is

``radiation_function(observer_state, log10_frequency, electron_index, parameters)``

and it returns log10 flux density in CGS units along the radial history.
``radiation_parameters`` remains dynamic and differentiable. The supplied
adapters are ``legacy_radiation_prescription`` (the exact native Redback sharp
spectrum with self-absorption), ``optically_thin_radiation_prescription``, and
``smooth_synchrotron_radiation_prescription``. The smooth prescription takes a
one-element tuple containing the break sharpness.

.. code-block:: python

   from redback_jax.afterglow import optically_thin_radiation_prescription

   flux_mjy = gaussian_redback(
       # Standard arguments omitted here for clarity.
       ...,
       radiation_function=optically_thin_radiation_prescription,
       radiation_parameters=(),
   )

The optically thin option matches the assumption made by De Colle &
Ramirez-Ruiz (2026); it should not be used for early radio data where
self-absorption is important.

Arbitrary radial CSM profiles
-----------------------------

Any non-refreshed ``_redback`` wrapper accepts a static JAX callable through
``density_function``. It receives ``log10(radius / cm)`` and a parameter PyTree,
and returns ``log10(number density / cm^-3)``. Profile parameters remain dynamic
and differentiable; changing the callable itself triggers compilation.

.. code-block:: python

   import jax.numpy as jnp

   from redback_jax.afterglow import smoothly_broken_power_law_log_density
   from redback_jax.models import gaussian_redback


   def nuclear_csm(log10_radius, parameters):
       return smoothly_broken_power_law_log_density(log10_radius, *parameters)


   flux_mjy = gaussian_redback(
       time=jnp.geomspace(0.1, 1000.0, 100),
       redshift=0.05,
       thv=0.2,
       loge0=52.0,
       thc=0.08,
       thj=0.4,
       logn0=0.0,
       p=2.2,
       logepse=-1.0,
       logepsb=-2.0,
       g0=100.0,
       xiN=1.0,
       frequency=3.0e9,
       output_format="flux_density",
       density_function=nuclear_csm,
       density_parameters=(3.0, 17.0, 0.5, 2.0, 5.0),
       expansion=False,
   )

The tuple above is ``(log10_n_break, log10_r_break, inner_index,
outer_index, smoothness)``. ``tabulated_log_density`` supports differentiable
log-log interpolation of a fixed radial table.

The default initial swept mass assumes the density at ``1e10 cm`` fills the
interior uniformly. Supply ``log10_swept_mass_initial`` when the inner profile
requires a different enclosed mass. The current arbitrary-CSM backend is
spherically radial. It cannot be combined with the native refreshed-shell
backend.

Non-impulsive central engines
------------------------------

Non-refreshed ``_redback`` wrappers accept an ``engine_function`` that returns
cumulative injected energy in ``log10(erg)``. Built-in histories are
``constant_engine_cumulative_log10``,
``fallback_engine_cumulative_log10`` (constant luminosity followed by
``t^-5/3``), and ``tabulated_engine_cumulative_log10``.

.. code-block:: python

   import numpy as np

   from redback_jax.afterglow import fallback_engine_cumulative_log10

   flux_mjy = gaussian_redback(
       # Standard arguments omitted here for clarity.
       ...,
       engine_function=fallback_engine_cumulative_log10,
       engine_parameters=(52.0, np.log10(30.0 * 86400.0)),
       gamma_engine=1000.0,
   )

These parameters are total injected ``log10(energy / erg)`` and engine break
time in ``log10(seconds)``. Injection reaches each angular shell at retarded
engine time ``t_lab - R / (beta_engine c)``.

This is a powered thin-shell approximation intended for fast inference. It
updates shell inertia as energy becomes available, but it does not model the
spatially extended forward shock, reverse shock, and cocoon produced by
relativistic hydrodynamics. It is not a numerical reproduction of `De Colle &
Ramirez-Ruiz (2026) <https://arxiv.org/abs/2607.03548>`_. Their
injection-duration diagnostic and released simulation products should be used
for calibration before drawing quantitative conclusions from this
approximation.

Numerical configuration
-----------------------

``res`` controls both polar and azimuthal resolution, producing ``res**2``
angular patches. ``steps`` controls the radial grid. Both are static compilation
settings and should be varied in convergence tests.
