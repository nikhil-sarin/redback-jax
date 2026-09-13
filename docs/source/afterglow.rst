Native Redback afterglows
=========================

The native ``_redback`` family shares a composable JAX engine for jet
structure, blast-wave dynamics, synchrotron radiation, and observer-time
integration. The public wrappers preserve Redback's parameter names and return
flux density in mJy or AB magnitude.

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
spherically radial and impulsive. It cannot yet be combined with the native
refreshed-shell backend; continuous engine injection is a separate planned
dynamics backend.

Numerical configuration
-----------------------

``res`` controls both polar and azimuthal resolution, producing ``res**2``
angular patches. ``steps`` controls the radial grid. Both are static compilation
settings and should be varied in convergence tests.
