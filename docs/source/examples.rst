Examples
========

Bolometric inference with BlackJAX
------------------------------------

Fit an Arnett model to observed bolometric luminosities using NUTS via BlackJAX.
Because all bolometric models return ``log10_lbol``, the likelihood is computed
in log10 space — this is both float32-safe and numerically well-conditioned.

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", False)

   import jax.numpy as jnp
   import blackjax
   from redback_jax.models.supernova_models import arnett_bolometric

   # Simulated observations
   time = jnp.linspace(5.0, 60.0, 30, dtype=jnp.float32)
   true_params = dict(f_nickel=0.4, mej=1.2, vej=9000.0, kappa=0.1, kappa_gamma=10.0)
   log10_lbol_true = arnett_bolometric(time, **true_params)
   log10_lbol_obs  = log10_lbol_true + 0.05 * jax.random.normal(jax.random.PRNGKey(0), time.shape)
   sigma = jnp.full_like(time, 0.05)

   # Log-likelihood in log10 space
   @jax.jit
   def log_likelihood(params):
       log10_lbol = arnett_bolometric(time, **params,
                                       vej=9000.0, kappa=0.1, kappa_gamma=10.0)
       return -0.5 * jnp.sum(((log10_lbol_obs - log10_lbol) / sigma)**2)

Photometry fitting with the inference API
------------------------------------------

Use the clean ``Prior`` / ``Likelihood`` / ``NestedSampler`` / ``MCMCSampler``
API for end-to-end Bayesian photometric fitting.  The ``Likelihood`` class
handles bandflux integration internally and is JIT-safe.

.. code-block:: python

   import jax
   from redback_jax.inference import Prior, Uniform, Likelihood, NestedSampler, MCMCSampler
   from redback_jax.utils import luminosity_distance_cm

   REDSHIFT = 0.01
   DL_CM    = luminosity_distance_cm(REDSHIFT)

   prior = Prior([
       Uniform(58580, 58620,  name='t0'),
       Uniform(0.05,  0.30,   name='f_nickel'),
       Uniform(0.5,   3.0,    name='mej'),
       Uniform(3000,  12000,  name='vej'),
   ])

   # transient.time (MJD), transient.y (AB mag), transient.y_err, transient.bands
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
   )

   # Nested sampling
   ns_result = NestedSampler(likelihood, prior, outdir='results/').run(jax.random.PRNGKey(0))
   ns_result.summary()

   # Or MCMC with NUTS
   mcmc_result = MCMCSampler(
       likelihood, prior, n_warmup=500, n_samples=2000, n_chains=4
   ).run(jax.random.PRNGKey(1))
   mcmc_result.summary()

Kilonova bolometric fitting
----------------------------

The kilonova models use an energy-normalised ODE scan for float32 safety.
They also return ``log10_lbol``:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", False)
   import jax.numpy as jnp
   from redback_jax.models.kilonova import metzger_kilonova_bolometric

   time = jnp.linspace(0.5, 20.0, 50, dtype=jnp.float32)

   log10_lbol = metzger_kilonova_bolometric(
       time,
       mej=0.05,    # solar masses
       vej=0.2,     # fraction of c
       beta=3.0,    # velocity profile slope
       kappa=1.0,   # cm^2/g
   )
   # Typical range: log10_lbol ~ [39, 42]

Magnetar-boosted kilonova:

.. code-block:: python

   from redback_jax.models.kilonova import magnetar_boosted_kilonova_bolometric

   log10_lbol = magnetar_boosted_kilonova_bolometric(
       time,
       mej=0.05, vej=0.2, beta=3.0, kappa=1.0,
       p0=1.0,          # spin period in ms
       bp=1.0,          # B-field in units of 1e14 G
       mass_ns=1.4,     # neutron star mass in solar masses
       theta_pb=0.0,    # spin-B field angle in radians
   )

Shock-powered models
---------------------

Shock cooling and cocoon models also return ``log10_lbol``. Note that
``shock_cooling_bolometric`` takes log10 inputs for the large physical
quantities:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", False)
   import jax.numpy as jnp
   from redback_jax.models.shock_powered_models import (
       shock_cooling_bolometric,
       shocked_cocoon_bolometric,
   )

   time = jnp.linspace(0.1, 10.0, 50, dtype=jnp.float32)

   # Inputs are log10(mass/Msun), log10(radius/cm), log10(energy/erg)
   log10_lbol = shock_cooling_bolometric(
       time,
       log10_mass=-2.0,     # 0.01 solar masses
       log10_radius=13.0,   # 1e13 cm
       log10_energy=51.0,   # 1e51 erg
       nn=10.0,             # outer density power-law slope
       delta=1.1,           # inner density power-law slope
       kappa=0.2,           # opacity in cm^2/g
   )

   log10_lbol = shocked_cocoon_bolometric(
       time,
       mej=0.01,
       vej=0.1,             # fraction of c
       eta=2.0,
       tshock=1.0,          # seconds
       shocked_fraction=0.5,
       cos_theta_cocoon=0.5,
       kappa=0.1,
   )
