"""
Prior distributions for redback-jax inference.

Usage::

    from redback_jax.inference import Uniform, LogUniform, Prior

    prior = Prior([
        Uniform(0.05, 0.20, name='f_nickel'),
        Uniform(0.8,  2.0,  name='mej'),
        Uniform(3000, 8000, name='vej'),
        Uniform(58580, 58620, name='t0'),
    ])

    # Draw from prior
    key = jax.random.PRNGKey(0)
    params = prior.sample(key)          # dict {name: scalar}
    particles = prior.sample_n(key, 100)  # jnp.ndarray (100, n_params)

    # Evaluate log-prior
    log_p = prior.log_prob(particles[0])
"""

import math as _math

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Sequence, Union


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _Distribution:
    """Base class for prior distributions."""

    def __init__(self, name: str):
        self.name = name

    def sample(self, key: jax.Array) -> jnp.ndarray:
        raise NotImplementedError

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    # Bounds used to initialise live points from the prior
    @property
    def low(self) -> float:
        raise NotImplementedError

    @property
    def high(self) -> float:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------

class Uniform(_Distribution):
    """Uniform prior between *minimum* and *maximum*.

    Parameters
    ----------
    minimum, maximum : float
        Support of the distribution.
    name : str
        Parameter name (used as dict key in likelihood calls).

    Examples
    --------
    >>> p = Uniform(0.05, 0.2, name='f_nickel')
    >>> p.log_prob(jnp.array(0.1))   # log(1 / (0.2 - 0.05))
    """

    def __init__(self, minimum: float, maximum: float, *, name: str):
        super().__init__(name)
        self._low  = float(minimum)
        self._high = float(maximum)
        self._log_prob_val = -_math.log(maximum - minimum)

    @property
    def low(self) -> float:
        return self._low

    @property
    def high(self) -> float:
        return self._high

    def sample(self, key: jax.Array) -> jnp.ndarray:
        return jax.random.uniform(key, minval=self._low, maxval=self._high)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        in_support = (value >= self._low) & (value <= self._high)
        return jnp.where(in_support, self._log_prob_val, -jnp.inf)

    def __repr__(self) -> str:
        return f"Uniform({self._low}, {self._high}, name={self.name!r})"


# ---------------------------------------------------------------------------
# LogUniform
# ---------------------------------------------------------------------------

class LogUniform(_Distribution):
    """Log-uniform (Jeffreys) prior between *minimum* and *maximum*.

    The density is proportional to 1/x, normalised over [minimum, maximum].

    Parameters
    ----------
    minimum, maximum : float
        Support of the distribution (must be > 0).
    name : str
        Parameter name.

    Examples
    --------
    >>> p = LogUniform(1e-2, 1e2, name='kappa')
    """

    def __init__(self, minimum: float, maximum: float, *, name: str):
        super().__init__(name)
        if minimum <= 0:
            raise ValueError("LogUniform requires minimum > 0")
        self._low  = float(minimum)
        self._high = float(maximum)
        self._log_norm = _math.log(_math.log(maximum / minimum))

    @property
    def low(self) -> float:
        return self._low

    @property
    def high(self) -> float:
        return self._high

    def sample(self, key: jax.Array) -> jnp.ndarray:
        log_lo = _math.log(self._low)
        log_hi = _math.log(self._high)
        return jnp.exp(jax.random.uniform(key, minval=log_lo, maxval=log_hi))

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        in_support = (value >= self._low) & (value <= self._high)
        return jnp.where(
            in_support,
            -jnp.log(jnp.maximum(value, 1e-300)) - self._log_norm,
            -jnp.inf,
        )

    def __repr__(self) -> str:
        return f"LogUniform({self._low}, {self._high}, name={self.name!r})"


# ---------------------------------------------------------------------------
# Gaussian (for constrained parameters)
# ---------------------------------------------------------------------------

class Gaussian(_Distribution):
    """Gaussian prior (truncated to finite support by the sampler's hard bounds).

    Parameters
    ----------
    mu, sigma : float
        Mean and standard deviation.
    name : str
        Parameter name.
    """

    def __init__(self, mu: float, sigma: float, *, name: str,
                 minimum: float = -jnp.inf, maximum: float = jnp.inf):
        super().__init__(name)
        self.mu    = float(mu)
        self.sigma = float(sigma)
        self._low  = float(minimum)
        self._high = float(maximum)

    @property
    def low(self) -> float:
        return self._low

    @property
    def high(self) -> float:
        return self._high

    def sample(self, key: jax.Array) -> jnp.ndarray:
        return jax.random.normal(key) * self.sigma + self.mu

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        in_support = (value >= self._low) & (value <= self._high)
        log_p = -0.5 * ((value - self.mu) / self.sigma) ** 2 - _math.log(
            self.sigma * _math.sqrt(2 * _math.pi)
        )
        return jnp.where(in_support, log_p, -jnp.inf)

    def __repr__(self) -> str:
        return f"Gaussian(mu={self.mu}, sigma={self.sigma}, name={self.name!r})"


# ---------------------------------------------------------------------------
# Constraint (physics-based differentiable barrier)
# ---------------------------------------------------------------------------

class Constraint:
    """Differentiable physics constraint added to the log-prior.

    ``fn(param_dict)`` must return a *slack* scalar: positive means the
    constraint is satisfied, negative means it is violated.  The
    log-contribution is ``-softplus(-20 * slack / scale)``, which is ≈ 0
    when slack >> 0 and steeply negative when violated, with gradients
    finite everywhere so NUTS works.

    Parameters
    ----------
    fn : callable
        ``fn(param_dict: dict) -> slack_scalar``.  Fixed parameters should
        be captured in the closure.
    scale : float
        Normalises the slack before applying the barrier.  Set to a typical
        value of the slack so the transition width is ~5% of that scale.
    name : str
        Human-readable label used in repr.
    """

    def __init__(self, fn, scale: float = 1.0, name: str = 'constraint'):
        self.fn    = fn
        self.scale = float(scale)
        self.name  = name

    def log_prob(self, param_dict: dict) -> jnp.ndarray:
        slack = self.fn(param_dict)
        return -jax.nn.softplus(-20.0 * slack / self.scale)

    def __repr__(self) -> str:
        return f"Constraint(name={self.name!r}, scale={self.scale})"


# ---------------------------------------------------------------------------
# Factory: CSM photosphere validity
# ---------------------------------------------------------------------------

def make_csm_photosphere_constraint_log(fixed_kappa: float = 0.1,
                                        scale: float = 0.1) -> Constraint:
    """Like make_csm_photosphere_constraint but expects log_rho and log_r0 in param_dict.

    Use this when rho and r0 are sampled in log-space (Uniform priors on their
    logarithms) so that gradients through the constraint are smooth.
    """
    _AU    = 1.496e13
    _MSUN  = 1.989e33
    _kappa = float(fixed_kappa)

    def _csm_slack(params: dict) -> jnp.ndarray:
        r0       = jnp.exp(params['log_r0'])
        csm_mass = params['csm_mass']
        rho      = jnp.exp(params['log_rho'])
        eta      = params['eta']

        r0_cm      = r0 * _AU
        csm_mass_g = csm_mass * _MSUN
        qq         = rho * r0_cm ** eta

        eta_safe = jnp.where(jnp.abs(eta - 1.0) < 0.05,
                             jnp.asarray(0.95, dtype=eta.dtype), eta)

        radius_csm = jnp.maximum(
            ((3.0 - eta) / (4.0 * jnp.pi * jnp.maximum(qq, 1e-300))
             * csm_mass_g
             + r0_cm ** (3.0 - eta)) ** (1.0 / (3.0 - eta)),
            r0_cm,
        )

        r_phot = jnp.abs(
            (-2.0 * (1.0 - eta_safe)
             / (3.0 * _kappa * jnp.maximum(qq, 1e-300))
             + radius_csm ** (1.0 - eta_safe)
             ) ** (1.0 / (1.0 - eta_safe))
        )

        mass_thresh = jnp.abs(
            4.0 * jnp.pi * jnp.maximum(qq, 1e-300) / (3.0 - eta)
            * (r_phot ** (3.0 - eta) - r0_cm ** (3.0 - eta))
        )

        return (csm_mass_g - mass_thresh) / jnp.maximum(csm_mass_g, 1e-300)

    return Constraint(_csm_slack, scale=scale, name='csm_photosphere_inside_shell')


def make_csm_photosphere_constraint(fixed_kappa: float = 0.1,
                                     scale: float = 0.1) -> Constraint:
    """Return a Constraint that enforces mass_csm_threshold < csm_mass.

    The photosphere must lie inside the CSM shell for the interaction model
    to be physical.  Captures ``fixed_kappa`` so that only the free
    parameters ``rho``, ``eta``, ``r0``, ``csm_mass`` are needed from the
    prior's param_dict.

    Parameters
    ----------
    fixed_kappa : float
        Opacity κ in cm²/g (typically fixed at 0.1).
    scale : float
        Normalisation for the log-barrier transition width.
    """
    _AU      = 1.496e13   # cm/AU
    _MSUN    = 1.989e33   # g/M_sun
    _kappa   = float(fixed_kappa)

    def _csm_slack(params: dict) -> jnp.ndarray:
        r0       = params['r0']
        csm_mass = params['csm_mass']
        rho      = params['rho']
        eta      = params['eta']

        r0_cm      = r0 * _AU
        csm_mass_g = csm_mass * _MSUN
        qq         = rho * r0_cm ** eta

        # Guard eta ≈ 1 where (1-eta) exponent causes a singularity.
        eta_safe = jnp.where(jnp.abs(eta - 1.0) < 0.05,
                             jnp.asarray(0.95, dtype=eta.dtype), eta)

        radius_csm = jnp.maximum(
            ((3.0 - eta) / (4.0 * jnp.pi * jnp.maximum(qq, 1e-300))
             * csm_mass_g
             + r0_cm ** (3.0 - eta)) ** (1.0 / (3.0 - eta)),
            r0_cm,
        )

        r_phot = jnp.abs(
            (-2.0 * (1.0 - eta_safe)
             / (3.0 * _kappa * jnp.maximum(qq, 1e-300))
             + radius_csm ** (1.0 - eta_safe)
             ) ** (1.0 / (1.0 - eta_safe))
        )

        mass_thresh = jnp.abs(
            4.0 * jnp.pi * jnp.maximum(qq, 1e-300) / (3.0 - eta)
            * (r_phot ** (3.0 - eta) - r0_cm ** (3.0 - eta))
        )

        return (csm_mass_g - mass_thresh) / jnp.maximum(csm_mass_g, 1e-300)

    return Constraint(_csm_slack, scale=scale,
                      name='csm_photosphere_inside_shell')


# ---------------------------------------------------------------------------
# Prior (composite)
# ---------------------------------------------------------------------------

class Prior:
    """A composite prior built from a list of 1-D distributions.

    Parameters
    ----------
    distributions : list of _Distribution
        One distribution per free parameter.  The list order determines the
        column order of the parameter vector passed to the likelihood.

    Examples
    --------
    >>> prior = Prior([
    ...     Uniform(58580, 58620, name='t0'),
    ...     Uniform(0.05, 0.20,   name='f_nickel'),
    ...     Uniform(0.8,  2.0,    name='mej'),
    ...     Uniform(3000, 8000,   name='vej'),
    ... ])
    >>> particles = prior.sample_n(jax.random.PRNGKey(0), 100)  # (100, 4)
    >>> log_p = prior.log_prob(particles[0])                    # scalar
    """

    def __init__(self, distributions: List[Union[_Distribution, Constraint]]):
        # Separate parametric distributions from physics constraints.
        self.distributions = [d for d in distributions
                              if isinstance(d, _Distribution)]
        self.constraints   = [d for d in distributions
                              if isinstance(d, Constraint)]
        self.names    = [d.name for d in self.distributions]
        self.n_params = len(self.distributions)
        self._lows    = jnp.array([d.low  for d in self.distributions])
        self._highs   = jnp.array([d.high for d in self.distributions])

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, key: jax.Array) -> dict:
        """Draw one sample; returns a dict {name: scalar}."""
        keys = jax.random.split(key, self.n_params)
        return {d.name: d.sample(k) for d, k in zip(self.distributions, keys)}

    def sample_n(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Draw *n* samples; returns an array of shape ``(n, n_params)``."""
        keys = jax.random.split(key, self.n_params)
        columns = []
        for d, k in zip(self.distributions, keys):
            # draw n values per parameter
            cols = jax.vmap(lambda ki: d.sample(ki))(jax.random.split(k, n))
            columns.append(cols)
        return jnp.stack(columns, axis=1)   # (n, n_params)

    # ------------------------------------------------------------------
    # Log-probability (JAX-traceable)
    # ------------------------------------------------------------------

    def log_prob(self, params: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the joint log-prior for a parameter vector of shape ``(n_params,)``.

        Includes contributions from all ``Constraint`` instances (differentiable
        log-barrier terms) in addition to the marginal distribution log-probs.
        JAX-traceable and usable inside ``@jax.jit``.
        """
        log_p = jnp.array(0.0)
        for i, d in enumerate(self.distributions):
            log_p = log_p + d.log_prob(params[i])
        if self.constraints:
            param_dict = {d.name: params[i]
                          for i, d in enumerate(self.distributions)}
            for c in self.constraints:
                log_p = log_p + c.log_prob(param_dict)
        return log_p

    def log_prob_fn(self):
        """Return a pure JAX-traceable function ``params -> log_prior``."""
        dists       = self.distributions
        constraints = self.constraints

        def _fn(params: jnp.ndarray) -> jnp.ndarray:
            log_p = jnp.array(0.0)
            for i, d in enumerate(dists):
                log_p = log_p + d.log_prob(params[i])
            if constraints:
                param_dict = {d.name: params[i] for i, d in enumerate(dists)}
                for c in constraints:
                    log_p = log_p + c.log_prob(param_dict)
            return log_p

        return _fn

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def params_to_dict(self, params: jnp.ndarray) -> dict:
        """Convert a parameter vector ``(n_params,)`` to a name-keyed dict."""
        return {d.name: params[i] for i, d in enumerate(self.distributions)}

    def dict_to_params(self, d: dict) -> jnp.ndarray:
        """Convert a name-keyed dict to a parameter vector."""
        return jnp.array([d[name] for name in self.names])

    def __len__(self) -> int:
        return self.n_params

    def __repr__(self) -> str:
        items = "\n  ".join(repr(d) for d in self.distributions)
        if self.constraints:
            items += "\n  " + "\n  ".join(repr(c) for c in self.constraints)
        return f"Prior([\n  {items}\n])"
