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
from typing import List, Sequence


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

    def __init__(self, distributions: List[_Distribution]):
        self.distributions = list(distributions)
        self.names = [d.name for d in self.distributions]
        self.n_params = len(self.distributions)
        self._lows  = jnp.array([d.low  for d in self.distributions])
        self._highs = jnp.array([d.high for d in self.distributions])

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

        This is JAX-traceable and can be used inside ``@jax.jit``.
        """
        log_p = jnp.array(0.0)
        for i, d in enumerate(self.distributions):
            log_p = log_p + d.log_prob(params[i])
        return log_p

    def log_prob_fn(self):
        """Return a pure JAX-traceable function ``params -> log_prior``."""
        dists = self.distributions  # captured in closure

        def _fn(params: jnp.ndarray) -> jnp.ndarray:
            log_p = jnp.array(0.0)
            for i, d in enumerate(dists):
                log_p = log_p + d.log_prob(params[i])
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
        return f"Prior([\n  {items}\n])"
