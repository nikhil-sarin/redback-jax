import jax.numpy as jnp
from jax import jit

import numpy as np

from redback_jax.utils.citation_wrapper import citation_wrapper


@jit  # Timing is ~10us slower than non-JAX version
def smooth_exponential_powerlaw(time, a_1, tpeak, alpha_1, alpha_2, smoothing_factor):
    """
    Smoothed version of exponential power law

    :param time: time array in seconds
    :param a_1: exponential amplitude scale
    :param alpha_1: first exponent
    :param alpha_2: second exponent
    :param tpeak: peak time in seconds
    :param smoothing_factor: controls transition smoothness (higher = smoother)
    :param kwargs: Additional parameters
    :return: In whatever units set by a_1
    """
    t_norm = jnp.asarray(time) / tpeak

    # Smooth transition function using tanh or similar
    transition = 0.5 * (1 + jnp.tanh(smoothing_factor * jnp.log(t_norm)))

    # Pre-peak behavior
    pre_peak = a_1 * (t_norm ** alpha_1)

    # Post-peak behavior
    post_peak = a_1 * (t_norm ** alpha_2)

    # Smooth combination
    return pre_peak * (1 - transition) + post_peak * transition


@jit  # Timing is same as the non-JAX version
def exp_rise_powerlaw_decline(t, t0, m_peak, tau_rise, alpha, t_peak, delta=0.5):
    """
    Compute a smooth light-curve model (in magnitudes) with an exponential rise
    transitioning into a power-law decline, with a smooth (blended) peak.
    In all filters the shape is determined by the same t0, tau_rise, alpha, and t_peak;
    only m_peak differs from filter to filter.

    For t < t0, the function returns jnp.nan.
    For t >= t0, the model is constructed as a blend of:

      Rising phase:
          m_rise(t)  = m_peak + 1.086 * ((t_peak - t) / tau_rise)
      Declining phase:
          m_decline(t) = m_peak + 2.5 * alpha * log10((t - t0)/(t_peak - t0))

    A smooth transition is achieved by the switching (weight) function:

          weight(t) = 0.5 * [1 + tanh((t - t_peak)/delta)]

    so that the final magnitude is:

          m(t) = (1 - weight(t)) * m_rise(t) + weight(t) * m_decline(t)

    At t = t_peak, weight = 0.5 and both m_rise and m_decline equal m_peak,
    ensuring a smooth peak.

    Parameters
    ----------
    t : array_like
        1D array of times (e.g., in modified Julian days) at which to evaluate the model.
    t0 : float
        Start time of the transient event (e.g., explosion), in MJD.
    m_peak : float or array_like
        Peak magnitude(s) at t = t_peak. If an array is provided, each element is taken
        to correspond to a different filter.
    tau_rise : float
        Characteristic timescale (in days) for the exponential rise.
    alpha : float
        Power-law decay index governing the decline.
    t_peak : float
        Time (in MJD) at peak brightness (must satisfy t_peak > t0).
    delta : float, optional
        Smoothing parameter (in days) controlling the width of the transition around t_peak.
        If not provided, defaults to 50% of (t_peak - t0).

    Returns
    -------
    m_model : ndarray
        If m_peak is an array (multiple filters), returns a 2D array of shape (n_times, n_filters);
        if m_peak is a scalar, returns a 1D array (with NaN for t < t0).

    Examples
    --------
    Single filter:

    >>> t = jnp.linspace(58990, 59050, 300)
    >>> model1 = exp_rise_powerlaw_decline(t, t0=59000, m_peak=17.0, tau_rise=3.0,
    ...                                     alpha=1.5, t_peak=59010)

    Multiple filters (e.g., g, r, i bands):

    >>> t = jnp.linspace(58990, 59050, 300)
    >>> m_peaks = jnp.array([17.0, 17.5, 18.0])
    >>> model_multi = exp_rise_powerlaw_decline(t, t0=59000, m_peak=m_peaks, tau_rise=3.0,
    ...                                          alpha=1.5, t_peak=59010)
    >>> print(model_multi.shape)  # Expected shape: (300, 3)
    """

    # Convert t to a JAX array and force 1D.
    t = jnp.asarray(t).flatten()

    # Define default smoothing parameter delta if not provided.
    delta = (t_peak - t0) * delta  # default: 50% of the interval [t0, t_peak]

    # Ensure m_peak is at least 1D (so a scalar becomes an array of length 1).
    m_peak = jnp.atleast_1d(m_peak)
    n_filters = m_peak.shape[0]
    n_times = t.shape[0]

    # Preallocate model magnitude array with shape (n_times, n_filters)
    m_model = jnp.full((n_times, n_filters), jnp.nan, dtype=float)

    # Reshape t into a column vector for broadcasting: shape (n_times, 1)
    t_col = t.reshape(-1, 1)

    # Compute the switching (weight) function: weight = 0 when t << t_peak, 1 when t >> t_peak.
    weight = 0.5 * (1 + jnp.tanh((t_col - t_peak) / delta))

    # Rising phase model: for t < t_peak the flux is rising toward peak.
    m_rise = m_peak[None, :] + 1.086 * ((t_peak - t_col) / tau_rise)

    # Declining phase model: power-law decline in flux gives a logarithmic increase in magnitude.
    ratio = (t_col - t0) / (t_peak - t0)
    m_decline = m_peak[None, :] + 2.5 * alpha * jnp.log10(ratio)

    # Blend the two components using the switching weight.
    # For t << t_peak, tanh term ≈ -1 so weight ~ 0 and m ~ m_rise.
    # For t >> t_peak, tanh term ≈ +1 so weight ~ 1 and m ~ m_decline.
    m_blend = (1 - weight) * m_rise + weight * m_decline

    # Use where to handle invalid times (t < t0) with NaN values
    valid = t >= t0
    m_model = jnp.where(valid[:, None], m_blend, jnp.nan)

    # If m_peak was given as a scalar, return a 1D array.
    if n_filters == 1:
        return m_model.flatten()
    return m_model


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2009A%26A...499..653B/abstract')
@jit  # Timing is same as the non-JAX version
def bazin_sne(time, aa, bb, t0, tau_rise, tau_fall, **kwargs):
    """
    Bazin function for CCSN light curves with vectorized inputs.

    :param time: time array in arbitrary units
    :param aa: array (or float) of normalisations, if array this is unique to each 'band'
    :param bb: array (or float) of additive constants, if array this is unique to each 'band'
    :param t0: start time
    :param tau_rise: exponential rise time
    :param tau_fall: exponential fall time
    :return: matrix of flux values in units set by AA
    """
    # Convert inputs to JAX arrays
    time = jnp.asarray(time)
    aa = jnp.atleast_1d(jnp.asarray(aa))
    bb = jnp.atleast_1d(jnp.asarray(bb))
    
    # Check if aa and bb have the same length
    if aa.shape[0] != bb.shape[0]:
        raise ValueError("Length of aa and bb must be the same.")
    
    # Reshape time for broadcasting: (n_times, 1)
    time_col = time.reshape(-1, 1)
    
    # Reshape aa and bb for broadcasting: (1, n_bands)
    aa_row = aa.reshape(1, -1)
    bb_row = bb.reshape(1, -1)
    
    # Compute the Bazin function for all bands simultaneously
    # Shape will be (n_times, n_bands)
    flux_matrix = aa_row * (jnp.exp(-((time_col - t0) / tau_fall)) / 
                           (1 + jnp.exp(-(time_col - t0) / tau_rise))) + bb_row
    
    # If original aa was scalar, return 1D array
    if aa.shape[0] == 1:
        return flux_matrix.flatten()
    return flux_matrix.T  # Return shape (n_bands, n_times) to match original behavior
