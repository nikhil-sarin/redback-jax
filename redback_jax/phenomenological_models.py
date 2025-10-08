import jax.numpy as jnp
from jax import jit


@jit
def smooth_exponential_powerlaw(time, a_1, tpeak, alpha_1, alpha_2, smoothing_factor, **kwargs):
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
