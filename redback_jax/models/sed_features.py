"""
JAX-friendly classes for supernova modeling.
"""

from jax import jit
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

from redback_jax.constants import *


class SEDFeatures:
    """A class representing a spectral feature(s) in the SED. This implements the PyTree interface for JAX,
    so it can be used in JIT-compiled functions.
    
    :param rest_wavelengths: Central wavelengths in Angstroms
    :param sigmas: Gaussian widths in Angstroms
    :param amplitudes: Amplitudes (negative=absorption, positive=emission), percentage of continuum
        (e.g., -0.4 = 40% absorption)
    :param t_starts: Start time in seconds
    :param t_ends: End time in seconds
    :param rise_times: Rise times in seconds (default: 2 days)
    :param fall_times: Fall times in seconds (default: 5 days)
    """
    def __init__(
            self,
            rest_wavelengths,
            sigmas,
            amplitudes,
            t_starts,
            t_ends,
            rise_times=2.0 * 24.0 * 3600.0,
            fall_times=5.0 * 24.0 * 3600.0,
        ):
        self.rest_wavelengths = jnp.atleast_1d(rest_wavelengths)
        self.sigmas = jnp.atleast_1d(sigmas)
        self.amplitudes = jnp.atleast_1d(amplitudes)
        self.t_starts = jnp.atleast_1d(t_starts)
        self.t_ends = jnp.atleast_1d(t_ends)
        self.rise_times = jnp.atleast_1d(rise_times)
        self.fall_times = jnp.atleast_1d(fall_times)

    @classmethod
    def from_feature_list(cls, feature_list):
        """Create SEDFeatures from a list of feature dictionaries.

        :param feature_list: List of dictionaries, each with keys:
            'rest_wavelength', 'sigma', 'amplitude', 't_start', 't_end', 'rise_time', 'fall_time'
        :return: SEDFeatures object
        """
        rest_wavelengths = []
        sigmas = []
        amplitudes = []
        t_starts = []
        t_ends = []
        rise_times = []
        fall_times = []

        for feature in feature_list:
            rest_wavelengths.append(feature["rest_wavelength"])
            sigmas.append(feature["sigma"])
            amplitudes.append(feature["amplitude"])
            t_starts.append(feature["t_start"])
            t_ends.append(feature["t_end"])
            rise_times.append(feature.get("rise_time", 2.0 * 24.0 * 3600.0))
            fall_times.append(feature.get("fall_time", 5.0 * 24.0 * 3600.0))

        return cls(
            jnp.array(rest_wavelengths),
            jnp.array(sigmas),
            jnp.array(amplitudes),
            jnp.array(t_starts),
            jnp.array(t_ends),
            jnp.array(rise_times),
            jnp.array(fall_times),
        )

    def tree_flatten(self):
        children = (
            self.rest_wavelengths,
            self.sigmas,
            self.amplitudes,
            self.t_starts,
            self.t_ends,
            self.rise_times,
            self.fall_times,
        )
        aux_data = None
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jit
    def calculate_smooth_evolution(self, time):
        """Calculate smooth transitions for a set of features.

        :param time: Time array in seconds (source frame)

        :return: time_factors array in [0, 1] representing the evolution of the feature over time
        """
        t_starts = self.t_starts
        t_ends = self.t_ends
        rise_times = self.rise_times
        fall_times = self.fall_times

        # Broadcast time array for vectorized operations
        time_grid = time[None, :]  # Shape: (1, n_times)

        # Compute the masks for each phase.
        in_rise = (time_grid >= t_starts[:, None]) & (time_grid < (t_starts + rise_times)[:, None])
        in_plateau = (time_grid >= (t_starts + rise_times)[:, None]) & (time_grid < (t_ends - fall_times)[:, None])
        in_fall = (time_grid >= (t_ends - fall_times)[:, None]) & (time_grid < t_ends[:, None])

        # Calculate smooth transitions
        # Rise phase
        x_rise = (time_grid - t_starts[:, None]) / rise_times[:, None]
        rise_factors = 0.5 * (1 + jnp.tanh(6 * (x_rise - 0.5)))

        # Fall phase
        x_fall = (t_ends[:, None] - time_grid) / fall_times[:, None]
        fall_factors = 0.5 * (1 + jnp.tanh(6 * (x_fall - 0.5)))

        # Combine all phases. This will use 0.0 outside the time ranges.
        time_factors = (
            in_rise.astype(float) * rise_factors +
            in_plateau.astype(float) * 1.0 +
            in_fall.astype(float) * fall_factors
        )
        return time_factors

register_pytree_node(SEDFeatures, SEDFeatures.tree_flatten, SEDFeatures.tree_unflatten)

# A Constant Non-feature object. The only setting that really matters is that amplitude is 0.0.
NO_SED_FEATURES = SEDFeatures(100.0, 1.0, 0.0, 0.0, 10.0, 1.0, 2.0)


@jit
def apply_sed_feature(features, base_flux, frequency, time):
    """Apply spectral features completely vectorized.
    
    :param features: SEDFeatures object
    :param base_flux as a 2-d array (time, wavelength) in erg/s/Hz/cm^2
    :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number.
                      In source frame.
    :param time: time array in seconds (source frame).

    :return: modified flux_density as a 2-d array (time, wavelength) in erg/s/Hz/cm^2
    """
    # Convert frequency to wavelength
    freq_for_wavelength = jnp.atleast_1d(frequency)
    wavelength_angstrom = speed_of_light / freq_for_wavelength * 1e8

    # Calculate the Gaussian profile.
    wl_diff = wavelength_angstrom[None, :] - features.rest_wavelengths[:, None]
    gaussian_profiles = jnp.exp(-0.5 * (wl_diff / features.sigmas[:, None]) ** 2)

    # Calculate the time factors for this feature
    time_factors = features.calculate_smooth_evolution(time)

    # flux is (time, freq)
    # Broadcast to (n_features, n_times, n_freq)
    feature_contributions = (
        features.amplitudes[:, None, None] *
        time_factors[:, :, None] *
        gaussian_profiles[:, None, :]
    )

    total_feature_factor = 1.0 + jnp.sum(feature_contributions, axis=0)
    return base_flux * total_feature_factor
