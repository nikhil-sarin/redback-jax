from redback_jax.constants import *


def calc_kcorrected_properties(frequency, redshift, time):
    """
    Perform k-correction

    :param frequency: observer frame frequency
    :param redshift: source redshift
    :param time: observer frame time
    :return: k-corrected frequency and source frame time
    """
    time = time / (1 + redshift)
    frequency = frequency * (1 + redshift)
    return frequency, time


def lambda_to_nu(wavelength):
    """
    :param wavelength: wavelength in Angstrom
    :return: frequency in Hertz
    """
    return speed_of_light_si / (wavelength * 1.e-10)


def nu_to_lambda(frequency):
    """
    :param frequency: frequency in Hertz
    :return: wavelength in Angstrom
    """
    return 1.e10 * (speed_of_light_si / frequency)