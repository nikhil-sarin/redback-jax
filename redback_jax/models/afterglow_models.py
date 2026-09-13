"""JAX implementations of Redback's native structured-jet afterglows."""

import jax.numpy as jnp
from wcosmo import wcosmo

from redback_jax.afterglow import native_afterglow_flux_density
from redback_jax.utils.citation_wrapper import citation_wrapper
from redback_jax.utils.cosmology import MPC_TO_CM, PLANCK18_H0, PLANCK18_OM0


def _format_afterglow_output(flux_mjy, output_format):
    if output_format == "flux_density":
        return flux_mjy
    if output_format == "magnitude":
        return -2.5 * jnp.log10(flux_mjy * 1.0e-3 / 3631.0)
    raise ValueError("output_format must be 'flux_density' or 'magnitude'")


def _native_redback_wrapper(
    time,
    redshift,
    thv,
    loge0,
    thc,
    thj,
    logn0,
    p,
    logepse,
    logepsb,
    g0,
    xiN,
    *,
    structure_kind,
    structure_energy,
    structure_gamma,
    frequency,
    output_format="flux_density",
    k=0.0,
    expansion=True,
    a1=1.0,
    res=50,
    steps=250,
    cosmo_H0=PLANCK18_H0,
    cosmo_Om0=PLANCK18_OM0,
    refreshed=False,
    gamma_injection=2.0,
    energy_factor=1.0,
    injection_index=0.0,
    density_function=None,
    density_parameters=None,
    log10_swept_mass_initial=None,
):
    distance = (
        wcosmo.luminosity_distance(redshift, cosmo_H0, cosmo_Om0).value * MPC_TO_CM
    )
    flux = native_afterglow_flux_density(
        time=time,
        frequency=frequency,
        redshift=redshift,
        theta_observer=thv,
        log10_energy=loge0,
        theta_core=thc,
        theta_jet=thj,
        log10_density=logn0,
        electron_index=p,
        log10_epsilon_e=logepse,
        log10_epsilon_b=logepsb,
        gamma_initial=g0,
        accelerated_fraction=xiN,
        log10_luminosity_distance=jnp.log10(distance),
        structure_kind=structure_kind,
        structure_energy=structure_energy,
        structure_gamma=structure_gamma,
        density_index=k,
        expansion=expansion,
        expansion_index=a1,
        resolution=res,
        steps=steps,
        refreshed=refreshed,
        gamma_injection=gamma_injection,
        energy_factor=energy_factor,
        injection_index=injection_index,
        density_function=density_function,
        density_parameters=density_parameters,
        log10_swept_mass_initial=log10_swept_mass_initial,
    )
    return _format_afterglow_output(flux, output_format)


_CITATION = "redback, https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.2581L/abstract"


@citation_wrapper(_CITATION)
def tophat_redback(
    time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, g0, xiN, **kwargs
):
    return _native_redback_wrapper(
        time,
        redshift,
        thv,
        loge0,
        thc,
        thc,
        logn0,
        p,
        logepse,
        logepsb,
        g0,
        xiN,
        structure_kind="tophat",
        structure_energy=0.01,
        structure_gamma=0.5,
        **kwargs,
    )


@citation_wrapper(_CITATION)
def gaussian_redback(
    time, redshift, thv, loge0, thc, thj, logn0, p, logepse, logepsb, g0, xiN, **kwargs
):
    return _native_redback_wrapper(
        time,
        redshift,
        thv,
        loge0,
        thc,
        thj,
        logn0,
        p,
        logepse,
        logepsb,
        g0,
        xiN,
        structure_kind="gaussian",
        structure_energy=0.01,
        structure_gamma=0.5,
        **kwargs,
    )


@citation_wrapper(_CITATION)
def twocomponent_redback(
    time, redshift, thv, loge0, thc, thj, logn0, p, logepse, logepsb, g0, xiN, **kwargs
):
    return _native_redback_wrapper(
        time,
        redshift,
        thv,
        loge0,
        thc,
        thj,
        logn0,
        p,
        logepse,
        logepsb,
        g0,
        xiN,
        structure_kind="two_component",
        structure_energy=kwargs.pop("ss", 0.01),
        structure_gamma=kwargs.pop("aa", 4.0),
        **kwargs,
    )


@citation_wrapper(_CITATION)
def powerlaw_redback(
    time, redshift, thv, loge0, thc, thj, logn0, p, logepse, logepsb, g0, xiN, **kwargs
):
    return _native_redback_wrapper(
        time,
        redshift,
        thv,
        loge0,
        thc,
        thj,
        logn0,
        p,
        logepse,
        logepsb,
        g0,
        xiN,
        structure_kind="powerlaw",
        structure_energy=kwargs.pop("ss", 3.0),
        structure_gamma=kwargs.pop("aa", -3.0),
        **kwargs,
    )


@citation_wrapper(_CITATION)
def alternativepowerlaw_redback(
    time, redshift, thv, loge0, thc, thj, logn0, p, logepse, logepsb, g0, xiN, **kwargs
):
    return _native_redback_wrapper(
        time,
        redshift,
        thv,
        loge0,
        thc,
        thj,
        logn0,
        p,
        logepse,
        logepsb,
        g0,
        xiN,
        structure_kind="alternative_powerlaw",
        structure_energy=kwargs.pop("ss", 3.0),
        structure_gamma=kwargs.pop("aa", 3.0),
        **kwargs,
    )


@citation_wrapper(_CITATION)
def doublegaussian_redback(
    time, redshift, thv, loge0, thc, thj, logn0, p, logepse, logepsb, g0, xiN, **kwargs
):
    return _native_redback_wrapper(
        time,
        redshift,
        thv,
        loge0,
        thc,
        thj,
        logn0,
        p,
        logepse,
        logepsb,
        g0,
        xiN,
        structure_kind="double_gaussian",
        structure_energy=kwargs.pop("ss", 0.1),
        structure_gamma=kwargs.pop("aa", 0.5),
        **kwargs,
    )


def _refreshed_kwargs(g1, et, s1):
    return dict(
        refreshed=True,
        gamma_injection=g1,
        energy_factor=et,
        injection_index=s1,
    )


@citation_wrapper(_CITATION)
def tophat_redback_refreshed(
    time,
    redshift,
    thv,
    loge0,
    thc,
    g1,
    et,
    s1,
    logn0,
    p,
    logepse,
    logepsb,
    g0,
    xiN,
    **kwargs,
):
    return _native_redback_wrapper(
        time,
        redshift,
        thv,
        loge0,
        thc,
        thc,
        logn0,
        p,
        logepse,
        logepsb,
        g0,
        xiN,
        structure_kind="tophat",
        structure_energy=0.01,
        structure_gamma=0.5,
        **_refreshed_kwargs(g1, et, s1),
        **kwargs,
    )


@citation_wrapper(_CITATION)
def gaussian_redback_refreshed(
    time,
    redshift,
    thv,
    loge0,
    thc,
    thj,
    g1,
    et,
    s1,
    logn0,
    p,
    logepse,
    logepsb,
    g0,
    xiN,
    **kwargs,
):
    return _native_redback_wrapper(
        time,
        redshift,
        thv,
        loge0,
        thc,
        thj,
        logn0,
        p,
        logepse,
        logepsb,
        g0,
        xiN,
        structure_kind="gaussian",
        structure_energy=0.01,
        structure_gamma=0.5,
        **_refreshed_kwargs(g1, et, s1),
        **kwargs,
    )


@citation_wrapper(_CITATION)
def twocomponent_redback_refreshed(
    time,
    redshift,
    thv,
    loge0,
    thc,
    thj,
    g1,
    et,
    s1,
    logn0,
    p,
    logepse,
    logepsb,
    g0,
    xiN,
    **kwargs,
):
    return _native_redback_wrapper(
        time,
        redshift,
        thv,
        loge0,
        thc,
        thj,
        logn0,
        p,
        logepse,
        logepsb,
        g0,
        xiN,
        structure_kind="two_component",
        structure_energy=kwargs.pop("ss", 0.01),
        structure_gamma=kwargs.pop("aa", 4.0),
        **_refreshed_kwargs(g1, et, s1),
        **kwargs,
    )


@citation_wrapper(_CITATION)
def powerlaw_redback_refreshed(
    time,
    redshift,
    thv,
    loge0,
    thc,
    thj,
    g1,
    et,
    s1,
    logn0,
    p,
    logepse,
    logepsb,
    g0,
    xiN,
    **kwargs,
):
    return _native_redback_wrapper(
        time,
        redshift,
        thv,
        loge0,
        thc,
        thj,
        logn0,
        p,
        logepse,
        logepsb,
        g0,
        xiN,
        structure_kind="powerlaw",
        structure_energy=kwargs.pop("ss", 3.0),
        structure_gamma=kwargs.pop("aa", -3.0),
        **_refreshed_kwargs(g1, et, s1),
        **kwargs,
    )


@citation_wrapper(_CITATION)
def alternativepowerlaw_redback_refreshed(
    time,
    redshift,
    thv,
    loge0,
    thc,
    thj,
    g1,
    et,
    s1,
    logn0,
    p,
    logepse,
    logepsb,
    g0,
    xiN,
    **kwargs,
):
    return _native_redback_wrapper(
        time,
        redshift,
        thv,
        loge0,
        thc,
        thj,
        logn0,
        p,
        logepse,
        logepsb,
        g0,
        xiN,
        structure_kind="alternative_powerlaw",
        structure_energy=kwargs.pop("ss", 3.0),
        structure_gamma=kwargs.pop("aa", 3.0),
        **_refreshed_kwargs(g1, et, s1),
        **kwargs,
    )


@citation_wrapper(_CITATION)
def doublegaussian_redback_refreshed(
    time,
    redshift,
    thv,
    loge0,
    thc,
    thj,
    g1,
    et,
    s1,
    logn0,
    p,
    logepse,
    logepsb,
    g0,
    xiN,
    **kwargs,
):
    return _native_redback_wrapper(
        time,
        redshift,
        thv,
        loge0,
        thc,
        thj,
        logn0,
        p,
        logepse,
        logepsb,
        g0,
        xiN,
        structure_kind="double_gaussian",
        structure_energy=kwargs.pop("ss", 0.1),
        structure_gamma=kwargs.pop("aa", 0.5),
        **_refreshed_kwargs(g1, et, s1),
        **kwargs,
    )
