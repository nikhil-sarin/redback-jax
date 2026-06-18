"""
JAX translation of redback's CutoffBlackbody SED.

Translates CutoffBlackbody._set_norm, _set_sed, and _SED.flux_density
from redback into pure JAX functions that are JIT-compilable and
differentiable.

Physics
-------
The CutoffBlackbody SED applies a UV suppression below λ_cutoff:

  λ ≥ λ_c :  sed ∝  r² / λ⁵          / expm1(hc / kTλ)   (Planck)
  λ < λ_c :  sed ∝  r² / (λ_c · λ⁴)  / expm1(hc / kTλ)   (suppressed)

A luminosity-conserving normalisation factor ensures that
  ∫ sed dλ = L_bol
at every epoch, regardless of how much flux is cut.

Reference
---------
Nicholl et al. 2017 ApJ 850 55
https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N/abstract
"""

import math as _math

import jax
import jax.numpy as jnp
from jax import jit

import astropy.constants as _cc
import astropy.units as _uu

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Physical constants — computed once via astropy, stored as Python floats
# ---------------------------------------------------------------------------

# h·c / k_B  [cm·K]  (redback X_CONST)
_X_CONST: float = float((_cc.h * _cc.c / _cc.k_B).cgs.value)

# 4π·2π·h·c²·Å_to_cm  [erg·cm³/s]  (redback FLUX_CONST)
_FLUX_CONST: float = float(
    (4.0 * _math.pi * 2.0 * _math.pi * _cc.h * _cc.c ** 2).cgs.value
    * float(_uu.Angstrom.cgs.scale)           # angstrom_cgs = 1e-8 cm/Å
)

# c  [cm/s]
_C_CM: float = float(_cc.c.cgs.value)

# 4π
_4PI: float = 4.0 * _math.pi

# nxcs = X_CONST * [1, 2, ..., 10]  — precomputed as a float64 JAX array
_NXCS = jnp.array([_X_CONST * n for n in range(1, 11)], dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Normalisation  (1:1 translation of redback CutoffBlackbody._set_norm)
# ---------------------------------------------------------------------------

@jit
def cutoff_blackbody_norm(
    luminosity,           # (N,) erg/s
    temperature,          # (N,) K
    r_photosphere,        # (N,) cm
    cutoff_wavelength_ang,   # scalar Å
    alpha_uv=1.0,            # scalar, power-law UV suppression index ∈ [0, 4)
):
    """Luminosity-conserving normalisation factor for the CutoffBlackbody SED.

    Translates redback ``CutoffBlackbody._set_norm`` with a generalised UV
    power-law index.  Setting ``alpha_uv=1.0`` (default) recovers the original
    redback formula exactly.

    Parameters
    ----------
    luminosity : (N,) erg/s
    temperature : (N,) K
    r_photosphere : (N,) cm
    cutoff_wavelength_ang : scalar Å  (e.g. 3000.0)
    alpha_uv : scalar, default 1.0
        Power-law index for UV suppression below λ_c.  The blue-side SED goes
        as Planck × (λ/λ_c)^alpha_uv.  Valid range: [0, 4).

    Returns
    -------
    norm : (N,)  dimensionless
    """
    from jax.scipy.special import gammaincc, gamma as _gamma_sp

    fp = luminosity.dtype
    lc = jnp.asarray(cutoff_wavelength_ang * 1e-8, dtype=fp)   # Å → cm

    # FLUX_CONST / angstrom_cgs  =  4π·2π·h·c²  (no Å factor)
    fc_no_ang = jnp.asarray(_FLUX_CONST / 1e-8, dtype=fp)

    # norm_init = L / (4π·2π·h·c² · r² · T)   (Eq. A3 in Nicholl+17)
    norm_init = luminosity / (fc_no_ang * r_photosphere ** 2 * temperature)

    # Broadcasting: nxcs (10,) vs T (N,1)
    nxcs  = jnp.asarray(_NXCS, dtype=fp)           # (10,)
    tp    = temperature[:, None]                    # (N, 1)
    z     = nxcs / (lc * tp)                       # (N, 10): nX/(T·λ_c)

    # Clamp to [0, 3.99]: Γ(4-α) is singular for α ≥ 4 (order ≤ 0).
    alpha = jnp.clip(jnp.asarray(alpha_uv, dtype=fp), 0.0, 3.99)
    order = jnp.asarray(4.0, dtype=fp) - alpha     # Γ(4-α, z) order

    # Generalised blue-side term: T^{3-α} / (nX^{4-α} · λ_c^α) · Γ(4-α, z)
    # Analytically: ∫₀^{λ_c} λ^{α-5}/λ_c^α exp(-nA/λ) dλ absorbed into norm_init.
    # For α=1 this reduces to the original closed-form term_1.
    # For α=0 the combined term_1+term_2 = 6T³/nX⁴ (pure Planck, lc-independent).
    upper_gamma = _gamma_sp(order) * gammaincc(order, z)   # Γ(4-α, z): (N, 10)
    term_1 = jnp.power(tp, jnp.asarray(3.0, dtype=fp) - alpha) * upper_gamma / (
        jnp.power(nxcs, jnp.asarray(4.0, dtype=fp) - alpha) * jnp.power(lc, alpha)
    )

    # Red-side term (λ ≥ λ_c, standard Planck) — unchanged for all α
    c1  = jnp.exp(jnp.clip(-z, jnp.asarray(-700.0, dtype=fp), jnp.asarray(0.0, dtype=fp)))
    tp2 = tp ** 2
    tp3 = tp ** 3
    term_2 = (
        (6.0 * tp3
         - c1 * (nxcs ** 3
                 + 3.0 * nxcs ** 2 * lc * tp
                 + 6.0 * (nxcs * lc ** 2 * tp2 + lc ** 3 * tp3))
         / lc ** 3)
        / nxcs ** 4
    )

    f_blue_reds = jnp.sum(term_1 + term_2, axis=1)   # (N,)
    norm = norm_init / jnp.maximum(
        f_blue_reds, jnp.asarray(1e-200, dtype=fp)
    )
    return norm


# ---------------------------------------------------------------------------
# Full flux density  (1:1 translation of _set_sed + _SED.flux_density)
# ---------------------------------------------------------------------------

@jit
def cutoff_blackbody_flux_density(
    frequency,            # (N,) Hz
    luminosity,           # (N,) erg/s
    temperature,          # (N,) K
    r_photosphere,        # (N,) cm
    luminosity_distance,  # scalar cm
    cutoff_wavelength_ang=3000.0,   # scalar Å
    alpha_uv=1.0,                   # scalar, UV power-law index ∈ [0, 4)
):
    """Observed flux density for the CutoffBlackbody SED.

    1:1 translation of redback ``CutoffBlackbody._set_sed`` +
    ``_SED.flux_density``, extended with a free UV power-law index.

    The SED below λ_c is suppressed by a factor (λ/λ_c)^alpha_uv relative to
    a standard Planck blackbody.  ``alpha_uv=1.0`` (default) matches the
    original redback CutoffBlackbody exactly.

    Parameters
    ----------
    frequency : (N,) Hz
        Source-frame frequency for each observation.
    luminosity : (N,) erg/s
        Bolometric luminosity at the observation time.
    temperature : (N,) K
        Photosphere temperature at the observation time.
    r_photosphere : (N,) cm
        Photosphere radius at the observation time.
    luminosity_distance : scalar cm
        Luminosity distance (pre-computed from redshift via astropy).
    cutoff_wavelength_ang : scalar Å, default 3000
        UV cutoff wavelength.
    alpha_uv : scalar, default 1.0
        Power-law UV suppression index.  Valid range [0, 4).

    Returns
    -------
    F_mjy : (N,) mJy
        Observed flux density.
    """
    fp = luminosity.dtype

    # --- Normalisation (depends only on L, T, r — NOT on frequency) ----------
    norm = cutoff_blackbody_norm(
        luminosity, temperature, r_photosphere, cutoff_wavelength_ang, alpha_uv
    )

    fc    = jnp.asarray(_FLUX_CONST,  dtype=fp)
    xc    = jnp.asarray(_X_CONST,     dtype=fp)
    c_cm  = jnp.asarray(_C_CM,        dtype=fp)
    lc    = jnp.asarray(cutoff_wavelength_ang * 1e-8, dtype=fp)   # Å → cm
    dl2   = jnp.asarray(luminosity_distance, dtype=fp) ** 2

    lam_cm = c_cm / frequency                                      # (N,) cm

    # Planck exponent, clipped to avoid overflow/underflow
    x = xc / (lam_cm * temperature)
    x = jnp.clip(
        x,
        jnp.asarray(1e-10, dtype=fp),
        jnp.asarray(500.0,  dtype=fp),
    )

    # --- SED (two branches on wavelength vs. cutoff) --------------------------
    # λ ≥ λ_c: standard Planck r²/λ⁵ / expm1(x)
    # λ < λ_c: suppressed by (λ/λ_c)^alpha_uv → r²/(λ_c^α · λ^(5-α)) / expm1(x)
    # Clamp to [0, 3.99] — same guard as in cutoff_blackbody_norm.
    alpha = jnp.clip(jnp.asarray(alpha_uv, dtype=fp), 0.0, 3.99)
    r2    = r_photosphere ** 2
    planck_r2 = jnp.where(
        lam_cm < lc,
        r2 / (jnp.power(lc, alpha) * jnp.power(lam_cm, jnp.asarray(5.0, dtype=fp) - alpha)),
        r2 / lam_cm ** 5,
    )
    sed = fc * planck_r2 / jnp.expm1(x) * norm   # erg/s/Å

    # --- Convert to flux density (mJy) ----------------------------------------
    # _SED.flux_density:
    #   F_ν = sed / (4π dl²) * λ_Å / ν_Hz    [erg/s/Hz/cm²]
    #   F_mJy = F_ν * 1e26
    lam_ang = lam_cm * jnp.asarray(1e8, dtype=fp)    # cm → Å
    F_nu    = sed * lam_ang / frequency / (jnp.asarray(_4PI, dtype=fp) * dl2)
    return F_nu * jnp.asarray(1e26, dtype=fp)         # → mJy
