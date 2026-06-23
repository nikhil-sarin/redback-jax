"""
JAX implementation of the tophat GRB afterglow model.

Numerically matches redback's ``tophat_redback`` (Lamb, Mandel & Resmi 2018)
to ~1 % at all frequencies and times, using the same RK4 / angular-integration
scheme but expressed entirely in JAX so the model can be JIT-compiled and
differentiated on CPU/GPU.

Public entry point
------------------
    tophat_redback(time, redshift, thv, loge0, thc, logn0, p,
                   logepse, logepsb, g0, xiN, frequency, dl,
                   *, res=100, steps=250, k=0, expansion=True, a1=1)

    Returns flux density in mJy, same shape as ``time``.

Usage notes
-----------
* ``res`` and ``steps`` are **static** (they set JAX array shapes).  Recompilation
  is triggered only when their values change.
* ``dl`` (luminosity distance in cm) must be supplied by the caller.
  Use ``wcosmo`` or ``astropy`` to compute it outside the JIT boundary.
* ``frequency`` must have the same length as ``time`` (one frequency per data point).
* For multi-band data, unique-frequency processing is handled automatically
  via ``jax.vmap`` over data points — no pre-sorting required.
"""

import math as _math
from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Physical constants (CGS) — identical values to redback's base_models.py
# ---------------------------------------------------------------------------
_MP    = 1.6726231e-24   # g  proton mass
_ME    = 9.1093897e-28   # g  electron mass
_CC    = 2.99792453e10   # cm/s
_QE    = 4.8032068e-10   # esu electron charge
_C2    = _CC * _CC
_SIGT  = (_QE * _QE / (_ME * _C2))**2 * (8.0 * _math.pi / 3.0)   # Thomson cross-section, cm²
_FOURPI = 4.0 * _math.pi
_DAY_TO_S = 86400.0
_MJY_PER_CGS = 1e26   # 1 erg/s/cm²/Hz → 1e26 mJy


# ---------------------------------------------------------------------------
# Step 0 — angular grid  (depends only on thj and res, both static once traced)
# ---------------------------------------------------------------------------

def _get_segments(thj, res):
    """Solid-angle grid identical to get_segments_numba in redback."""
    latstep  = thj / res
    rotstep  = 2.0 * _math.pi / res

    i_vals = jnp.arange(res, dtype=jnp.float64)
    Omi_per_lat = rotstep * (jnp.cos(i_vals * latstep) - jnp.cos((i_vals + 1) * latstep))
    Omi  = jnp.repeat(Omi_per_lat, res)    # (res², )  — res phi bins per lat ring

    thi  = jnp.linspace(latstep * 0.5, res * latstep - latstep * 0.5, res)
    phii = jnp.linspace(rotstep * 0.5, res * rotstep - rotstep * 0.5, res)
    return Omi, thi, phii, rotstep, latstep


# ---------------------------------------------------------------------------
# Step 1 — jet structure (tophat only)
# ---------------------------------------------------------------------------

def _get_structure_tophat(g0, e0, thi, thc, thj):
    """Energy / Lorentz-factor profile for a tophat jet."""
    thj_used = jnp.minimum(thc, thj)
    # Smooth step: erf(-(θ - θ_j)×1000) × 0.5 + 0.5  → 1 inside jet, 0 outside
    fac = jax.scipy.special.erf(-(thi - thj_used) * 1000.0) * 0.5 + 0.5
    Gs  = (g0  - 1.0) * fac + 1.0000000000001
    Ei  = e0  * fac
    return Gs, Ei    # both (n_thi,)


# ---------------------------------------------------------------------------
# Step 2 — observer angles
# ---------------------------------------------------------------------------

def _get_obsangle(phii, thi, tho):
    """Observer angle for each (thi, phii) pair."""
    # phi = 0 (rotational symmetry)
    cos_ang = jnp.clip(
        jnp.cos(tho) * jnp.cos(thi)[:, None]
        + jnp.sin(tho) * jnp.cos(phii)[None, :] * jnp.sin(thi)[:, None],
        -1.0, 1.0,
    )
    return jnp.arccos(cos_ang).reshape(-1)   # (n_thi * n_phi,)


# ---------------------------------------------------------------------------
# Step 3 — Lorentz-factor / swept-mass evolution (RK4 scan)
# ---------------------------------------------------------------------------

def _rk4_step(ghat, dm_log, G, M, fac):
    """Single Blandford-McKee RK4 step."""
    ghatm1   = ghat - 1.0
    dm_lin   = jnp.power(10.0, dm_log)
    G2       = G * G
    num  = fac * dm_lin * (ghat * (G2 - 1.0) - ghatm1 * (G - 1.0 / G))
    denom = M + dm_lin * (2.0 * ghat * G - ghatm1 * (1.0 + 1.0 / G2))
    return num / denom


def _get_gamma(G0, Eps, steps, n0, k):
    """Evolve Lorentz factor from swept-mass log-grid using RK4.

    Parameters
    ----------
    G0  : (n_thi,) initial Lorentz factors
    Eps : (n_thi,) initial kinetic energies (erg)
    steps : int  (static)
    n0  : float  ambient number density (cm⁻³ for k=0, A* for k=2)
    k   : float  density profile index (0 or 2)

    Returns
    -------
    state_G   : (n_thi, steps)  Lorentz factor at each mass step
    state_dm  : (n_thi, steps)  swept mass at each step (g)
    state_ghat: (n_thi, steps)  adiabatic index at each step
    """
    Rmin = 1e10
    Rmax = 1e24
    Nmin = (_FOURPI / (3.0 - k)) * n0 * Rmin ** (3.0 - k)
    Nmax = (_FOURPI / (3.0 - k)) * n0 * Rmax ** (3.0 - k)
    dlogm = jnp.log10(Nmin * _MP)
    h     = jnp.log10(Nmax / Nmin) / steps
    fac   = -h * jnp.log(10.0)              # negative → deceleration

    M = Eps / (G0 * _C2)   # (n_thi,)

    def _ghat_from_G(G):
        G2m1      = G * G - 1.0
        G2m1_root = jnp.sqrt(jnp.maximum(G2m1, 0.0))
        theta     = G2m1_root * (G2m1_root + 1.07 * G2m1) / (3.0 * (1.0 + G2m1_root + 1.07 * G2m1))
        z         = theta / (0.24 + theta)
        return (((((( 1.07136 * z
                    - 2.39332) * z
                    + 2.32513) * z
                    - 0.96583) * z
                    + 0.18203) * z
                    - 1.21937) * z
                    + 5.0) / 3.0

    def _scan_step(carry, _):
        G, dm = carry
        ghat = _ghat_from_G(G)

        F1 = _rk4_step(ghat, dm,             G,             M, fac)
        F2 = _rk4_step(ghat, dm + 0.5 * h,   G + 0.5 * F1,  M, fac)
        F3 = _rk4_step(ghat, dm + 0.5 * h,   G + 0.5 * F2,  M, fac)
        F4 = _rk4_step(ghat, dm + h,          G + F3,        M, fac)

        dG   = (F1 + 2.0 * (F2 + F3) + F4) / 6.0
        G_new  = G + dG + 1e-15
        dm_new = dm + h
        return (G_new, dm_new), (G, dm, ghat)

    G_init  = G0.astype(jnp.float64)
    dm_init = jnp.full_like(G0, dlogm, dtype=jnp.float64)
    _, (state_G, state_dm_log, state_ghat) = jax.lax.scan(_scan_step, (G_init, dm_init), None, length=steps)

    # scan outputs have leading axis = steps; transpose to (n_thi, steps)
    return state_G.T, (10.0 ** state_dm_log).T, state_ghat.T


# ---------------------------------------------------------------------------
# Step 4 — synchrotron parameters for one angular element (vectorised over steps)
# ---------------------------------------------------------------------------

def _calc_step1(G, dm, p, xp, Fx, epsb, epse, n, k, thi, ghat, rotstep, latstep, xiN, expansion, a1, res):
    """Synchrotron parameters for a single theta-bin.

    Inputs are all 1-D arrays over the ``steps`` dimension.
    """
    G2   = G * G
    Gm1  = G - 1.0
    beta = jnp.sqrt(1.0 - 1.0 / G2)
    Ne   = dm / _MP

    # Sound-speed expansion angle
    cs   = jnp.sqrt(_C2 * ghat * (ghat - 1.0) * Gm1 / (1.0 + ghat * Gm1))
    te   = jnp.arcsin(jnp.clip(cs / (_CC * jnp.sqrt(jnp.maximum(G2 - 1.0, 1e-30))), -1.0, 1.0))

    fac = 0.5 * latstep
    if expansion:
        ex   = te / G ** (a1 + 1.0)
        OmG  = rotstep * (jnp.cos(thi - fac) - jnp.cos(ex / res + thi + fac))
    else:
        ex   = jnp.ones_like(G)
        OmG  = jnp.full_like(G, rotstep * (jnp.cos(thi - fac) - jnp.cos(thi + fac)))

    # Effective radius
    exponent_num  = 1.0 - jnp.cos(latstep + ex[0])
    exponent_denom = 1.0 - jnp.cos(latstep + ex)
    exponent = jnp.where(
        jnp.arange(len(G)) > 0,
        jnp.sqrt(jnp.maximum(exponent_num / jnp.maximum(exponent_denom, 1e-30), 0.0)),
        1.0,
    )

    R_orig = ((3.0 - k) * Ne / (_FOURPI * n)) ** (1.0 / (3.0 - k))
    dR_raw = jnp.concatenate([R_orig[:1], jnp.diff(R_orig)])
    R      = jnp.cumsum(dR_raw * exponent ** (1.0 / (3.0 - k)))

    n0 = n * R ** (-k)

    B   = jnp.sqrt(2.0 * _FOURPI * epsb * n0 * _MP * _C2 *
                   ((ghat * G + 1.0) / (ghat - 1.0)) * Gm1)
    gmm = jnp.sqrt(1.5 * _FOURPI * _QE / (_SIGT * B))

    # Minimum electron Lorentz factor (three cases for p > 2, = 2, < 2)
    base_gm  = (epse / xiN) * Gm1 * (_MP / _ME)
    gm_gt2   = (p - 2.0) / (p - 1.0) * base_gm
    log_rat  = jnp.log(jnp.maximum(gmm / jnp.maximum(base_gm, 1e-300), 1e-300))
    log_rat_safe = jnp.where(log_rat > 0.0, log_rat, jnp.ones_like(log_rat))
    gm_eq2   = jnp.where(log_rat > 0.0, base_gm / log_rat_safe, base_gm)
    # gm_lt2 base is negative when p > 2 (the "not taken" branch) — guard it
    # so the fractional exponent doesn't produce NaN in autodiff.
    inner_lt2 = (((2.0 - p) / (p - 1.0)) * (_MP / _ME) * (epse / xiN) * Gm1 *
                 gmm ** (p - 2.0))
    inner_lt2_safe = jnp.where(p < 2.0, inner_lt2, jnp.ones_like(inner_lt2))
    gm_lt2   = inner_lt2_safe ** (1.0 / (p - 1.0))
    gm       = jnp.where(p > 2.0, gm_gt2, jnp.where(p >= 2.0, gm_eq2, gm_lt2))

    nump = 3.0 * xp * gm * gm * _QE * B / (_FOURPI * _ME * _CC)
    Pp   = xiN * Fx * _ME * _C2 * _SIGT * B / (3.0 * _QE)
    KT   = gm * _ME * _C2

    return beta, Ne, OmG, R, B, gm, nump, Pp, KT


# ---------------------------------------------------------------------------
# Step 5 — emission properties for one (thi, phii) segment
# ---------------------------------------------------------------------------

def _calc_step2(Dl, Om0, rotstep, latstep, Obsa, beta, Ne, OmG, R, B, gm, nump, Pp, KT, G, expansion):
    """Observed emission properties for a single angular segment."""
    Dl2       = Dl * Dl
    NO        = Om0 * Ne / _FOURPI
    cos_Obsa  = jnp.cos(Obsa)

    if expansion:
        Om = jnp.maximum(Om0, OmG)
    else:
        Om = OmG
    thii = jnp.arccos(jnp.clip(1.0 - Om / rotstep, -1.0, 1.0))

    R_diff = jnp.concatenate([R[:1], jnp.diff(R)])
    dt     = R_diff * (1.0 / beta - cos_Obsa) / _CC
    dto    = R_diff * (1.0 / beta - 1.0)      / _CC
    tobs   = jnp.cumsum(dt)
    tobso  = jnp.cumsum(dto)

    dop  = 1.0 / (G * (1.0 - beta * cos_Obsa))
    gc   = 6.0 * _math.pi * _ME * _CC / (G * _SIGT * B * B * jnp.maximum(tobso, 1e-300))
    nucp = 0.286 * 3.0 * gc * gc * _QE * B / (_FOURPI * _ME * _CC)

    num  = dop * nump
    nuc  = dop * nucp
    Fmax = NO * Pp * dop**3 / (_FOURPI * Dl2)
    FBB  = 2.0 * Om * jnp.cos(thii) * dop * KT * R * R / (_C2 * Dl2)

    return FBB, Fmax, nuc, num, tobs


# ---------------------------------------------------------------------------
# Step 6 — synchrotron spectrum at a single frequency
# ---------------------------------------------------------------------------

def _get_ag(FBB, nuc, num, nu1, Fmax, p):
    """Synchrotron flux at frequency nu1 for all ``steps`` mass-steps.

    Reproduces the overlapping-if logic of get_ag_numba exactly.
    """
    fast = nuc < num

    # Guard denominators so power-law gradients are finite in the "not taken"
    # branch — jnp.where evaluates both sides during autodiff.
    nuc_f = jnp.where(fast,  nuc, jnp.ones_like(nuc))   # safe when fast=True
    num_f = jnp.where(fast,  num, jnp.ones_like(num))   # safe when fast=True
    nuc_s = jnp.where(~fast, nuc, jnp.ones_like(nuc))   # safe when fast=False
    num_s = jnp.where(~fast, num, jnp.ones_like(num))   # safe when fast=False

    # Fast-cooling (nuc < num)
    f1 = Fmax * (nu1 / nuc_f) ** (1.0 / 3.0)
    f2 = Fmax * (nu1 / nuc_f) ** (-0.5)
    f3 = Fmax * (num_f / nuc_f) ** (-0.5) * (nu1 / num_f) ** (-p / 2.0)
    # Slow-cooling (num < nuc)
    f4 = Fmax * (nu1 / num_s) ** (1.0 / 3.0)
    f5 = Fmax * (nu1 / num_s) ** (-(p - 1.0) / 2.0)
    f6 = Fmax * (nuc_s / num_s) ** (-(p - 1.0) / 2.0) * (nu1 / nuc_s) ** (-p / 2.0)

    # Overlapping conditions: later assignments overwrite earlier ones
    fluxt = jnp.zeros_like(num)
    fluxt = jnp.where(fast  & (nu1 <  nuc),  f1, fluxt)
    fluxt = jnp.where(fast  & (nu1 >  nuc),  f2, fluxt)
    fluxt = jnp.where(fast  & (nu1 >  num),  f3, fluxt)
    fluxt = jnp.where(~fast & (nu1 <  num),  f4, fluxt)
    fluxt = jnp.where(~fast & (nu1 >  num),  f5, fluxt)
    fluxt = jnp.where(~fast & (nu1 >  nuc),  f6, fluxt)

    # Self-absorption cap
    FBB_adj = FBB * nu1**2 * jnp.maximum(1.0, (nu1 / jnp.maximum(num, 1e-300))**0.5)
    return jnp.minimum(fluxt, FBB_adj)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=('res', 'steps', 'k', 'expansion', 'a1'))
def tophat_redback(
    time,
    redshift,
    thv,
    loge0,
    thc,
    logn0,
    p,
    logepse,
    logepsb,
    g0,
    xiN,
    frequency,
    dl,
    *,
    res=100,
    steps=250,
    k=0,
    expansion=True,
    a1=1,
):
    """JAX tophat GRB afterglow model.

    Numerically equivalent to ``redback.transient_models.afterglow_models.tophat_redback``.

    Parameters
    ----------
    time : (n,) float64
        Observer-frame times in days.
    redshift : float
        Source redshift.
    thv : float
        Viewing angle in radians.
    loge0 : float
        log10 jet kinetic energy (erg).
    thc : float
        Jet half-opening angle = jet edge angle for tophat (radians).
    logn0 : float
        log10 ISM number density (cm⁻³ for k=0) or log10 A* (for k=2).
    p : float
        Electron power-law index.
    logepse : float
        log10 electron energy fraction.
    logepsb : float
        log10 magnetic energy fraction.
    g0 : float
        Initial bulk Lorentz factor.
    xiN : float
        Fraction of electrons accelerated (typically 1).
    frequency : (n,) float64
        Observer-frame frequencies in Hz (one per data point).
    dl : float
        Luminosity distance in cm.
    res : int (static, default 100)
        Angular resolution (number of latitude / azimuth bins).
    steps : int (static, default 250)
        Number of RK4 mass steps in the Lorentz-factor evolution.
    k : int (static, default 0)
        ISM density profile: 0 = constant, 2 = wind.
    expansion : bool (static, default True)
        Include lateral sound-speed expansion.
    a1 : float (static, default 1)
        Expansion description: 0 = sound speed, 1 = Granot & Piran 2012.

    Returns
    -------
    flux_density : (n,) float64
        Flux density in mJy.
    """
    # --- unpack parameters ---
    nism  = 10.0 ** logn0
    e0    = 10.0 ** loge0
    epse  = 10.0 ** logepse
    epsb  = 10.0 ** logepsb
    thj   = thc           # tophat: jet edge == core angle

    # --- Wijers & Galama 1999 spectral tables ---
    pxf = jnp.array([
        [1.0, 3.0, 0.41], [1.2, 1.4, 0.44], [1.4, 1.1, 0.48],
        [1.6, 0.86, 0.53], [1.8, 0.725, 0.56], [2.0, 0.637, 0.59],
        [2.2, 0.579, 0.612], [2.5, 0.520, 0.630], [2.7, 0.487, 0.641],
        [3.0, 0.451, 0.659], [3.2, 0.434, 0.660], [3.4, 0.420, 0.675],
    ], dtype=jnp.float64)
    xp = jnp.interp(p, pxf[:, 0], pxf[:, 1])
    Fx = jnp.interp(p, pxf[:, 0], pxf[:, 2])

    # --- angular grid ---
    Omi, thi, phii, rotstep, latstep = _get_segments(thj, res)

    # --- jet structure ---
    Gs, Ei = _get_structure_tophat(g0, e0, thi, thc, thj)

    # --- Lorentz factor evolution ---
    G, SM, ghat = _get_gamma(Gs, Ei, steps, nism, k)
    # G, SM, ghat: (n_thi, steps)

    # --- observer angles for every segment ---
    Obsa = _get_obsangle(phii, thi, thv)   # (n_thi * n_phi,)

    n_phi = res
    n_segs = res * res

    # --- step 1: synchrotron parameters for every latitude bin ---
    def _step1_one(args):
        G_i, SM_i, thi_i, ghat_i = args
        return _calc_step1(
            G_i, SM_i, p, xp, Fx, epsb, epse, nism, k,
            thi_i, ghat_i, rotstep, latstep, xiN, expansion, a1, res,
        )

    beta_all, Ne_all, OmG_all, R_all, B_all, gm_all, nump_all, Pp_all, KT_all = jax.vmap(
        _step1_one
    )((G, SM, thi, ghat))
    # All (n_thi, steps)

    # --- step 2: emission properties for every (thi, phii) segment ---
    # Map each flat segment index kk → i = kk // n_phi
    kk_arr = jnp.arange(n_segs)
    i_arr  = kk_arr // n_phi

    def _step2_one(args):
        (beta_k, Ne_k, OmG_k, R_k, B_k, gm_k, nump_k, Pp_k, KT_k,
         G_k, Om0_k, Obsa_k) = args
        return _calc_step2(
            dl, Om0_k, rotstep, latstep, Obsa_k,
            beta_k, Ne_k, OmG_k, R_k, B_k, gm_k, nump_k, Pp_k, KT_k,
            G_k, expansion,
        )

    FBB_all, Fmax_all, nuc_all, num_all, tobs_all = jax.vmap(_step2_one)((
        beta_all[i_arr], Ne_all[i_arr], OmG_all[i_arr],
        R_all[i_arr],    B_all[i_arr],  gm_all[i_arr],
        nump_all[i_arr], Pp_all[i_arr], KT_all[i_arr],
        G[i_arr],
        Omi,
        Obsa,
    ))
    # All (n_segs, steps), tobs_all (n_segs, steps)

    # --- step 3: build lightcurve for each data point ---
    # Source-frame time and frequency for each data point
    t_src  = time / (1.0 + redshift)           # (n,) days
    nu_src = frequency * (1.0 + redshift)       # (n,) Hz

    def _lc_one_datapoint(t_i, nu_i):
        """Evaluate the afterglow lightcurve at a single (t, ν) data point."""
        def _contrib_one_seg(FBB_k, Fmax_k, nuc_k, num_k, tobs_k):
            flux_k = _get_ag(FBB_k, nuc_k, num_k, nu_i, Fmax_k, p)  # (steps,)
            return jnp.interp(t_i * _DAY_TO_S, tobs_k, flux_k)

        contributions = jax.vmap(_contrib_one_seg)(
            FBB_all, Fmax_all, nuc_all, num_all, tobs_all
        )   # (n_segs,)
        return jnp.sum(contributions) * (1.0 + redshift)

    LC = jax.vmap(_lc_one_datapoint)(t_src, nu_src)   # (n,)

    # Convert from redback internal units (Jy-like) to mJy
    return LC / 1e-26
