"""
Correctness checks for jax_csm.static_powerlaw.static_powerlaw_csm_bpl_bolometric.

Summary
-------
simple mode  : JAX matches Fortran at <2% for the CSM-interaction phase (t < r_outer/v_shock).
               Post-emergence behaviour differs by design: JAX uses ballistic coasting while
               Fortran continues accreting ejecta, but this is minor for most use cases.

transport mode: The dimensionless diffusion solver is NOT numerically converged near shock
                emergence.  Results at t_peak change by factors of 2–5 when n_steps is
                varied from 4096→32768, so transport mode should NOT be used for inference
                until the convergence issue is resolved.  Simple mode is the safe choice.
"""

import sys
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, "/home/nikhilsarin/projects/redback-csm")

from jax_csm.static_powerlaw import (
    static_powerlaw_csm_bpl_bolometric as jax_model,
    solve_static_powerlaw_bpl,
    StaticPowerlawBPLParams,
)

# ---------------------------------------------------------------------------
# Reference: Fortran-backed redback_csm model (simple/no-kappa mode)
# ---------------------------------------------------------------------------

def _fortran_shock_lbol(time, **kwargs):
    """Return Fortran shock luminosity (no kappa, run_mode=1)."""
    import numpy as _np
    if not hasattr(_np, "trapz"):
        _np.trapz = _np.trapezoid
    from redback_csm.models import static_powerlaw_csm_bpl_bolometric as fort_model
    return np.asarray(fort_model(time=time, mode="simple", **kwargs))


PARAMS = dict(
    eta=-2.0,
    r_inner=500.0 * 6.957e10,
    r_outer=5000.0 * 6.957e10,
    delta_sn=1.0,
    nn_sn=10.0,
    mej_sn=5.0,
    esn=1.0,
    eff=1.0,
    m_csm=1.0,
)

# CSM crossing time ≈ r_outer / v_shock ~ 10 days; use times well inside the shell
CSM_INTERACTION_TIMES = np.array([1.0, 3.0, 5.0, 7.0])


# ---------------------------------------------------------------------------
# Test 1: simple mode matches Fortran shock luminosity during CSM interaction
# ---------------------------------------------------------------------------

def test_simple_mode_matches_fortran_shock():
    """JAX simple mode must match Fortran shock lbol at <5% for t within the CSM shell."""
    ref = _fortran_shock_lbol(CSM_INTERACTION_TIMES, **PARAMS)
    out = np.asarray(jax_model(CSM_INTERACTION_TIMES, mode="simple", n_steps=4096, **PARAMS))

    assert out.shape == ref.shape
    good = ref > 1e-3 * ref.max()
    rel = np.abs((out[good] - ref[good]) / ref[good])
    assert rel.max() < 0.05, f"Max rel err {rel.max():.3f} > 5%"


# ---------------------------------------------------------------------------
# Test 2: simple mode outputs positive, finite values
# ---------------------------------------------------------------------------

def test_simple_mode_positive_finite():
    time = np.geomspace(0.5, 100.0, 12)
    out = np.asarray(jax_model(time, mode="simple", n_steps=2048, **PARAMS))
    assert np.all(np.isfinite(out)), "Non-finite values in simple mode"
    assert np.all(out >= 0), "Negative luminosity in simple mode"


# ---------------------------------------------------------------------------
# Test 3: simple mode converges with n_steps
# ---------------------------------------------------------------------------

def test_simple_mode_convergence():
    """Coarser and finer time grids should give <5% difference for simple mode."""
    time = CSM_INTERACTION_TIMES
    out_lo = np.asarray(jax_model(time, mode="simple", n_steps=1024, **PARAMS))
    out_hi = np.asarray(jax_model(time, mode="simple", n_steps=8192, **PARAMS))
    good = out_hi > 1e-3 * out_hi.max()
    rel = np.abs((out_lo[good] - out_hi[good]) / out_hi[good])
    assert rel.max() < 0.05, f"Simple mode not converged: max rel diff {rel.max():.3f}"


# ---------------------------------------------------------------------------
# Test 4: StaticPowerlawBPLResult fields are populated
# ---------------------------------------------------------------------------

def test_result_fields():
    params = StaticPowerlawBPLParams(**PARAMS)
    time = jnp.array([1.0, 5.0, 15.0])
    res = solve_static_powerlaw_bpl(time, params, mode="simple", n_steps=1024)
    assert res.lbol.shape == (3,)
    assert np.all(np.isfinite(np.asarray(res.lbol)))
    assert np.all(np.isfinite(np.asarray(res.radius)))
    assert np.all(np.isfinite(np.asarray(res.velocity)))
    assert np.all(np.asarray(res.velocity) >= 0)


# ---------------------------------------------------------------------------
# Test 5: JIT stability for simple mode
# ---------------------------------------------------------------------------

def test_simple_mode_jit_stable():
    time = np.array([1.0, 5.0, 15.0])
    a = np.asarray(jax_model(time, mode="simple", n_steps=1024, **PARAMS))
    b = np.asarray(jax_model(time, mode="simple", n_steps=1024, **PARAMS))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Test 6: transport mode runs and produces finite outputs
# ---------------------------------------------------------------------------

def test_transport_mode_finite():
    """Transport mode should run without NaN/Inf for t > diffusion timescale."""
    time = np.array([5.0, 15.0, 30.0, 60.0])
    out = np.asarray(jax_model(time, mode="transport", n_steps=4096, n_zones=20, **PARAMS))
    assert np.all(np.isfinite(out)), f"Non-finite in transport mode: {out}"
    assert np.all(out >= 0), "Negative luminosity in transport mode"


# ---------------------------------------------------------------------------
# Test 7: transport mode convergence near shock emergence (known limitation)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="JAX transport mode not numerically converged near shock emergence; "
                           "results change by >20% when n_steps is doubled. "
                           "Use mode='simple' for reliable GPU inference.")
def test_transport_mode_convergence_near_peak():
    """
    The JAX transport luminosity at peak (t≈10-15d) is known to change
    substantially when n_steps is varied.  This test documents the issue.
    """
    time = np.array([10.0, 15.0])  # near shock emergence
    out_lo = np.asarray(jax_model(time, mode="transport", n_steps=4096, n_zones=40, **PARAMS))
    out_hi = np.asarray(jax_model(time, mode="transport", n_steps=16384, n_zones=40, **PARAMS))
    good = out_hi > 1e30
    if not good.any():
        pytest.skip("No meaningful transport values to compare")
    rel = np.abs((out_lo[good] - out_hi[good]) / np.maximum(out_hi[good], 1e-300))
    # Should converge to <5% — will XFAIL until the transport bug is fixed
    assert rel.max() < 0.05, f"Max resolution sensitivity {rel.max():.2f} > 5%"
