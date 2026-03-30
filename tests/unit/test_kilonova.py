"""
Tests for redback_jax.models.kilonova module.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from redback_jax.models.kilonova import (
    metzger_kilonova_bolometric,
    magnetar_boosted_kilonova_bolometric,
)


class TestMetzgerKilonova:
    """Tests for metzger_kilonova_bolometric."""

    def test_returns_log10_lbol(self):
        """Output is log10(L) in physical range, no nan/inf."""
        time = jnp.linspace(0.5, 20.0, 50, dtype=jnp.float32)
        log10_lbol = metzger_kilonova_bolometric(time, mej=0.05, vej=0.2,
                                                  beta=3.0, kappa=1.0)
        assert log10_lbol.shape == time.shape
        assert not jnp.any(jnp.isnan(log10_lbol))
        assert not jnp.any(jnp.isinf(log10_lbol))
        # Physical range: ~39-43 erg/s for typical kilonova parameters
        assert jnp.all(log10_lbol > 35.0)
        assert jnp.all(log10_lbol < 50.0)

    def test_decreases_at_late_times(self):
        """Kilonova fades after peak — late-time log10_lbol < early-time."""
        time = jnp.linspace(0.5, 30.0, 80, dtype=jnp.float32)
        log10_lbol = metzger_kilonova_bolometric(time, mej=0.05, vej=0.2,
                                                  beta=3.0, kappa=1.0)
        # Overall trend: later times are fainter
        assert float(log10_lbol[-1]) < float(log10_lbol[10])

    def test_mass_scaling(self):
        """Higher ejecta mass changes the light curve."""
        time = jnp.linspace(1.0, 20.0, 30, dtype=jnp.float32)
        lo = metzger_kilonova_bolometric(time, mej=0.01, vej=0.2, beta=3.0, kappa=1.0)
        hi = metzger_kilonova_bolometric(time, mej=0.1,  vej=0.2, beta=3.0, kappa=1.0)
        assert not jnp.allclose(lo, hi, rtol=1e-3)

    def test_opacity_scaling(self):
        """Higher opacity slows diffusion and changes light-curve shape.

        With neutron_precursor=True the effective opacity is dominated by the
        neutron-fraction weighting and the ``kappa`` parameter has negligible
        impact (the Xr fraction is ~1e-6).  Test with neutron_precursor=False
        where kappa enters the diffusion timescale directly.
        """
        time = jnp.linspace(1.0, 20.0, 30, dtype=jnp.float32)
        lo_kap = metzger_kilonova_bolometric(time, mej=0.05, vej=0.2, beta=3.0,
                                              kappa=0.5, neutron_precursor=False)
        hi_kap = metzger_kilonova_bolometric(time, mej=0.05, vej=0.2, beta=3.0,
                                              kappa=10.0, neutron_precursor=False)
        assert not jnp.allclose(lo_kap, hi_kap, atol=0.05)

    def test_float32_safe(self):
        """No overflow when run in float32 mode."""
        import jax
        time = jnp.linspace(0.5, 20.0, 50, dtype=jnp.float32)
        log10_lbol = metzger_kilonova_bolometric(time, mej=0.05, vej=0.2,
                                                  beta=3.0, kappa=1.0)
        # dtype is float32 when x64 is disabled, float64 when enabled — both are fine
        if not jax.config.x64_enabled:
            assert log10_lbol.dtype == jnp.float32
        assert not jnp.any(jnp.isnan(log10_lbol))
        assert not jnp.any(jnp.isinf(log10_lbol))


class TestMagnetarKilonova:
    """Tests for magnetar_boosted_kilonova_bolometric."""

    def test_returns_log10_lbol(self):
        """Output is log10(L) in physical range, no nan/inf."""
        time = jnp.linspace(0.5, 20.0, 50, dtype=jnp.float32)
        log10_lbol = magnetar_boosted_kilonova_bolometric(
            time, mej=0.05, vej=0.2, beta=3.0, kappa=1.0,
            p0=1.0, bp=1.0, mass_ns=1.4, theta_pb=0.0,
        )
        assert log10_lbol.shape == time.shape
        assert not jnp.any(jnp.isnan(log10_lbol))
        assert not jnp.any(jnp.isinf(log10_lbol))
        assert jnp.all(log10_lbol > 35.0)
        assert jnp.all(log10_lbol < 50.0)

    def test_magnetar_brighter_than_plain(self):
        """Magnetar injection raises luminosity above r-process only."""
        # Use a strong magnetar (low p0, high bp) so the energy injection
        # significantly exceeds r-process heating over the light curve.
        time = jnp.linspace(0.5, 20.0, 50, dtype=jnp.float32)
        plain = metzger_kilonova_bolometric(time, mej=0.05, vej=0.2,
                                             beta=3.0, kappa=1.0)
        boosted = magnetar_boosted_kilonova_bolometric(
            time, mej=0.05, vej=0.2, beta=3.0, kappa=1.0,
            p0=0.7, bp=10.0, mass_ns=1.4, theta_pb=0.0,
        )
        # The magnetar-boosted model should be brighter on average
        assert float(jnp.mean(boosted)) > float(jnp.mean(plain))

    def test_spin_period_scaling(self):
        """Faster spin (lower p0) → more energy injection → higher mean luminosity."""
        # Use later times to let the magnetar spin-down difference accumulate.
        time = jnp.linspace(2.0, 20.0, 30, dtype=jnp.float32)
        fast = magnetar_boosted_kilonova_bolometric(
            time, mej=0.05, vej=0.2, beta=3.0, kappa=1.0,
            p0=0.5, bp=1.0, mass_ns=1.4, theta_pb=0.0,
        )
        slow = magnetar_boosted_kilonova_bolometric(
            time, mej=0.05, vej=0.2, beta=3.0, kappa=1.0,
            p0=10.0, bp=1.0, mass_ns=1.4, theta_pb=0.0,
        )
        assert float(jnp.mean(fast)) > float(jnp.mean(slow))

    def test_float32_safe(self):
        """No overflow when run in float32 mode."""
        import jax
        time = jnp.linspace(0.5, 20.0, 50, dtype=jnp.float32)
        log10_lbol = magnetar_boosted_kilonova_bolometric(
            time, mej=0.05, vej=0.2, beta=3.0, kappa=1.0,
            p0=1.0, bp=1.0, mass_ns=1.4, theta_pb=0.0,
        )
        if not jax.config.x64_enabled:
            assert log10_lbol.dtype == jnp.float32
        assert not jnp.any(jnp.isnan(log10_lbol))
        assert not jnp.any(jnp.isinf(log10_lbol))
