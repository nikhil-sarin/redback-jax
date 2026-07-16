"""
Tests for redback_jax.inference module.
"""
import pytest

from redback_jax import inference


def test_inference_module_import():
    """Test that inference module can be imported."""
    assert inference is not None


class TestInferenceModule:
    """Test class for inference module functionality."""

    def test_module_exists(self):
        """Test that the inference module exists."""
        assert inference is not None


class TestNSResult:
    """Test cases for the NSResult container."""

    def _make_result(self):
        import jax.numpy as jnp

        from redback_jax.inference.nested_sampler import NSResult

        samples = {
            "t0": jnp.array([1.0, 2.0, 3.0]),
            "amp": jnp.array([0.4, 0.5, 0.6]),
        }
        return NSResult(
            logZ=-3.14,
            samples=samples,
            dead=None,
            log_weights=jnp.array([0.0, -1.0, -2.0]),
            param_names=["t0", "amp"],
        )

    def test_summary_and_repr(self):
        """NSResult renders a weighted-stats summary table and a repr."""
        res = self._make_result()
        res.summary()  # exercises the per-parameter weighted-stats loop
        text = repr(res)
        assert "NSResult" in text
        assert "n_samples=3" in text


class TestNestedSamplerGuard:
    """The NS fork guard: without handley-lab blackjax, init must fail clearly."""

    def test_requires_blackjax_fork(self):
        from redback_jax.inference import nested_sampler as ns

        if ns.HAS_BLACKJAX:
            pytest.skip("blackjax NS fork installed; guard not exercised")
        # The import failure is captured so the message can name the cause.
        assert ns._BLACKJAX_NS_IMPORT_ERROR is not None
        with pytest.raises(ImportError, match="nested-sampling API unavailable"):
            ns.NestedSampler(likelihood=None, prior=None)
