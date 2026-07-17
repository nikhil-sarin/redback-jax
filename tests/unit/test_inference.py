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


class TestAdaptiveTemperedSMC:
    """run_nested_sampling must actually anneal to the posterior.

    The truth sits far from the prior midpoint, so a sampler that returned
    prior draws (rather than posterior samples) would land on the prior
    median with the prior's width and fail every assertion here.
    """

    TRUTH = {"x": 4.0, "y": 3.0}
    SIGMA = 0.5
    BOUNDS = {"x": (-10.0, 10.0), "y": (0.0, 20.0)}

    def _run(self):
        import jax

        from redback_jax.inference.sampler import HAS_BLACKJAX, run_nested_sampling

        if not HAS_BLACKJAX:
            pytest.skip("blackjax not installed")

        def loglike(p):
            return -0.5 * (
                ((p["x"] - self.TRUTH["x"]) / self.SIGMA) ** 2
                + ((p["y"] - self.TRUTH["y"]) / self.SIGMA) ** 2
            )

        return run_nested_sampling(
            loglike,
            self.BOUNDS,
            n_particles=500,
            num_mcmc_steps=10,
            rng_key=jax.random.PRNGKey(0),
            verbose=False,
        )

    def test_recovers_truth_not_prior(self):
        """Posterior medians track the truth, not the prior midpoint."""
        import numpy as np

        result = self._run()
        for name, truth in self.TRUTH.items():
            samples = np.asarray(result.samples[name])
            lo, hi = self.BOUNDS[name]
            prior_median = (lo + hi) / 2
            median = float(np.median(samples))
            assert abs(median - truth) < 0.25, f"{name}: {median} != {truth}"
            # Guards the original bug: prior draws would sit at prior_median.
            assert abs(median - truth) < abs(median - prior_median)

    def test_posterior_width_matches_likelihood(self):
        """Spread is the likelihood's, not the prior's (uniform std ~5.8)."""
        import numpy as np

        result = self._run()
        for name in self.TRUTH:
            std = float(np.std(np.asarray(result.samples[name])))
            assert abs(std - self.SIGMA) < 0.25, f"{name}: std {std}"

    def test_log_evidence_matches_analytic(self):
        """Uniform prior x Gaussian likelihood has a closed-form evidence."""
        import numpy as np

        result = self._run()
        volume = np.prod([hi - lo for lo, hi in self.BOUNDS.values()])
        expected = np.log(
            (2 * np.pi * self.SIGMA**2) ** (len(self.BOUNDS) / 2) / volume
        )
        assert abs(result.log_evidence - expected) < 0.5
        # The old estimator (std of the log-likelihoods) ran to ~1e5 here.
        assert 0.0 < result.log_evidence_error < 1.0

    def test_anneals_to_posterior(self):
        """SMC reaches temperature 1 and reports its method."""
        result = self._run()
        assert result.metadata["method"] == "adaptive_tempered_smc"
        assert result.metadata["final_temperature"] == pytest.approx(1.0)
        assert result.metadata["n_tempering_steps"] >= 1


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
