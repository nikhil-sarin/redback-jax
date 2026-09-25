import jax.numpy as jnp
import numpy as np
import pytest

from redback_jax.inference import BatchedDataset, Likelihood, Prior, Uniform
from redback_jax.transient import Transient
from redback_jax.utils import luminosity_distance_cm


def test_batched_dataset_pads_observations_and_bands():
    t1 = Transient(
        time=np.array([1.0, 2.0, 3.0]),
        y=np.array([20.0, 20.5, 21.0]),
        y_err=np.array([0.1, 0.2, 0.3]),
        bands=["bessellb", "bessellv", "bessellb"],
        data_mode="magnitude",
        name="one",
    )
    t2 = Transient(
        time=np.array([4.0]),
        y=np.array([19.0]),
        y_err=np.array([0.4]),
        bands=["bessellr"],
        data_mode="magnitude",
        name="two",
    )

    batch = BatchedDataset([t1, t2])

    assert batch.shape == (2, 3)
    assert batch.bands == ("bessellb", "bessellv", "bessellr")
    np.testing.assert_array_equal(np.asarray(batch.mask), [[True, True, True], [True, False, False]])
    assert np.isinf(np.asarray(batch.obs_errs)[1, 1])
    assert int(batch.obs_band_idx[1, 0]) == batch.band_to_idx["bessellr"]


def test_batched_photometric_log_likelihood_matches_single_likelihoods():
    pytest.importorskip("jax_supernovae")
    from redback_jax.inference import make_batched_log_likelihood
    from redback_jax.models import arnett_spectra
    from redback_jax.sources import PrecomputedSpectraSource

    redshift = 0.01
    lum_dist = luminosity_distance_cm(redshift)
    fixed = {
        "temperature_floor": 5000.0,
        "kappa": 0.07,
        "kappa_gamma": 0.1,
    }
    fixed_single = {"redshift": redshift, "lum_dist": lum_dist, **fixed}
    true_params = {"t0": 58600.0, "f_nickel": 0.1, "mej": 1.4, "vej": 5000.0}
    prior = Prior([
        Uniform(58580, 58620, name="t0"),
        Uniform(0.05, 0.20, name="f_nickel"),
        Uniform(0.8, 2.0, name="mej"),
        Uniform(3000, 8000, name="vej"),
    ])

    out = arnett_spectra(**fixed_single, **{k: v for k, v in true_params.items() if k != "t0"})
    source = PrecomputedSpectraSource(
        phases=out.time,
        wavelengths=out.lambdas,
        flux_grid=out.spectra,
    )

    def make_transient(name, rel_times, bands):
        times = np.asarray(rel_times) + true_params["t0"]
        mags = [
            float(source.bandmag({"amplitude": 1.0}, band, rel_time))
            for rel_time, band in zip(rel_times, bands)
        ]
        return Transient(
            time=times,
            y=np.asarray(mags),
            y_err=np.full(len(mags), 0.5),
            bands=list(bands),
            data_mode="magnitude",
            name=name,
            redshift=redshift,
        )

    transient_a = make_transient(
        "a",
        np.array([5.0, 12.0, 30.0, 45.0]),
        ["bessellb", "bessellv", "bessellr", "besselli"],
    )
    transient_b = make_transient(
        "b",
        np.array([8.0, 22.0]),
        ["bessellv", "bessellr"],
    )
    params = prior.dict_to_params(true_params)

    single_a = Likelihood(
        "arnett_spectra",
        transient_a,
        fixed_single,
        evaluation_mode="full",
    )._make_log_likelihood(prior)(params)
    single_b = Likelihood(
        "arnett_spectra",
        transient_b,
        fixed_single,
        evaluation_mode="full",
    )._make_log_likelihood(prior)(params)

    batch = BatchedDataset([transient_a, transient_b])
    batched_loglike = make_batched_log_likelihood(
        "arnett_spectra",
        {
            "redshift": jnp.array([redshift, redshift]),
            "lum_dist": jnp.array([lum_dist, lum_dist]),
        },
        prior,
        None,
        batch,
        fixed_params=fixed,
        evaluation_mode="full",
    )
    batched = batched_loglike(jnp.stack([params, params]))

    np.testing.assert_allclose(
        np.asarray(batched),
        np.asarray([single_a, single_b]),
        rtol=1e-5,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Batched flux-density likelihood: padded rows must not affect the result
# ---------------------------------------------------------------------------

def _toy_flux_model(t, nu, amp, tau, **_):
    # 1/nu makes the model blow up at nu=0, as padded rows used to have.
    return amp * jnp.exp(-t / tau) * 1e14 / nu


def test_batched_flux_density_padding_is_ignored():
    from redback_jax.inference import (
        BatchedFluxDensityDataset, Prior, Uniform,
        make_batched_flux_density_log_likelihood,
    )

    t_short = np.array([1.0, 2.0, 3.0])
    t_long = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    nu_short = np.full(3, 5e14)
    nu_long = np.full(5, 5e14)
    truth = dict(amp=2.0, tau=3.0)

    def synth(t, nu):
        return np.asarray(_toy_flux_model(jnp.asarray(t), jnp.asarray(nu), **truth))

    f_short, f_long = synth(t_short, nu_short), synth(t_long, nu_long)
    err_short, err_long = 0.1 * f_short, 0.1 * f_long

    padded = BatchedFluxDensityDataset.from_arrays(
        [t_short, t_long], [nu_short, nu_long], [f_short, f_long], [err_short, err_long],
    )
    alone = BatchedFluxDensityDataset.from_arrays(
        [t_short], [nu_short], [f_short], [err_short],
    )
    prior = Prior([Uniform(0.5, 5.0, name='amp'), Uniform(1.0, 6.0, name='tau')])

    ll_padded = make_batched_flux_density_log_likelihood(
        _toy_flux_model, padded, prior, fixed_params_batch={},
    )
    ll_alone = make_batched_flux_density_log_likelihood(
        _toy_flux_model, alone, prior, fixed_params_batch={},
    )

    params = jnp.array([[2.2, 3.5], [2.2, 3.5]])
    out_padded = ll_padded(params)
    out_alone = ll_alone(params[:1])

    assert np.all(np.isfinite(np.asarray(out_padded)))
    assert float(out_padded[0]) > -1e29, "padded rows must not trigger the -1e30 sentinel"
    np.testing.assert_allclose(float(out_padded[0]), float(out_alone[0]), rtol=1e-10)
