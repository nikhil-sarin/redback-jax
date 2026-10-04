"""Tests for FluxDensityLikelihood, the batched likelihood builders, dataset
validation and the grid_photometry / bandflux evaluation modes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from redback_jax.inference import (
    BatchedDataset,
    BatchedFluxDensityDataset,
    FluxDensityLikelihood,
    Likelihood,
    Prior,
    Uniform,
    make_batched_flux_density_log_likelihood,
)
from redback_jax.transient import Transient
from redback_jax.utils import luminosity_distance_cm


# ---------------------------------------------------------------------------
# Toy flux-density model: F = amp * exp(-t/tau) * (1e14 / nu)
# ---------------------------------------------------------------------------

def toy_flux(t, nu, amp, tau, **_):
    return amp * jnp.exp(-t / tau) * 1e14 / nu


def _synthetic(t, nu, amp=2.0, tau=3.0, rel_err=0.1):
    f = np.asarray(toy_flux(jnp.asarray(t), jnp.asarray(nu), amp, tau))
    return f, rel_err * f


# ---------------------------------------------------------------------------
# FluxDensityLikelihood
# ---------------------------------------------------------------------------

def test_flux_density_likelihood_matches_manual_chi2():
    t = np.linspace(1.0, 9.0, 8)
    nu = np.full(8, 5e14)
    f, e = _synthetic(t, nu)
    prior = Prior([Uniform(0.5, 5.0, name="amp"), Uniform(1.0, 6.0, name="tau")])

    like = FluxDensityLikelihood(toy_flux, t, nu, f, e, fixed_params={})
    ll = like._make_log_likelihood(prior)

    at_truth = float(ll(jnp.array([2.0, 3.0])))
    off = float(ll(jnp.array([3.0, 3.0])))
    assert at_truth == pytest.approx(0.0, abs=1e-10)
    pred = np.asarray(toy_flux(jnp.asarray(t), jnp.asarray(nu), 3.0, 3.0))
    assert off == pytest.approx(-0.5 * np.sum(((pred - f) / e) ** 2), rel=1e-8)
    assert "FluxDensityLikelihood" in repr(like) and "toy_flux" in repr(like)


def test_flux_density_likelihood_t0_and_transforms():
    t0_true = 58600.0
    t_rel = np.linspace(1.0, 9.0, 8)
    t_mjd = t_rel + t0_true
    nu = np.full(8, 5e14)
    f, e = _synthetic(t_rel, nu)
    # sample log10(amp) and map it onto the model's ``amp``
    prior = Prior([
        Uniform(58590.0, 58610.0, name="t0"),
        Uniform(-0.5, 1.0, name="log10_amp"),
        Uniform(1.0, 6.0, name="tau"),
    ])
    like = FluxDensityLikelihood(
        toy_flux, t_mjd, nu, f, e, fixed_params={},
        t0_key="t0",
        param_transforms={"log10_amp": ("amp", lambda x: 10.0 ** x)},
    )
    ll = like._make_log_likelihood(prior)
    good = float(ll(jnp.array([t0_true, np.log10(2.0), 3.0])))
    bad = float(ll(jnp.array([t0_true + 5.0, np.log10(2.0), 3.0])))
    assert good == pytest.approx(0.0, abs=1e-6)
    assert bad < good - 10.0


def test_flux_density_likelihood_nonfinite_model_gets_sentinel():
    t = np.linspace(1.0, 5.0, 4)
    nu = np.full(4, 5e14)
    f, e = _synthetic(t, nu)

    def nan_model(t, nu, amp, **_):
        return jnp.full_like(t, jnp.nan) * amp

    prior = Prior([Uniform(0.5, 5.0, name="amp")])
    ll = FluxDensityLikelihood(nan_model, t, nu, f, e, {})._make_log_likelihood(prior)
    assert float(ll(jnp.array([1.0]))) == -1e30


# ---------------------------------------------------------------------------
# Batched flux-density likelihood
# ---------------------------------------------------------------------------

def _flux_dataset():
    ts = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
    nus = [np.full(len(t), 5e14) for t in ts]
    fs, es = zip(*[_synthetic(t, nu) for t, nu in zip(ts, nus)])
    return BatchedFluxDensityDataset.from_arrays(ts, nus, fs, es, names=["a", "b"])


def test_flux_dataset_shape_names_and_defaults():
    ds = _flux_dataset()
    assert ds.shape == (2, 5) and len(ds) == 2
    assert ds.names == ("a", "b")
    assert np.asarray(ds.mask).sum(axis=1).tolist() == [3, 5]
    assert np.isinf(np.asarray(ds.obs_flux_err)[0, 3:]).all()
    np.testing.assert_allclose(np.asarray(ds.t0_refs), [1.0, 1.0])

    default_named = BatchedFluxDensityDataset.from_arrays(
        [np.array([1.0])], [np.array([5e14])], [np.array([1.0])], [np.array([0.1])],
    )
    assert default_named.names == ("sn_0",)


def test_batched_flux_density_t0_offset_transform_and_indexed():
    """t0 is an offset from each SN's first observation; .indexed agrees with the vmapped form."""
    # Two SNe with different absolute MJD ranges but the same physical light curve.
    t_rel = np.linspace(1.0, 8.0, 7)
    starts = np.array([58000.0, 59500.0])
    ts = [t_rel + s for s in starts]
    nus = [np.full(7, 5e14)] * 2
    f, e = _synthetic(t_rel, nus[0])
    ds = BatchedFluxDensityDataset.from_arrays(ts, nus, [f, f], [e, e])

    # t0 sampled as (negative) days-before-first-obs; log10_amp goes through a transform
    prior = Prior([
        Uniform(-10.0, -0.01, name="t0"),
        Uniform(-0.5, 1.0, name="log10_amp"),
        Uniform(1.0, 6.0, name="tau"),
    ])
    ll = make_batched_flux_density_log_likelihood(
        toy_flux, ds, prior, fixed_params_batch={},
        param_transforms={"log10_amp": ("amp", lambda x: 10.0 ** x)},
    )
    # first obs is at t_rel[0]=1 d after explosion  ->  t0 offset = -1
    theta = jnp.array([-1.0, np.log10(2.0), 3.0])
    out = np.asarray(ll(jnp.stack([theta, theta])))
    np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-8)

    wrong = np.asarray(ll(jnp.stack([theta, theta.at[0].set(-6.0)])))
    assert wrong[0] == pytest.approx(0.0, abs=1e-8) and wrong[1] < -5.0

    # .indexed under vmap(axis_name='batch') must reproduce the direct batched result
    indexed = jax.vmap(ll.indexed, axis_name="batch")(jnp.stack([theta, theta]))
    np.testing.assert_allclose(np.asarray(indexed), out, atol=1e-8)


def test_batched_flux_density_fixed_params_batch_and_nonfinite():
    ds = _flux_dataset()
    prior = Prior([Uniform(0.5, 5.0, name="amp")])

    def model(t, nu, amp, scale, tau=3.0, **_):
        return scale * amp * jnp.exp(-t / tau) * 1e14 / nu

    ll = make_batched_flux_density_log_likelihood(
        model, ds, prior,
        fixed_params_batch={"scale": jnp.array([1.0, 1.0])},
        fixed_params={"tau": 3.0},
    )
    out = np.asarray(ll(jnp.array([[2.0], [2.0]])))
    np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-8)

    def nan_model(t, nu, amp, **_):
        return jnp.full_like(t, jnp.nan) * amp

    ll_nan = make_batched_flux_density_log_likelihood(nan_model, ds, prior, {})
    assert np.all(np.asarray(ll_nan(jnp.array([[1.0], [1.0]]))) == -1e30)


# ---------------------------------------------------------------------------
# BatchedDataset validation
# ---------------------------------------------------------------------------

def _mag_transient(name="t", n=3, bands=None, **overrides):
    kwargs = dict(
        time=np.arange(1.0, n + 1),
        y=np.full(n, 20.0),
        y_err=np.full(n, 0.1),
        bands=bands or ["bessellb"] * n,
        data_mode="magnitude",
        name=name,
    )
    kwargs.update(overrides)
    return Transient(**kwargs)


def test_batched_dataset_rejects_empty_and_unknown_bands():
    with pytest.raises(ValueError, match="at least one transient"):
        BatchedDataset([])
    with pytest.raises(ValueError, match="absent from global band set"):
        BatchedDataset([_mag_transient()], bands=["bessellv"])
    with pytest.raises(ValueError, match="at least one photometric band"):
        BatchedDataset([_mag_transient()], bands=[])

    class NoBandsAttr:  # band inference must not crash with AttributeError
        pass

    with pytest.raises(ValueError, match="at least one photometric band"):
        BatchedDataset([NoBandsAttr()])


def test_batched_dataset_rejects_malformed_transients():
    class Bare:
        pass

    with pytest.raises(ValueError, match="missing required attributes"):
        BatchedDataset([Bare()], bands=["bessellb"])

    class Ragged:
        time = np.array([1.0, 2.0])
        y = np.array([20.0])
        y_err = np.array([0.1, 0.1])
        bands = ["bessellb", "bessellb"]

    with pytest.raises(ValueError, match="same length"):
        BatchedDataset([Ragged()], bands=["bessellb"])

    class BadBands(Ragged):
        y = np.array([20.0, 20.0])
        bands = ["bessellb"]

    with pytest.raises(ValueError, match="bands must have the same length"):
        BatchedDataset([BadBands()], bands=["bessellb"])

    class Empty:
        time = np.array([])
        y = np.array([])
        y_err = np.array([])
        bands = []

    with pytest.raises(ValueError, match="no observations"):
        BatchedDataset([Empty()], bands=["bessellb"])

    class NoneData(Ragged):
        y = None

    with pytest.raises(ValueError, match="must all be provided"):
        BatchedDataset([NoneData()], bands=["bessellb"])

    class NoneBands(Ragged):
        y = np.array([20.0, 20.0])
        bands = None

    with pytest.raises(ValueError, match="bands must be provided"):
        BatchedDataset([NoneBands()], bands=["bessellb"])


def test_batched_dataset_len_and_explicit_band_order():
    ds = BatchedDataset(
        [_mag_transient("a", bands=["bessellv"] * 3), _mag_transient("b", n=2)],
        bands=["bessellb", "bessellv"],
    )
    assert len(ds) == 2 and ds.bands == ("bessellb", "bessellv")


# ---------------------------------------------------------------------------
# Photometric modes (need jax_supernovae)
# ---------------------------------------------------------------------------

REDSHIFT = 0.01
LUM_DIST = luminosity_distance_cm(REDSHIFT)
FIXED = {"temperature_floor": 5000.0, "kappa": 0.07, "kappa_gamma": 0.1}
TRUE = {"t0": 58600.0, "f_nickel": 0.1, "mej": 1.4, "vej": 5000.0}


@pytest.fixture(scope="module")
def photometry_setup():
    pytest.importorskip("jax_supernovae")
    from redback_jax.models import arnett_spectra
    from redback_jax.sources import PrecomputedSpectraSource

    prior = Prior([
        Uniform(58580, 58620, name="t0"),
        Uniform(0.05, 0.20, name="f_nickel"),
        Uniform(0.8, 2.0, name="mej"),
        Uniform(3000, 8000, name="vej"),
    ])
    out = arnett_spectra(
        redshift=REDSHIFT, lum_dist=LUM_DIST, **FIXED,
        **{k: v for k, v in TRUE.items() if k != "t0"},
    )
    source = PrecomputedSpectraSource(
        phases=out.time, wavelengths=out.lambdas, flux_grid=out.spectra,
    )
    rel = np.array([5.0, 12.0, 30.0, 45.0])
    bands = ["bessellb", "bessellv", "bessellr", "besselli"]
    mags = np.array([float(source.bandmag({"amplitude": 1.0}, b, r)) for b, r in zip(bands, rel)])
    transient = Transient(
        time=rel + TRUE["t0"], y=mags, y_err=np.full(4, 0.3), bands=bands,
        data_mode="magnitude", name="a", redshift=REDSHIFT,
    )
    return prior, transient


def _single_like(mode, prior, transient):
    return Likelihood(
        "arnett_spectra", transient, {"redshift": REDSHIFT, "lum_dist": LUM_DIST, **FIXED},
        evaluation_mode=mode,
    )._make_log_likelihood(prior)


@pytest.mark.parametrize("mode", ["compact_source", "grid_photometry", "direct_photometry"])
def test_batched_modes_match_full_path(photometry_setup, mode):
    from redback_jax.inference import make_batched_log_likelihood

    prior, transient = photometry_setup
    params = prior.dict_to_params(TRUE)
    reference = float(_single_like("full", prior, transient)(params))

    batch = BatchedDataset([transient])
    ll = make_batched_log_likelihood(
        "arnett_spectra",
        {"redshift": jnp.array([REDSHIFT]), "lum_dist": jnp.array([LUM_DIST])},
        prior, None, batch, fixed_params=FIXED, evaluation_mode=mode,
    )
    out = float(ll(params[None, :])[0])
    assert np.isfinite(out)
    # fast modes approximate the full path; agree well where the fit is decent
    assert out == pytest.approx(reference, rel=0.05, abs=0.5)


def test_bandflux_mode_flux_space_likelihood(photometry_setup):
    """bandflux uses a Gaussian in flux space: finite, maximal near truth."""
    from redback_jax.inference import make_batched_log_likelihood

    prior, transient = photometry_setup
    good = prior.dict_to_params(TRUE)
    bad = prior.dict_to_params({**TRUE, "mej": 2.0, "vej": 3200.0})

    single = _single_like("bandflux", prior, transient)
    assert np.isfinite(float(single(good)))
    assert float(single(good)) > float(single(bad))

    batch = BatchedDataset([transient])
    ll = make_batched_log_likelihood(
        "arnett_spectra",
        {"redshift": jnp.array([REDSHIFT]), "lum_dist": jnp.array([LUM_DIST])},
        prior, None, batch, fixed_params=FIXED, evaluation_mode="bandflux",
    )
    batched = np.asarray(ll(good[None, :]))
    np.testing.assert_allclose(batched[0], float(single(good)), rtol=1e-4, atol=1e-4)


def test_batched_likelihood_input_validation(photometry_setup):
    from redback_jax.inference import make_batched_log_likelihood

    prior, transient = photometry_setup
    batch = BatchedDataset([transient])
    ok = {"redshift": jnp.array([REDSHIFT]), "lum_dist": jnp.array([LUM_DIST])}

    with pytest.raises(ValueError, match="leading size"):
        make_batched_log_likelihood(
            "arnett_spectra", {"redshift": jnp.array([REDSHIFT, REDSHIFT]), "lum_dist": ok["lum_dist"]},
            prior, None, batch, fixed_params=FIXED,
        )
    with pytest.raises(ValueError, match="evaluation_mode must be one of"):
        make_batched_log_likelihood(
            "arnett_spectra", ok, prior, None, batch, fixed_params=FIXED, evaluation_mode="nope",
        )

    def unsupported(*args, **kwargs):  # no fast-path attributes
        raise AssertionError("should not be evaluated")

    for mode in ("compact_source", "direct_photometry", "grid_photometry", "bandflux"):
        with pytest.raises(ValueError, match="does not support"):
            make_batched_log_likelihood(
                unsupported, ok, prior, None, batch, fixed_params=FIXED, evaluation_mode=mode,
            )


# ---------------------------------------------------------------------------
# CutoffBlackbody spectra factory (general magnetar, diffrax)
# ---------------------------------------------------------------------------

MAG_FIXED = {
    "redshift": REDSHIFT,
    "lum_dist": LUM_DIST,
    "temperature_floor": 3000.0,
    "log10_E_sn": 51.0,
    "kappa": 0.1,
    "tau_sd": 1e6,
    "nn": 3.0,
    "kappa_gamma": 1.0,
    "f_nickel": 0.1,
}
MAG_TRUE = {"t0": 58600.0, "mej": 1.0, "log10_l0": 45.0}


def test_cutoff_spectra_model_output():
    pytest.importorskip("jax_supernovae")
    from redback_jax.models import general_magnetar_supernova_spectra_diffrax as model

    out = model(**{k: v for k, v in MAG_FIXED.items()}, mej=1.0, log10_l0=45.0)
    spectra = np.asarray(out.spectra)
    assert spectra.shape == (len(out.time), len(out.lambdas))
    assert np.all(np.isfinite(spectra)) and np.all(spectra >= 0)
    assert spectra.max() > 0

    # the UV cutoff must suppress flux blueward of the cutoff relative to a higher cutoff
    hard = model(**MAG_FIXED, mej=1.0, log10_l0=45.0, cutoff_wavelength=6000.0)
    lam = np.asarray(out.lambdas)
    uv = lam < 3000.0
    assert np.asarray(hard.spectra)[:, uv].sum() < spectra[:, uv].sum()


def test_cutoff_spectra_grid_photometry_matches_full_path():
    pytest.importorskip("jax_supernovae")
    from redback_jax.models import general_magnetar_supernova_spectra_diffrax as model
    from redback_jax.sources import PrecomputedSpectraSource

    out = model(**MAG_FIXED, **{k: v for k, v in MAG_TRUE.items() if k != "t0"})
    source = PrecomputedSpectraSource(
        phases=out.time, wavelengths=out.lambdas, flux_grid=out.spectra,
    )
    rel = np.array([10.0, 25.0, 40.0, 60.0])
    bands = ["bessellb", "bessellv", "bessellr", "besselli"]
    mags = np.array([float(source.bandmag({"amplitude": 1.0}, b, r)) for b, r in zip(bands, rel)])
    transient = Transient(
        time=rel + MAG_TRUE["t0"], y=mags, y_err=np.full(4, 0.3), bands=bands,
        data_mode="magnitude", name="mag", redshift=REDSHIFT,
    )
    prior = Prior([
        Uniform(58580, 58620, name="t0"),
        Uniform(0.5, 3.0, name="mej"),
        Uniform(44.0, 46.0, name="log10_l0"),
    ])
    fixed = {k: v for k, v in MAG_FIXED.items()}
    params = prior.dict_to_params(MAG_TRUE)

    def loglike(mode):
        return float(Likelihood(
            "general_magnetar_supernova_spectra_diffrax", transient, fixed,
            evaluation_mode=mode,
        )._make_log_likelihood(prior)(params))

    full = loglike("full")
    grid = loglike("grid_photometry")
    assert np.isfinite(full) and np.isfinite(grid)
    assert grid == pytest.approx(full, rel=0.05, abs=0.5)
