import jax.numpy as jnp
import numpy as np
import pytest

jax_supernovae = pytest.importorskip("jax_supernovae")

from redback_jax.inference import Likelihood, Prior, Uniform
from redback_jax.models import arnett_spectra, csm_nickel_spectra
from redback_jax.models.supernova_models import arnett_with_features_cosmology
from redback_jax.sources import PrecomputedSpectraSource
from redback_jax.transient import Transient
from redback_jax.utils import luminosity_distance_cm


REDSHIFT = 0.01
LUM_DIST = luminosity_distance_cm(REDSHIFT)
TRUE_PARAMS = {'t0': 58600.0, 'f_nickel': 0.1, 'mej': 1.4, 'vej': 5000.0}
FIXED = {
    'redshift': REDSHIFT,
    'lum_dist': LUM_DIST,
    'temperature_floor': 5000.0,
    'kappa': 0.07,
    'kappa_gamma': 0.1,
}


def _make_transient():
    out = arnett_spectra(
        **FIXED,
        vej=TRUE_PARAMS['vej'],
        f_nickel=TRUE_PARAMS['f_nickel'],
        mej=TRUE_PARAMS['mej'],
    )
    source = PrecomputedSpectraSource(
        phases=out.time,
        wavelengths=out.lambdas,
        flux_grid=out.spectra,
    )
    obs_times_rel = jnp.linspace(5.0, 50.0, 6)
    obs_times_mjd = obs_times_rel + TRUE_PARAMS['t0']
    bands = ['bessellb', 'bessellv', 'bessellr', 'besselli']

    times_list, bands_list, mags_list = [], [], []
    for band in bands:
        mags = np.asarray(source.bandmag({'amplitude': 1.0}, band, obs_times_rel))
        times_list.extend(np.asarray(obs_times_mjd).tolist())
        bands_list.extend([band] * len(obs_times_rel))
        mags_list.extend(mags.tolist())

    return Transient(
        time=np.array(times_list),
        y=np.array(mags_list),
        y_err=np.full(len(mags_list), 0.5),
        bands=bands_list,
        data_mode='magnitude',
        name='arnett_likelihood_modes',
        redshift=REDSHIFT,
    )


def _make_prior():
    return Prior([
        Uniform(58580, 58620, name='t0'),
        Uniform(0.05, 0.20, name='f_nickel'),
        Uniform(0.8, 2.0, name='mej'),
        Uniform(3000, 8000, name='vej'),
    ])


def test_likelihood_fast_modes_match_full_path():
    transient = _make_transient()
    prior = _make_prior()
    params = prior.dict_to_params(TRUE_PARAMS)

    full = Likelihood(
        model='arnett_spectra',
        transient=transient,
        fixed_params=FIXED,
        evaluation_mode='full',
    )
    compact = Likelihood(
        model='arnett_spectra',
        transient=transient,
        fixed_params=FIXED,
        evaluation_mode='compact_source',
        compact_time_grid_size=256,
        compact_grid_pad_days=5.0,
    )
    direct = Likelihood(
        model='arnett_spectra',
        transient=transient,
        fixed_params=FIXED,
        evaluation_mode='direct_photometry',
    )
    grid = Likelihood(
        model='arnett_spectra',
        transient=transient,
        fixed_params=FIXED,
        evaluation_mode='grid_photometry',
        compact_time_grid_size=256,
        compact_grid_pad_days=5.0,
    )

    full_val = float(full._make_log_likelihood(prior)(params))
    compact_val = float(compact._make_log_likelihood(prior)(params))
    direct_val = float(direct._make_log_likelihood(prior)(params))
    grid_val = float(grid._make_log_likelihood(prior)(params))

    assert abs(compact_val - full_val) < 5e-2
    assert abs(direct_val - full_val) < 5e-2
    assert abs(grid_val - full_val) < 5e-2


def test_direct_photometry_requires_supported_model():
    transient = _make_transient()
    prior = _make_prior()

    likelihood = Likelihood(
        model=arnett_with_features_cosmology,
        transient=transient,
        fixed_params=FIXED,
        evaluation_mode='direct_photometry',
    )

    with pytest.raises(ValueError, match='does not support direct_photometry'):
        likelihood._make_log_likelihood(prior)


def test_grid_photometry_requires_supported_model():
    transient = _make_transient()
    prior = _make_prior()

    likelihood = Likelihood(
        model=arnett_with_features_cosmology,
        transient=transient,
        fixed_params=FIXED,
        evaluation_mode='grid_photometry',
    )

    with pytest.raises(ValueError, match='does not support grid_photometry'):
        likelihood._make_log_likelihood(prior)


def test_csm_grid_photometry_matches_full_path():
    fixed = {
        'redshift': 0.05,
        'lum_dist': 7e26,
        'temperature_floor': 3000.0,
        'kappa': 0.1,
    }
    true_params = {
        't0': 58600.0,
        'f_nickel': 0.1,
        'mej': 5.0,
        'csm_mass': 2.0,
        'vej': 10000.0,
        'eta': 1.0,
        'rho': 1e-12,
        'r0': 100.0,
        'kappa_gamma': 1.0,
        'cutoff_wavelength': 3000.0,
        'absorption_index': 1.0,
    }
    out = csm_nickel_spectra(
        **fixed,
        **{k: v for k, v in true_params.items() if k != 't0'},
    )
    source = PrecomputedSpectraSource(
        phases=out.time,
        wavelengths=out.lambdas,
        flux_grid=out.spectra,
    )
    obs_times_rel = np.array([5.0, 12.0, 30.0, 45.0, 80.0, 150.0])
    bands = ['bessellb', 'bessellv', 'bessellr', 'besselli']

    times_list, bands_list, mags_list = [], [], []
    for band in bands:
        mags = np.asarray(source.bandmag({'amplitude': 1.0}, band, obs_times_rel))
        times_list.extend((obs_times_rel + true_params['t0']).tolist())
        bands_list.extend([band] * len(obs_times_rel))
        mags_list.extend(mags.tolist())

    transient = Transient(
        time=np.array(times_list),
        y=np.array(mags_list),
        y_err=np.full(len(mags_list), 0.5),
        bands=bands_list,
        data_mode='magnitude',
        name='csm_grid_photometry',
        redshift=fixed['redshift'],
    )
    prior = Prior([
        Uniform(58580, 58620, name='t0'),
        Uniform(0.05, 0.20, name='f_nickel'),
        Uniform(1.0, 10.0, name='mej'),
        Uniform(0.5, 5.0, name='csm_mass'),
        Uniform(5000.0, 15000.0, name='vej'),
        Uniform(0.0, 2.0, name='eta'),
        Uniform(1e-13, 1e-11, name='rho'),
        Uniform(50.0, 200.0, name='r0'),
        Uniform(0.5, 2.0, name='kappa_gamma'),
        Uniform(2500.0, 3500.0, name='cutoff_wavelength'),
        Uniform(0.5, 2.0, name='absorption_index'),
    ])
    params = prior.dict_to_params(true_params)

    full_val = float(Likelihood(
        model='csm_nickel_spectra',
        transient=transient,
        fixed_params=fixed,
        evaluation_mode='full',
    )._make_log_likelihood(prior)(params))
    grid_val = float(Likelihood(
        model='csm_nickel_spectra',
        transient=transient,
        fixed_params=fixed,
        evaluation_mode='grid_photometry',
        compact_time_grid_size=200,
        compact_grid_pad_days=5.0,
    )._make_log_likelihood(prior)(params))

    assert abs(grid_val - full_val) < 5e-2
