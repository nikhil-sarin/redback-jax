# Nested Sampling Examples

This directory contains examples of using nested sampling to fit transient models to photometric data.

## Float32 mode

These examples run in float32 (GPU-safe). The Arnett model and all other
bolometric models return `log10_lbol` rather than linear luminosity — the
spectra pipeline consumes this directly without ever materialising the linear
value. No configuration is needed; JAX defaults to float32 unless `jax_enable_x64`
is set globally.

## Requirements

These examples require the Handley Lab fork of BlackJAX (not yet merged with the main branch):

```bash
pip install git+https://github.com/handley-lab/blackjax@proposal
pip install anesthetic  # For corner plots
```

For more information, see the [Nested Sampling Book](https://handley-lab.co.uk/nested-sampling-book/intro.html).

## Quick Start

**Before running nested sampling, test your setup:**

```bash
python test_setup.py
```

This will verify:
- Source creation works
- Magnitude calculations are correct
- Likelihood function is working
- Estimated runtime for nested sampling

## Examples

### `test_setup.py` - Setup Verification

Quick test to verify everything is working before running full nested sampling.

**Run time:** ~5 seconds

### `arnett_ns_quick.py` - Quick Nested Sampling Test

Simplified version with reduced settings for quick testing:
- Fewer observations (2 bands × 8 epochs)
- Fewer live points (50 instead of 125)
- Reduced time resolution

**Run time:** ~1 minute

Good for development and testing before full runs.

### `arnett_ns.py` - Full Arnett Model Nested Sampling

Demonstrates nested sampling for the Arnett supernova model using the `PrecomputedSpectraSource` API.

**What it does:**
1. Generates fake magnitude data from the Arnett model with known parameters
2. Sets up nested sampling with uniform priors
3. Fits the model parameters: `t0`, `f_nickel`, `mej`, `vej`
4. Creates corner plots showing posterior distributions
5. Compares recovered parameters to true values

**Run the example:**
```bash
python arnett_ns.py
```

**Outputs:**
- `fake_data.png` - Plot of the generated fake observations
- `corner_plot.png` - Corner plot of parameter posteriors with true values marked
- `parameter_statistics.txt` - Summary statistics comparing recovered vs true parameters
- `chains/` - Directory containing MCMC chains in anesthetic format

**Typical runtime:** ~2 minutes depending on your hardware

**Parameters being fit:**
- `t0`: Explosion time (MJD)
- `f_nickel`: Nickel mass fraction
- `mej`: Ejecta mass (solar masses)
- `vej`: Ejecta velocity (km/s)

**Fixed parameters:**
- `kappa`: Optical opacity (0.07)
- `kappa_gamma`: Gamma-ray opacity (0.1)
- `temperature_floor`: Minimum temperature (5000 K)
- `redshift`: Source redshift (0.01)

## Customization

### Adjusting Nested Sampling Settings

In the script, you can modify:

```python
NS_SETTINGS = {
    'n_delete': 20,      # Number of points to delete per iteration
    'n_live': 125,       # Number of live points
    'num_mcmc_steps_multiplier': 5  # MCMC steps = n_params × this
}
```

- **More live points** → Better sampling but slower
- **More MCMC steps** → Better exploration but slower
- **More delete points** → Faster but coarser sampling

### Changing Priors

Modify the `PRIOR_BOUNDS` dictionary:

```python
PRIOR_BOUNDS = {
    't0': {'min': 58580.0, 'max': 58620.0},
    'f_nickel': {'min': 0.05, 'max': 0.2},
    'mej': {'min': 0.8, 'max': 2.0},
    'vej': {'min': 3000.0, 'max': 8000.0}
}
```

### Using Real Data

To use real data instead of fake data, replace the data generation section with:

```python
# Load your data
times = jnp.array([...])  # Observer frame times
mags = jnp.array([...])   # Magnitudes
mag_errs = jnp.array([...])  # Magnitude errors
bands = [...]  # List of band names ('bessellb', 'bessellv', etc.)
band_indices = jnp.array([...])  # Index into unique_bands for each observation
unique_bands = ['bessellb', 'bessellv', 'bessellr', 'besselli']
```

## Understanding the Results

### Corner Plot

The corner plot shows:
- **Diagonal**: 1D marginal posterior distributions
- **Off-diagonal**: 2D joint posterior distributions
- **Red dashed lines**: True parameter values (for fake data examples)

### Parameter Statistics

The output includes:
- **True**: The true parameter value used to generate fake data
- **Mean**: Posterior mean
- **Std Dev**: Posterior standard deviation
- **Diff (σ)**: Difference between mean and true value in units of standard deviations

Good recovery should have `|Diff (σ)| < 2` for most parameters.

### Log Evidence

The log evidence (logZ) is useful for model comparison:
- Higher logZ indicates better fit to data
- Difference of ~5 in logZ suggests strong preference
- Use Bayes factors: `K = exp(logZ1 - logZ2)`

## Tips for Faster Runs

1. **Reduce time resolution**: Use fewer time points in `from_arnett_model(n_times=...)`
2. **Reduce live points**: Start with `n_live=50` for testing
3. **Use fewer observations**: Reduce `n_obs_per_band`
4. **Consider GPU**: JAX will use GPU automatically if available

## Troubleshooting

**Problem**: "Module 'blackjax' has no attribute 'nss'"
- Solution: Make sure you installed the Handley Lab fork, not the main BlackJAX
  ```bash
  pip install git+https://github.com/handley-lab/blackjax@proposal
  ```

**Problem**: Distrax import errors with JAX 0.7+
- This is a known compatibility issue between distrax and newer JAX versions
- The nested sampling examples should still work fine
- If you see distrax errors during `test_setup.py`, you can ignore them

**Problem**: Very slow performance
- Solution: Check you're using JAX with GPU (`jax.devices()`)
- Reduce `n_times` in source creation
- Reduce `n_live` and `n_delete`

**Problem**: Poor parameter recovery
- Solution: Increase `n_live` and `num_mcmc_steps_multiplier`
- Check your priors aren't too restrictive
- Add more observations or reduce noise

## References

- [BlackJAX](https://github.com/blackjax-devs/blackjax) - Probabilistic sampling in JAX
- [Nested Sampling Book](https://handley-lab.co.uk/nested-sampling-book/intro.html) - Theory and practice
- [Anesthetic](https://github.com/handley-lab/anesthetic) - Nested sampling visualization
