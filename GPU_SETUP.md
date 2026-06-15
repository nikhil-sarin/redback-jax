# redback-jax on GPU with `uv` — setup & porting guide

This documents a known-good, reproducible setup for running the **nested sampling**
example (`examples/arnett_ns.py`) on an NVIDIA GPU using JAX, installed entirely
with [`uv`](https://docs.astral.sh/uv/). It is written so it can be ported to
another machine with minimal changes.

## What was verified

Confirmed working end-to-end on:

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 4500 Ada Generation (sm_89, 24 GB) |
| NVIDIA driver | 610.43.02 |
| Python | 3.12.13 (uv-managed) |
| `jax` / `jaxlib` | 0.10.1 (`jax[cuda12]` wheels, CUDA 12.9 bundled) |
| `blackjax` | `0.1.0b1.dev194+g2180e29ff` (handley-lab `nested_sampling` branch) |
| `anesthetic` | 2.14.7 |
| `numpy` / `scipy` | 2.4.6 / 1.17.1 |

`examples/arnett_ns.py` runs on `cuda:0` in ~80 s and recovers the injected
parameters; a corner plot is written to `examples/results/corner.png`.

## Prerequisites

1. **An NVIDIA GPU + recent driver.** You do **not** need a system CUDA toolkit
   installed — the `jax[cuda12]` wheels bundle their own CUDA 12 + cuDNN. You only
   need the kernel driver (check with `nvidia-smi`). The driver's reported CUDA
   version must be ≥ 12 (here it is 13.3, which is backward-compatible with the
   CUDA 12 runtime the wheels ship).
2. **`uv`** (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

## Setup (copy/paste)

Run from the repository root. `uv` manages the Python interpreter, so the system
Python (e.g. a conda base env) is irrelevant.

```bash
# 1. Create a Python 3.12 virtual environment (uv downloads CPython if needed)
uv venv --python 3.12 .venv

# All uv pip commands below target this venv explicitly via VIRTUAL_ENV so they
# don't accidentally pick up an active conda env. UV_LINK_MODE=copy silences a
# hardlink warning when the uv cache and .venv are on different filesystems.
export VIRTUAL_ENV="$PWD/.venv"
export UV_LINK_MODE=copy

# 2. JAX with bundled CUDA 12 (GPU). Do this FIRST and verify the GPU before
#    installing anything else, so a GPU problem is isolated from package issues.
uv pip install "jax[cuda12]"

# 3. redback-jax itself, editable, CORE deps only.
#    NB: do NOT use the `inference` extra — see "Gotcha" below.
uv pip install -e .

# 4. The nested-sampling stack: handley-lab blackjax NS fork + anesthetic.
#    (tqdm and wcosmo are already pulled in as transitive deps of the above.)
uv pip install \
  "blackjax @ git+https://github.com/handley-lab/blackjax@nested_sampling" \
  anesthetic
```

### Verify the GPU is seen by JAX

```bash
.venv/bin/python -c "import jax; print(jax.default_backend(), jax.devices())"
# expected: gpu [CudaDevice(id=0)]
```

### Run the example

```bash
.venv/bin/python examples/arnett_ns.py
# prints 'device: cuda:0', a logZ estimate, a parameter table,
# and writes examples/results/corner.png + examples/results/chains/
```

## Gotcha: the *wrong* blackjax

redback-jax's `pyproject.toml` declares an `inference` optional-dependency group
with **`blackjax>=1.0.0`**. That pulls the **PyPI** blackjax, which is a *different
package* from the nested-sampling code this project's `NestedSampler` needs.

The NS code requires the **handley-lab fork**, `nested_sampling` branch
(`blackjax.nss`, `blackjax.ns.utils`, `state.integrator.logZ`). Its version is
`0.1.0b1.dev…`, i.e. **< 1.0.0**, so it does *not* satisfy the `inference` pin.

Therefore:
- Install redback-jax with **`-e .`** (core), **not** `-e .[inference]` / `-e .[all]`.
- Install the fork explicitly as in step 4. Installing it *after* core deps
  ensures it wins if anything else dragged in PyPI blackjax.

Confirm you have the fork (version must start `0.1.0b1.dev`):

```bash
.venv/bin/python -c "import blackjax; print(blackjax.__version__); \
  from blackjax.ns.utils import log_weights, finalise; print('NS API OK', hasattr(blackjax,'nss'))"
```

## Porting to another machine

- **Same recipe works as-is** as long as the target has an NVIDIA driver. Nothing
  is pinned to this host's CUDA install because CUDA ships inside the wheels.
- **CPU-only / non-NVIDIA host:** replace step 2 with `uv pip install jax` (plain).
  The examples still run, just on CPU.
- **Pin for exact reproducibility:** to freeze the fork to the exact commit used
  here, replace the branch with the commit in step 4:
  `blackjax @ git+https://github.com/handley-lab/blackjax@2180e29ffb645b2c46b76c768229c7c24212446c`
- **Different GPU:** any CUDA-capable card with a driver ≥ CUDA 12 works; no change
  needed. For very new/old cards just ensure the driver is current.

## Notes on the examples (API drift fixed)

Two pre-existing issues were fixed so the examples run against the current libraries:

1. **`examples/arnett_ns.py` and `examples/arnett_mcmc.py` data generation** called
   `bandmag(..., band_indices=, bridges=, unique_bands=)`, a multi-band signature
   that only `bandflux` supports. Changed to the single-band
   `bandmag({'amplitude': 1.0}, band, obs_times_rel)`.

2. **`redback_jax/inference/nested_sampler.py`** was written against an older
   blackjax NS API. It was ported to the current `nested_sampling` branch:
   - evidence now lives on the integrator: `state.integrator.logZ` /
     `state.integrator.logZ_live` (was `state.logZ` / `state.logZ_live`);
   - `algo.step(key, state)` returns `(state, NSInfo)`; dead-point positions are at
     `info.particles.position`, with `.loglikelihood` / `.loglikelihood_birth`;
   - dead + live points are combined with `blackjax.ns.utils.finalise`, and
     `log_weights` now returns shape `(n_points, n_mc)` (marginalise over the last
     axis for evidence; mean over it for per-point weights);
   - chains are written directly in anesthetic dead-birth format.

### Sampler note

`arnett_ns.py` (nested sampling) is the recommended, well-behaved path — it
recovers the truth with tight posteriors. `arnett_mcmc.py` (NUTS) runs on GPU but
mixes poorly as written: it uses a fixed NUTS step size with no
`window_adaptation` and hard `-inf` prior boundaries, so chains pin against the
prior edges. Treat its output as indicative only unless the NUTS warmup is
improved.

## Fast likelihood modes for GPU fitting

The default `Likelihood(..., evaluation_mode="full")` keeps the original
source-model behaviour, but there are now two **opt-in** fitting accelerators:

1. `evaluation_mode="compact_source"` — still uses
   `jax_supernovae.timeseries_multiband_flux`, but builds the source on a
   dataset-specific phase grid instead of the model's large default grid.
2. `evaluation_mode="direct_photometry"` — bypasses full source-cube
   materialization and integrates the blackbody model directly through the
   precomputed bandpasses. This is currently the fastest path for
   `arnett_spectra` photometric inference.

Example:

```python
likelihood = Likelihood(
    model='arnett_spectra',
    transient=transient,
    fixed_params=FIXED,
    evaluation_mode='direct_photometry',
)
```

If you want to keep the full `jax_supernovae` source/interpolation path, use
`compact_source` instead. The defaults are unchanged either way.
