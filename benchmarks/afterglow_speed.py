"""Compare native Redback and redback-jax afterglow execution times.

The benchmark excludes import time and reports compilation separately from
steady-state execution.  Run it from the redback-jax repository, for example::

    conda run -n py312 python benchmarks/afterglow_speed.py \
        --redback-path /path/to/redback --resolution 20

Use a fresh process for each resolution so both compilation measurements are
cold.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys
import tempfile
from time import perf_counter

import numpy as np


def _timed_call(function, repetitions):
    samples = []
    result = None
    for _ in range(repetitions):
        start = perf_counter()
        result = function()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        samples.append(perf_counter() - start)
    return result, samples


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--redback-path",
        type=Path,
        required=True,
        help="Path to a native Redback source checkout.",
    )
    parser.add_argument("--resolution", type=int, default=20)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=30)
    return parser.parse_args()


def main():
    args = _arguments()
    checkout = args.redback_path.expanduser().resolve()
    if not (checkout / "redback").is_dir():
        raise SystemExit(f"No Redback package found below {checkout}")

    # Redback's Numba decorators need a writable cache. A new directory also
    # makes its first-call compilation cost reproducible.
    numba_cache = tempfile.TemporaryDirectory(prefix="redback-numba-benchmark-")
    os.environ["NUMBA_CACHE_DIR"] = numba_cache.name
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-benchmark-"))
    sys.path.insert(0, str(checkout))

    import jax
    import redback
    from redback.transient_models.afterglow_models import (
        tophat_redback as native_tophat,
    )
    from redback_jax.models.afterglow_models import tophat_redback as jax_tophat

    time = np.geomspace(0.1, 1000.0, args.epochs)
    common = dict(
        redshift=0.01,
        thv=0.05,
        loge0=52.0,
        thc=0.2,
        logn0=0.0,
        p=2.2,
        logepse=-1.0,
        logepsb=-2.0,
        g0=100.0,
        xiN=1.0,
        frequency=3.0e9,
        output_format="flux_density",
        k=0.0,
        expansion=False,
        steps=args.steps,
    )

    print(f"Redback {redback.__version__}; JAX {jax.__version__}; {jax.devices()[0]}")
    print(f"{args.epochs} epochs at 3 GHz; steps={args.steps}")
    print("res  jax cold  native cold  cold x  jax warm  native warm  warm x  rel flux")
    kwargs = {**common, "res": args.resolution}

    jax_result, jax_cold = _timed_call(
        lambda: jax_tophat(time, **kwargs), repetitions=1
    )
    native_result, native_cold = _timed_call(
        lambda: native_tophat(time, **kwargs), repetitions=1
    )
    jax_result, jax_warm = _timed_call(
        lambda: jax_tophat(time, **kwargs), repetitions=args.repetitions
    )
    native_result, native_warm = _timed_call(
        lambda: native_tophat(time, **kwargs), repetitions=args.repetitions
    )

    jax_warm_median = statistics.median(jax_warm)
    native_warm_median = statistics.median(native_warm)
    relative_flux = abs(
        np.asarray(jax_result).sum() - np.asarray(native_result).sum()
    ) / abs(np.asarray(native_result).sum())
    print(
        f"{args.resolution:3d}  {jax_cold[0]:8.4f}  {native_cold[0]:11.4f}  "
        f"{native_cold[0] / jax_cold[0]:6.2f}  {jax_warm_median:8.5f}  "
        f"{native_warm_median:11.5f}  {native_warm_median / jax_warm_median:6.2f}  "
        f"{relative_flux:.3e}"
    )


if __name__ == "__main__":
    main()
