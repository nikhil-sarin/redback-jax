# Redback-JAX

[![Documentation Status](https://readthedocs.org/projects/redback-jax/badge/?version=latest)](https://redback-jax.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/nikhil-sarin/redback-jax/workflows/Tests/badge.svg)](https://github.com/nikhil-sarin/redback-jax/actions)

A lightweight JAX-only version of the [redback](https://github.com/nikhil-sarin/redback) electromagnetic transient analysis package.

## Overview

Redback-JAX provides JAX-based implementations for electromagnetic transient modeling and Bayesian inference, focusing on performance and automatic differentiation capabilities. This package is designed to be a lightweight alternative to the full redback package, leveraging JAX for fast computation and gradient-based inference.

## Features

- **JAX-based models**: Fast, differentiable implementations of electromagnetic transient models
- **Bayesian inference**: Integration with modern probabilistic programming libraries
- **Automatic differentiation**: Gradient-based optimization and sampling
- **GPU/TPU support**: Leverage JAX's hardware acceleration capabilities

## Installation

### From PyPI (when available)
```bash
pip install redback-jax
```

### From source
```bash
git clone https://github.com/nikhil-sarin/redback-jax.git
cd redback-jax
pip install -e .
```

### With optional dependencies
```bash
pip install redback-jax[all]
```

## Quick Start

```python
import redback_jax

# Example usage will be added as the package develops
```

## Dependencies

**Python 3.12+ required**

Core dependencies:
- JAX (>= 0.4.0)
- NumPy (>= 1.20.0)
- SciPy (>= 1.7.0)
- Pandas (>= 1.3.0)
- Matplotlib (>= 3.5.0)
- Astropy (>= 4.0.0)

Optional dependencies (install with `[all]`):
- NumPyro (>= 0.12.0)
- BlackJAX (>= 1.0.0)
- Optax (>= 0.1.0)

## Contributing

Contributions are welcome! Please see the contributing guidelines in the main [redback repository](https://github.com/nikhil-sarin/redback) for details.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Documentation

Full documentation is available at [redback-jax.readthedocs.io](https://redback-jax.readthedocs.io/en/latest/).

## Acknowledgments

This package is based on the original [redback](https://github.com/nikhil-sarin/redback) package. Please cite the original redback paper if you use this software in your research.

## Related Projects

- [redback](https://github.com/nikhil-sarin/redback) - The full-featured electromagnetic transient analysis package
- [JAX](https://github.com/google/jax) - The underlying numerical computing library