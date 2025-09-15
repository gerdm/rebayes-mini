# Rebayes Mini: Minimalist Recursive Bayesian Estimation

A lightweight Python library implementing various recursive Bayesian estimation methods, built with JAX for high-performance automatic differentiation and JIT compilation.

## Overview

Rebayes Mini provides implementations of state-of-the-art Bayesian filtering and online learning algorithms, including:

- **Kalman Filters**: Standard, Extended, and Square-root variants
- **Robust Filters**: Student-t and other heavy-tailed distributions
- **Low-rank Approximations**: Memory-efficient filtering for high-dimensional problems
- **Gaussian Processes**: Online GP regression and classification
- **Ensemble Methods**: Ensemble Kalman Filters
- **Changepoint Detection**: Bayesian Online Changepoint Detection (BOCD)
- **Adaptive Methods**: Various adaptive filtering techniques

## Installation

### From Source (Recommended)
```bash
git clone https://github.com/gerdm/rebayes-mini.git
cd rebayes-mini
pip install -e .
```

### Direct from GitHub
```bash
pip install git+https://github.com/gerdm/rebayes-mini.git
```

### Dependencies
Rebayes Mini requires Python ≥3.9 and the following packages:
- JAX ≥0.4.2 (automatic differentiation and JIT compilation)
- NumPy ≥1.12 (numerical operations)
- TensorFlow Probability (JAX backend)
- Chex (type safety and testing)
- Additional: Einops, Optax, Distrax, Flax

For examples with visualization:
```bash
pip install matplotlib
```

## Quick Start

```python
import jax.numpy as jnp
from rebayes_mini.methods.gauss_filter import KalmanFilter
from rebayes_mini.states import GaussState

# Simple 1D Kalman filter example
kf = KalmanFilter(
    transition_matrix=jnp.array([[1.0]]),
    dynamics_covariance=jnp.array([[0.1]]),
    observation_covariance=jnp.array([[1.0]])
)

# Initialize belief state
initial_mean = jnp.array([0.0])
bel = kf.init_bel(initial_mean, cov=1.0)

# Process observations
observations = jnp.array([1.0, 2.0, 1.5])
for y in observations:
    bel, _ = kf.step(bel, y, jnp.array([[1.0]]), lambda *args: None)
    print(f"Updated mean: {bel.mean}")
```

## Project Structure

```
rebayes_mini/
├── __init__.py           # Package initialization
├── states.py            # State data structures for all filters
├── callbacks.py         # Callback functions for filter outputs
├── methods/             # Core filtering algorithms
│   ├── base_filter.py   # Abstract base classes
│   ├── gauss_filter.py  # Kalman filter implementations
│   ├── robust_filter.py # Robust filtering methods
│   ├── low_rank_*.py    # Low-rank approximation filters
│   ├── gaussian_process.py # GP-based methods
│   └── ...              # Other specialized filters
└── datasets/            # Synthetic datasets for testing
    └── linear_ssm.py    # Linear state space models
```

## Key Features

- **JAX-native**: Leverages JAX for automatic differentiation and JIT compilation
- **Functional design**: Immutable state updates and pure functions
- **Modular architecture**: Easy to extend with new filtering methods
- **Memory efficient**: Low-rank approximations for high-dimensional problems
- **Robust**: Handles outliers and non-Gaussian noise
- **Fast**: JIT-compiled implementations for production use

## Documentation

- [API Reference](docs/api.md) - Detailed API documentation
- [Examples](examples/) - Jupyter notebooks with usage examples
- [Developer Guide](docs/developer.md) - Guidelines for contributing

## Dependencies

- JAX >= 0.4.2
- NumPy >= 1.12
- TensorFlow Probability (JAX backend)
- Chex, Einops, Optax, Distrax, Flax

## License

MIT License - see LICENSE file for details.
