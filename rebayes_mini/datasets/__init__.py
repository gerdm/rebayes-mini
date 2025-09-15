"""
Synthetic Datasets for Testing and Benchmarking

This module provides synthetic datasets and data generators for testing
and benchmarking Bayesian filtering algorithms. All datasets are designed
to work seamlessly with JAX and support various challenging scenarios
for filter evaluation.

Available Datasets:
    linear_ssm.py: Linear State Space Models with various noise patterns
        - ContaminatedSSM: SSM with contaminated observations (outliers)
        - Standard linear Gaussian SSM
        - Multi-dimensional state spaces
        - Time-varying parameters

The datasets support:
    - JAX random key management for reproducibility
    - Configurable noise levels and contamination rates
    - Ground truth state sequences for evaluation
    - Various observation patterns (missing data, irregular sampling)

Example:
    >>> from rebayes_mini.datasets.linear_ssm import ContaminatedSSM
    >>> import jax
    >>> 
    >>> key = jax.random.PRNGKey(42)
    >>> ssm = ContaminatedSSM(
    ...     transition_matrix=jnp.eye(2),
    ...     projection_matrix=jnp.ones((1, 2)),
    ...     dynamics_covariance=0.1 * jnp.eye(2),
    ...     observation_covariance=jnp.array([[1.0]]),
    ...     p_contamination=0.1,
    ...     contamination_value=10.0
    ... )
    >>> 
    >>> # Generate synthetic data
    >>> initial_state = jnp.zeros(2)
    >>> data = ssm.sample(key, initial_state, n_steps=100)
"""

__all__ = ["linear_ssm"]