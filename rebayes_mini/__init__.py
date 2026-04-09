"""
Rebayes Mini: Minimalist Recursive Bayesian Estimation Library

This package provides a collection of efficient recursive Bayesian estimation
algorithms implemented in JAX. It includes various filters for online learning,
state estimation, and Bayesian inference.

Main Components:
    - states: Data structures for filter belief states
    - methods: Core filtering algorithms (Kalman, robust, low-rank, etc.)
    - datasets: Synthetic datasets for testing and benchmarking
    - callbacks: Utility functions for extracting filter outputs

Example:
    >>> import jax.numpy as jnp
    >>> from rebayes_mini.methods.gauss_filter import KalmanFilter
    >>> 
    >>> # Create a simple 1D Kalman filter
    >>> kf = KalmanFilter(
    ...     transition_matrix=jnp.array([[1.0]]),
    ...     dynamics_covariance=jnp.array([[0.1]]),
    ...     observation_covariance=jnp.array([[1.0]])
    ... )
    >>> 
    >>> # Initialize and run
    >>> bel = kf.init_bel(jnp.array([0.0]))
    >>> bel, _ = kf.step(bel, 1.0, jnp.array([[1.0]]), lambda *args: None)
"""

__version__ = "0.2.0"
__author__ = "Gerardo Duran-Martin"
__email__ = "g.duran@me.com"

# Core imports for easy access
from . import states
from . import callbacks
from . import datasets

# Common state types
from .states import GaussState

__all__ = [
    "states",
    "callbacks", 
    "datasets",
    "GaussState",
]