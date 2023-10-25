"""
Roth, Michael, et al.
"Robust Bayesian filtering and smoothing using Student's t distribution."
arXiv preprint arXiv:1703.02428 (2017).
"""

import jax
import chex
import jax.numpy as jnp

@chex.dataclass
class StudentTState:
    mean: chex.Array
    precision: chex.Array
    dof: chex.Array

class STFilter:
    """
    """
    ...