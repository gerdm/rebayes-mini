"""
State Data Structures for Bayesian Filters

This module defines the core state data structures used by various Bayesian
filtering algorithms in rebayes-mini. All states are implemented as immutable
dataclasses using Chex for type safety and JAX compatibility.

The state structures represent the belief (posterior distribution) of the filter
at each time step, encapsulating both the parameters of the distribution and
any auxiliary information needed for the filtering process.

Common State Types:
    GaussState: Basic Gaussian belief state (mean and covariance)
    PULSEGaussState: Gaussian state for PULSE (Probabilistic Updates of Last Subspace Estimate)
    BOCDGaussState: State for Bayesian Online Changepoint Detection
    ABOCDGaussState: Adaptive BOCD state with posterior over changepoints
    BOCHDGaussState: BOCD with unknown hazard rate
    GammaFilterState: Robust filter state with gamma-distributed precision
    
Changepoint Detection States:
    BOCDPosGaussState: BOCD with stored feature positions
    BernoullChangeGaussState: Bernoulli changepoint model
    GreedyRunlengthGaussState: Greedy runlength estimation

Specialized States:
    ABOCDLoFiState: Low-rank filter with soft changepoint detection
    MixtureExpertsGaussState: Mixture of experts model state

Each state class is decorated with @chex.dataclass for immutability and includes
type annotations using chex.Array for JAX compatibility.
"""

import chex

@chex.dataclass
class GaussState:
    mean: chex.Array
    cov: chex.Array


@chex.dataclass
class PULSEGaussState:
    mean_hidden: chex.Array
    prec_hidden: chex.Array
    mean_last: chex.Array
    prec_last: chex.Array


@chex.dataclass
class BOCDPosGaussState:
    """
    State for a
    Bayesian online changepoint detection with stored position
    of the features since the last changepoint (BOCDP)
    with Gaussian posterior
    """
    mean: chex.Array
    cov: chex.Array
    log_joint: chex.Array
    runlength: chex.Array
    last_x: chex.Array


@chex.dataclass
class BOCDGaussState:
    mean: chex.Array
    cov: chex.Array
    log_joint: chex.Array
    runlength: chex.Array


@chex.dataclass
class ABOCDGaussState:
    mean: chex.Array
    cov: chex.Array
    log_joint: chex.Array
    runlength: chex.Array
    log_posterior: chex.Array


@chex.dataclass
class BOCHDGaussState:
    """
    Bayesian Online Changepoint Detection with unknown hazard rate
    """
    mean: chex.Array
    cov: chex.Array
    log_joint: chex.Array
    runlength: chex.Array
    changepoints: chex.Array


@chex.dataclass
class BernoullChangeGaussState:
    mean: chex.Array
    cov: chex.Array
    log_weight: chex.Array
    segment: chex.Array


@chex.dataclass
class GammaFilterState:
    mean: chex.Array
    cov: chex.Array
    eta: float


@chex.dataclass
class ABOCDLoFiState:
    """
    Low-rank filter state with
    soft changepoint detection
    """
    mean: chex.Array
    diagonal: chex.Array
    low_rank: chex.Array
    runlength: int
    log_posterior: float


@chex.dataclass
class MixtureExpertsGaussState:
    mean: chex.Array
    cov: chex.Array
    factors: chex.Array
    log_weights: chex.Array


@chex.dataclass
class GreedyRunlengthGaussState:
    mean: chex.Array
    cov: chex.Array
    runlength: int
    log_posterior: float
