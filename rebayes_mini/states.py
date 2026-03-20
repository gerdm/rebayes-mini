import chex
import dataclasses
from functools import lru_cache

@chex.dataclass
class GaussState:
    mean: chex.Array
    cov: chex.Array


@chex.dataclass
class OutlierEKFState:
    """State of the outlier-detection EKF."""
    mean: chex.Array
    cov: chex.Array
    alpha: float
    beta: float
    pr_inlier: float
    tau: float


@chex.dataclass
class WOCFState:
    """Weighted observation-covariance filter state."""
    mean: chex.Array
    cov: chex.Array
    key: chex.Array
    weighting_term: float = 1.0


@chex.dataclass
class KFTState:
    """State of a linear Kalman filter with tracked residual."""
    mean: chex.Array
    cov: chex.Array
    err: chex.Array


@chex.dataclass
class RobustStState:
    """State for robust Student-t filtering with adaptive observation noise."""
    mean: chex.Array
    cov: chex.Array
    obs_cov_scale: chex.Array
    obs_cov_dof: float
    weighting_shape: float
    weighting_rate: float
    dof_shape: float
    dof_rate: float
    rho: float
    dim_obs: int


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
class GreedyRunlengtState:
    runlength: int
    log_posterior: float


@chex.dataclass
class GreedyRunlengthGaussState(GaussState):
    runlength: int
    log_posterior: float


@lru_cache(maxsize=None)
def make_greedy_runlength_state(filter_state_class):
    """
    Dynamically create a chex dataclass that inherits from `filter_state_class`
    and appends `runlength` and `log_posterior` fields — analogous to C = {**A, **B}.
    Results are cached so the same class is reused across calls.
    """
    @chex.dataclass
    class GreedyRunlengthState(filter_state_class):
        runlength: int
        log_posterior: float

    GreedyRunlengthState.__name__ = f"GreedyRunlength{filter_state_class.__name__}"
    GreedyRunlengthState.__qualname__ = f"GreedyRunlength{filter_state_class.__name__}"
    return GreedyRunlengthState
