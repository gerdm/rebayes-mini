import chex

@chex.dataclass
class GaussState:
    mean: chex.Array
    cov: chex.Array


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