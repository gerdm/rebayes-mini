import chex

@chex.dataclass
class GaussState:
    mean: chex.Array
    cov: chex.Array


@chex.dataclass
class BOCDGaussState:
    mean: chex.Array
    cov: chex.Array
    log_joint: chex.Array
    runlength: chex.Array


@chex.dataclass
class BernoullChangeGaussState:
    mean: chex.Array
    cov: chex.Array
    log_weight: chex.Array
    runlength: chex.Array