import chex

@chex.dataclass
class GaussRunlenght:
    """
    Gaussian posterior with runlenght
    """
    mean: chex.Array
    cov: chex.Array
    log_joint: chex.Array
    runlength: chex.Array

@chex.dataclass
class Gaussian:
    mean: chex.Array
    cov: chex.Array