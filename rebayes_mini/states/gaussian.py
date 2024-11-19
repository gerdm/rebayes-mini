import chex
import einops
import jax.numpy as jnp

@chex.dataclass
class GaussGreedyRunlenght:
    mean: chex.Array
    cov: chex.Array
    runlenght: int
    log_posterior: int


@chex.dataclass
class GaussRunlength:
    """
    Gaussian posterior with runlength
    """
    mean: chex.Array
    cov: chex.Array
    log_joint: chex.Array
    runlength: chex.Array

    @staticmethod
    def init_bel(K, mean, cov, log_joint_init=0.0):
        bel = GaussRunlength(
            mean=einops.repeat(mean, "i -> k i", k=K),
            cov=einops.repeat(cov, "i j -> k i j", k=K),
            log_joint=(jnp.ones((K,)) * -jnp.inf).at[0].set(log_joint_init),
            runlength=jnp.zeros(K)
        )
        return bel


@chex.dataclass
class Gaussian:
    mean: chex.Array
    cov: chex.Array