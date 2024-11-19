import einops
import jax.numpy as jnp
from functools import partial
from rebayes_mini.auxiliary.runlenght import Runlength
from rebayes_mini.states.gaussian import GaussRunlenght
from rebayes_mini.updater.full_rank_gaussian import GaussianFilter


class GaussianRunlenghtPriorReset(Runlength):
    """
    Gaussian measurement model with runlength and prior reset.
    Runlength prior reset (RL-PR)
    """
    def __init__(self, apply_fn, p_change, K, observation_variance):
        super().__init__(p_change, K)
        self.updater = GaussianFilter(apply_fn, observation_variance)

    def log_predictive_density(self, y, X, bel):
        return self.updater.log_predictive_density(y, X, bel)

    def update_bel(self, bel, y, X):
        bel = self.updater.update(bel, y, X)
        return bel

    def init_bel(self, mean, cov, log_joint_init=0.0):
        """
        Initialize belief state
        """
        state_updater = self.updater.init_bel(mean, cov)
        mean = state_updater.mean
        cov = state_updater.cov

        bel = GaussRunlenght(
            mean=einops.repeat(mean, "i -> k i", k=self.K),
            cov=einops.repeat(cov, "i j -> k i j", k=self.K),
            log_joint=(jnp.ones((self.K,)) * -jnp.inf).at[0].set(log_joint_init),
            runlength=jnp.zeros(self.K)
        )

        return bel
