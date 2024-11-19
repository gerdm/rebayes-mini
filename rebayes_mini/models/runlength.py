import einops
import jax.numpy as jnp
from functools import partial
from rebayes_mini.auxiliary.runlength import Runlength, GreedyRunlength, MomentMatchedPriorReset
from rebayes_mini.states.gaussian import GaussRunlength


class GaussianPriorReset(Runlength):
    def __init__(self, updater, p_change, K):
        super().__init__(p_change, K)
        self.updater = updater

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
        bel = GaussRunlength.init_bel(self.K, mean, cov, log_joint_init)
        return bel


class GaussianMomentMatchedPriorReset(MomentMatchedPriorReset):
    def __init__(self, updater, p_change, K):
        super().__init__(p_change, K)
        self.updater = updater
    
    def log_predictive_density(self, y, X, bel):
        return self.updater.log_predictive_density(y, X, bel)

    def update_bel(self, bel, y, X):
        bel = self.updater.update(bel, y, X)
        return bel
    
    def moment_match_prior(self, bel, y, X, bel_prior):
        weights_prior = bel.log_joint - jax.nn.logsumexp(bel.log_joint)
        weights_prior = jnp.exp(weights_prior) * self.p_change
        bel_hat = jax.vmap(self.update_bel, in_axes=(None, None, 0))(y, X, bel)

        # Moment-matched mean
        mean_prior = jnp.einsum("k,kd->d", weights_prior, bel_hat.mean)
        # Moment-matched covariance
        E2 = bel_hat.cov + jnp.einsum("ki,kj->kij", bel_hat.mean, bel_hat.mean)
        cov_prior = (
            jnp.einsum("k,kij->ij", weights_prior, E2) -
            jnp.einsum("i,j->ij", mean_prior, mean_prior)
        )

        bel_prior = bel_prior.replace(
            mean=mean_prior,
            cov=cov_prior,
            runlength=0.0
        )
        return bel_prior


    def init_bel(self, mean, cov, log_joint_init=0.0):
        """
        Initialize belief state
        """
        state_updater = self.updater.init_bel(mean, cov)
        mean = state_updater.mean
        cov = state_updater.cov
        bel = GaussRunlength.init_bel(self.K, mean, cov, log_joint_init)
        return bel
    
