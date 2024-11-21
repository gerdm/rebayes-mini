import jax
import jax.numpy as jnp
from rebayes_mini.auxiliary.runlength import Runlength, GreedyRunlength
from rebayes_mini.states.gaussian import GaussRunlength, GaussGreedyRunlenght


class GaussianPriorReset(Runlength):
    def __init__(self, updater, p_change, K):
        super().__init__(p_change, K)
        self.updater = updater


    def log_predictive_density(self, y, X, bel):
        return self.updater.log_predictive_density(y, X, bel)


    def update_bel(self, bel, y, X):
        bel = self.updater.update(bel, y, X)
        return bel
    

    def predict(self, bel, X):
        yhat_hypotheses = jax.vmap(self.updater.predict_fn, in_axes=(0, None))(bel.mean, X)
        log_posterior = self.get_log_posterior(bel)
        posterior = jnp.exp(log_posterior)
        yhat = jnp.einsum("km,k->m", yhat_hypotheses, posterior)
        return yhat


    def init_bel(self, mean, cov, log_joint_init=0.0):
        """
        Initialize belief state
        """
        state_updater = self.updater.init_bel(mean, cov)
        mean = state_updater.mean
        cov = state_updater.cov
        bel = GaussRunlength.init_bel(self.K, mean, cov, log_joint_init)
        return bel


class GaussianMomentMatchedPriorReset(GaussianPriorReset):
    def __init__(self, updater, p_change, K):
        super().__init__(updater, p_change, K)
    
    
    def moment_match_prior(self, bel, y, X, bel_prior):
        log_posterior = self.get_log_posterior(bel)
        weights_prior = jnp.exp(log_posterior) * self.p_change
        bel_hat = jax.vmap(self.update_bel, in_axes=(0, None, None))(bel, y, X)

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
            runlength=jnp.array(0.0)
        )
        return bel_prior


    def step(self, bel, y, X, bel_prior, callback_fn):
        """
        Update belief state and log-joint for a single observation
        """
        bel_prior = self.moment_match_prior(bel, y, X, bel_prior)
        bel_posterior, out = super().step(bel, y, X, bel_prior, callback_fn)
        return bel_posterior, out


    def init_bel(self, mean, cov, log_joint_init=0.0):
        """
        Initialize belief state
        """
        state_updater = self.updater.init_bel(mean, cov)
        mean = state_updater.mean
        cov = state_updater.cov
        bel = GaussRunlength.init_bel(self.K, mean, cov, log_joint_init)
        return bel
    


class GaussianGreedyOUPriorReset(GreedyRunlength):
    def __init__(self, updater, p_change, threshold=0.5, shock=1.0, deflate_mean=True):
        super().__init__(p_change, threshold)
        self.updater = updater
        self.deflate_mean = deflate_mean
        self.shock = shock


    def log_predictive_density(self, y, X, bel):
        return self.updater.log_predictive_density(y, X, bel)


    def init_bel(self, mean, cov, log_posterior_init=0.0):
        """
        Initialize belief state
        """
        state_updater = self.updater.init_bel(mean, cov)
        mean = state_updater.mean
        cov = state_updater.cov
        bel = GaussGreedyRunlenght.init_bel(mean, cov, log_posterior_init)
        return bel

    def conditional_prior(self, bel):
        gamma = jnp.exp(bel.log_posterior)
        dim = bel.mean.shape[0]
        deflate_mean = gamma ** self.deflate_mean

        new_mean = bel.mean * deflate_mean
        new_cov = bel.cov * gamma ** 2 + (1 - gamma ** 2) * jnp.eye(dim) * self.shock
        bel = bel.replace(mean=new_mean, cov=new_cov)
        return bel

    def update_bel(self, bel, y, X):
        bel = self.conditional_prior(bel)
        bel = self.updater.update(bel, y, X)
        return bel

    def predict(self, bel, X):
        yhat = self.updater.predict_fn(bel.mean, X)
        return yhat
