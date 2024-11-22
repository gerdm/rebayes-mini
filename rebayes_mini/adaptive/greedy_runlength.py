import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from rebayes_mini import callbacks
from rebayes_mini.states import gaussian


class GreedyRunlength(ABC):
    def __init__(self, p_change, threshold=0.5):
        self.p_change = p_change
        self.threshold = threshold


    @abstractmethod
    def init_bel(self, y, X, bel_init):
        ...


    @abstractmethod
    def log_predictive_density(self, y, X, bel):
        ...


    @abstractmethod
    def update_bel(self, y, X, bel):
        """
        Update belief state (posterior)
        """
        ...


    @abstractmethod
    def conditional_prior(self, bel, bel_prior):
        ...


    def compute_log_posterior(self, y, X, bel, bel_prior):
        log_joint_increase = self.log_predictive_density(y, X, bel) + jnp.log1p(-self.p_change)
        log_joint_reset = self.log_predictive_density(y, X, bel_prior) + jnp.log(self.p_change)

        # Concatenate log_joints
        log_joint = jnp.array([log_joint_reset, log_joint_increase])
        log_joint = jnp.nan_to_num(log_joint, nan=-jnp.inf, neginf=-jnp.inf)

        # Compute log-posterior before reducing
        log_posterior_increase = log_joint_increase - jax.nn.logsumexp(log_joint)
        log_posterior_reset = log_joint_reset - jax.nn.logsumexp(log_joint)

        return log_posterior_increase, log_posterior_reset


    def step(self, bel, y, X, bel_prior, callback_fn):
        """
        Update belief state and log-joint for a single observation
        """
        log_posterior_increase, log_posterior_reset = self.compute_log_posterior(y, X, bel, bel_prior)
        bel_update = bel.replace(runlength=bel.runlength + 1, log_posterior=log_posterior_increase)
        bel_update = self.update_bel(bel_update, y, X)

        posterior_increase = jnp.exp(log_posterior_increase)
        bel_prior = bel_prior.replace(log_posterior=log_posterior_reset)

        no_changepoint = posterior_increase >= self.threshold
        bel_update = jax.tree.map(
            lambda update, prior: update * no_changepoint + prior * (1 - no_changepoint),
            bel_update, bel_prior
        )

        out = callback_fn(bel_update, bel, y, X, self)
        return bel_update, out


    def scan(self, bel, y, X, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        bel_prior = bel
        def _step(bel, yX):
            y, X = yX
            bel, out = self.step(bel, y, X, bel_prior, callback_fn)
            return bel, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist


class GaussianOUPriorReset(GreedyRunlength):
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
        bel = gaussian.GaussGreedyRunlenght.init_bel(mean, cov, log_posterior_init)
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
