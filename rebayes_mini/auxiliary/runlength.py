import jax
import jax.numpy as jnp
from functools import partial
from abc import ABC, abstractmethod
from rebayes_mini import callbacks

class Runlength(ABC):
    def __init__(self, p_change, K):
        self.p_change = p_change
        self.K = K


    @abstractmethod
    def init_bel(self, y, X, bel_init):
        ...


    @abstractmethod
    def log_predictive_density(self, y, X, bel):
        ...


    @abstractmethod
    def update_bel(self, bel, y, X):
        """
        Update belief state (posterior)
        """
        ...


    def get_log_posterior(self, bel):
        log_posterior = bel.log_joint - jax.nn.logsumexp(bel.log_joint)
        return log_posterior

    
    def get_expected_runlength(self, bel):
        log_posterior = self.get_log_posterior(bel)
        posterior = jnp.exp(log_posterior)
        return jnp.nansum(bel.runlength * posterior)


    def get_mode_runlength(self, bel):
        log_posterior = self.get_log_posterior(bel)
        kmax = jnp.argmax(log_posterior)
        return bel.runlength[kmax]


    @partial(jax.vmap, in_axes=(None, None, None, 0))
    def update_log_joint_increase(self, y, X, bel):
        log_p_pred = self.log_predictive_density(y, X, bel)
        log_joint = log_p_pred + bel.log_joint + jnp.log(1 - self.p_change)
        return log_joint


    def update_log_joint_reset(self, y, X, bel, bel_prior):
        log_p_pred = self.log_predictive_density(y, X, bel_prior)
        log_joint = log_p_pred + jax.nn.logsumexp(bel.log_joint) + jnp.log(self.p_change)
        return jnp.atleast_1d(log_joint)


    def update_log_joint(self, y, X, bel, bel_prior):
        log_joint_reset = self.update_log_joint_reset(y, X, bel, bel_prior)
        log_joint_increase = self.update_log_joint_increase(y, X, bel)
        # Expand log-joint
        log_joint = jnp.concatenate([log_joint_reset, log_joint_increase])
        log_joint = jnp.nan_to_num(log_joint, nan=-jnp.inf, neginf=-jnp.inf)
        # reduce to K values --- index 0 is a changepoint
        _, top_indices = jax.lax.top_k(log_joint, k=self.K)
        return log_joint, top_indices


    def update_beliefs(self, bel, y, X, bel_prior):
        """
        Update belief state (posterior) for the chosen indices
        """
        # Update all belief states if a changepoint did not happen
        vmap_update_bel = jax.vmap(self.update_bel, in_axes=(0, None, None))
        bel = vmap_update_bel(bel, y, X)
        # Update all runlenghts
        bel = bel.replace(runlength=bel.runlength+1)
        # Increment belief state by adding bel_prior
        bel = jax.tree.map(lambda prior, updates: jnp.concatenate([prior[None], updates]), bel_prior, bel)
        return bel


    def step(self, bel, y, X, bel_prior, callback_fn):
        """
        Update belief state and log-joint for a single observation
        """
        log_joint_full, top_indices = self.update_log_joint(y, X, bel, bel_prior)
        bel_posterior = self.update_beliefs(bel, y, X, bel_prior)
        bel_posterior = bel_posterior.replace(log_joint=log_joint_full)
        bel_posterior = jax.tree.map(lambda param: jnp.take(param, top_indices, axis=0), bel_posterior)
        out = callback_fn(bel_posterior, bel, y, X, self)

        return bel_posterior, out


    def scan(self, bel, y, X, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        bel_prior = jax.tree.map(lambda x: x[0], bel)
        def _step(bel, yX):
            y, X = yX
            bel, out = self.step(bel, y, X, bel_prior, callback_fn)
            return bel, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist


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
        bel_update = self.conditional_prior(bel_update, bel_prior)
        bel_update = self.update_bel(y, X, bel_update)

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
