"""
Roth, Michael, et al.
"Robust Bayesian filtering and smoothing using Student's t distribution."
arXiv preprint arXiv:1703.02428 (2017).
"""

import jax
import chex
import jax.numpy as jnp
from functools import partial
from rebayes_mini import callbacks
from rebayes_mini.methods import gauss_filter as kf

@chex.dataclass
class StudentTState:
    mean: chex.Array
    scale: chex.Array
    dof: chex.Array


class LinearFilter(kf.KalmanFilter):
    """
    """
    def __init__(
        self, transition_matrix, dynamics_covariance, observation_covariance,
        dof_latent, dof_observed,
    ):
        self.transition_matrix = transition_matrix
        self.dynamics_covariance = dynamics_covariance
        self.observation_covariance = observation_covariance
        self.dof_latent = dof_latent
        self.dof_observed = dof_observed
    
    def init_bel(self, mean, scale, dof):
        return StudentTState(
            mean=mean,
            scale=jnp.eye(len(mean)) * scale,
            dof=dof,
        )
    
    def predict_step(self, bel):
        mean_pred = self.transition_matrix @ bel.mean
        cov_pred = self.transition_matrix @ bel.scale @ self.transition_matrix.T + self.dynamics_covariance
        dof_pred = jnp.minimum(bel.dof, self.dof_latent)

        state_pred = StudentTState(
            mean=mean_pred,
            scale=cov_pred,
            dof=dof_pred,
        )
        return state_pred

    def update_step(self, bel, y, obs_matrix):
        yhat = obs_matrix @ bel.mean
        S = obs_matrix @ bel.scale @ obs_matrix.T + self.observation_covariance
        Sinv = jnp.linalg.inv(S)
        # K S = bel_scale @ obs_matrix.T
        K = jnp.linalg.solve(S, obs_matrix @ bel.scale).T
        err = y - yhat

        mean_new = bel.mean + K @ err
        dof_prime = jnp.minimum(bel.dof, self.dof_observed)
        scale_new_prime = bel.scale - K @ S @ K.T
        dof_new = dof_prime + len(y)
        scale_new = (
            dof_prime + err.T @ Sinv @ err
        ) / dof_new * scale_new_prime

        state_new = StudentTState(
            mean=mean_new,
            scale=scale_new,
            dof=dof_new,
        )
        return state_new

    def step(self, bel, y, obs_matrix, callback_fn):
        bel_pred = self.predict_step(bel)
        bel_update = self.update_step(bel_pred, y, obs_matrix)

        output = callback_fn(bel_update, bel_pred, y, obs_matrix)
        return bel_update, output
    

class ExpfamFilter(kf.ExpfamFilter):
    def __init__(
        self, apply_fn, log_partition, suff_statistic, dynamics_covariance,
        dof_latent, dof_observed,
    ):
        self.apply_fn = apply_fn
        self.log_partition = log_partition
        self.suff_statistic = suff_statistic
        self.dynamics_covariance = dynamics_covariance
        self.dof_latent = dof_latent
        self.dof_observed = dof_observed
    
    def init_bel(self, params, scale, dof):
        self.rfn, self.link_fn, init_params = self._initialise_link_fn(self.apply_fn, params)
        self.grad_link_fn = jax.jacrev(self.link_fn)
        nparams = len(init_params)

        return StudentTState(
            mean=init_params,
            scale=jnp.eye(nparams) * scale,
            dof=dof,
        )
    
    def predict_step(self, bel):
        mean_pred =  bel.mean
        cov_pred = bel.scale + self.dynamics_covariance
        dof_pred = jnp.minimum(bel.dof, self.dof_latent)

        state_pred = StudentTState(
            mean=mean_pred,
            scale=cov_pred,
            dof=dof_pred,
        )
        return state_pred

    def update_step(self, bel, y, x):
        eta = self.link_fn(bel.mean, x).astype(float)
        yhat = self.mean(eta)
        err = y - yhat
        Ht = self.grad_link_fn(bel.mean, x)
        Rt = self.covariance(eta)

        S = Ht @ bel.scale @ Ht.T + Rt
        Sinv = jnp.linalg.inv(S)
        # K S = bel_scale @ Ht.T
        K = jnp.linalg.solve(S, Ht @ bel.scale).T

        mean_new = bel.mean + K @ err
        dof_prime = jnp.minimum(bel.dof, self.dof_observed)
        scale_new_prime = bel.scale - K @ S @ K.T
        dof_new = dof_prime + len(jnp.atleast_1d(y))
        scale_new = (
            dof_prime + err.T @ Sinv @ err
        ) / dof_new * scale_new_prime

        state_new = StudentTState(
            mean=mean_new,
            scale=scale_new,
            dof=dof_new,
        )
        return state_new

    def step(self, bel, xs, callback_fn):
        x, y = xs
        bel_pred = self.predict_step(bel)
        bel_update = self.update_step(bel_pred, y, x)

        output = callback_fn(bel_update, bel_pred, y, x)
        return bel_update, output

    def scan(self, bel, y, X, callback=None):
        xs = (X, y)
        callback = callbacks.get_null if callback is None else callback
        _step = partial(self.step, callback_fn=callback)
        bels, hist = jax.lax.scan(_step, bel, xs)
        return bels, hist


class GaussianFilter(ExpfamFilter):
    def __init__(self, apply_fn, dynamics_covariance,
                 dof_latent, dof_observed, variance=1.0):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat, dynamics_covariance,
            dof_latent, dof_observed,
        )
        self.variance = variance

    @partial(jax.jit, static_argnums=(0,))
    def _log_partition(self, eta):
        return (eta ** 2 / 2).sum()

    @partial(jax.jit, static_argnums=(0,))
    def _suff_stat(self, y):
        return y / jnp.sqrt(self.variance)


class HeteroskedasticGaussianFilter(ExpfamFilter):
    def __init__(
        self, apply_fn, dynamics_covariance, dof_latent, dof_observed,
    ):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat, dynamics_covariance,
            dof_latent, dof_observed,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _log_partition(self, eta):
        eta1, eta2 = eta
        return -eta1 ** 2 / (4 * eta2) - jnp.log(-2 * eta2) / 2

    @partial(jax.jit, static_argnums=(0,))
    def _suff_stat(self, y):
        return jnp.array([y, y ** 2])
