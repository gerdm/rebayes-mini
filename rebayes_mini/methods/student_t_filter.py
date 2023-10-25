"""
Roth, Michael, et al.
"Robust Bayesian filtering and smoothing using Student's t distribution."
arXiv preprint arXiv:1703.02428 (2017).
"""

import jax
import chex
import jax.numpy as jnp

@chex.dataclass
class StudentTState:
    mean: chex.Array
    scale: chex.Array
    dof: chex.Array


class LinearFilter:
    """
    """
    def __init__(
        self, transition_matrix, transition_covariance, observation_covariance,
        dof_latent, dof_observed,
    ):
        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance
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
        mean_pred = self.transition_covariance @ bel.mean
        cov_pred = self.transition_matrix @ bel.scale @ self.transition_matrix.T + self.transition_covariance
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
        K = jnp.linalg.solve(S, obs_matrix @ bel.scale).T
        err = y - yhat

        mean_new = bel.mean + K @ err
        dof_prime = jnp.minimum(bel.dof, self.dof_observed)
        scale_new_prime = bel.scale - K @ S @ K.T
        dof_new = bel.dof + len(y)
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
        ...