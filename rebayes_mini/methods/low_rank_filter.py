import jax
import chex
import jax.numpy as jnp
from functools import partial
from jax.flatten_util import ravel_pytree
from rebayes_mini.methods import kalman_filter as kf

@chex.dataclass
class LoFiState:
    """State of the Low Rank Filter"""
    mean: chex.Array
    diagonal: chex.Array
    low_rank: chex.Array

class ExpfamFilter(kf.ExpfamFilter):
    def __init__(
        self, apply_fn, log_partition, suff_statistic, dynamics_covariance,
        rank=10,
    ):
        super().__init__(apply_fn, log_partition, suff_statistic, dynamics_covariance)
        self.rank = rank
    
    def init_bel(self, params, cov=1.0):
        self.rfn, self.link_fn, init_params = self._initialise_link_fn(self.apply_fn, params)
        self.grad_link_fn = jax.jacfwd(self.link_fn)
        nparams = len(init_params)

        low_rank = jnp.zeros((nparams, self.rank))
        diagonal = jnp.ones(nparams) * cov

        return LoFiState(
            mean=init_params,
            low_rank=low_rank,
            diagonal=diagonal,
        )
    
    def _predict(self, state):
        mean_pred = state.mean
        diag_pred = 1 / (1 / state.diagonal + self.dynamics_covariance)

        C = jnp.einsum("ji,j,j,jk->ik",
            state.low_rank, diag_pred, 1 / self.diagonal, state.low_rank.T
        )
        C = jnp.linalg.inv(C)

        low_rank_pred = jnp.einsum(
            "i,i,ij,jk->ik",
            diag_pred, 1 / self.diagonal, state.low_rank, C
        )

        state_pred = state.replace(
            mean=mean_pred,
            diagonal=diag_pred,
            low_rank=low_rank_pred,
        )
        return state_pred

    def _update(self, state_pred, x, y):
        ...

    def step(self, bel, xs, callback_fn):
        xt, yt = xs
        bel_pred = self._predict(bel)

        eta = self.link_fn(bel_pred.mean, xt).astype(float)
        yhat = self.mean(eta)
        yobs = self.suff_statistic(yt)
        Rt = self.covariance(eta)
        Ht = self.grad_link_fn(bel_pred.mean, xt)

        At = jnp.linalg.inv(jnp.linalg.cholesky(Rt))
        low_rank_hat = jnp.c_[bel_pred, Ht.T @ At.T]
        Gt = jnp.linalg.inv(
            jnp.eye(self.rank + 1) + low_rank_hat.T @ low_rank_hat / self.dynamics_covariance
        )
