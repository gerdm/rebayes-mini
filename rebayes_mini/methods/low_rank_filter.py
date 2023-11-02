import jax
import chex
import jax.numpy as jnp
from functools import partial
from rebayes_mini.methods import gauss_filter as kf

@chex.dataclass
class LoFiState:
    """State of the Low Rank Filter"""
    mean: chex.Array
    diagonal: chex.Array
    low_rank: chex.Array


class ExpfamFilter(kf.ExpfamFilter):
    def __init__(
        self, apply_fn, log_partition, suff_statistic, dynamics_covariance,
        rank,
    ):
        super().__init__(apply_fn, log_partition, suff_statistic, dynamics_covariance)
        self.rank = rank
    
    def init_bel(self, params, cov=1.0):
        self.rfn, self.link_fn, init_params = self._initialise_link_fn(self.apply_fn, params)
        self.grad_link_fn = jax.jacrev(self.link_fn)
        nparams = len(init_params)

        low_rank = jnp.zeros((nparams, self.rank))
        diagonal = jnp.ones(nparams) / cov # From covariance to precision term

        return LoFiState(
            mean=init_params,
            low_rank=low_rank,
            diagonal=diagonal,
        )
    
    def _predict(self, state):
        I_lr = jnp.eye(self.rank)
        mean_pred = state.mean
        diag_pred = 1 / (1 / state.diagonal + self.dynamics_covariance)

        C = jnp.einsum("ji,j,j,jk->ik",
            state.low_rank, diag_pred, 1 / state.diagonal, state.low_rank
        )
        C = jnp.linalg.inv(I_lr + self.dynamics_covariance * C)
        cholC = jnp.linalg.cholesky(C)

        low_rank_pred = jnp.einsum(
            "i,i,ij,jk->ik",
            diag_pred, 1 / state.diagonal, state.low_rank, cholC
        )

        state_pred = state.replace(
            mean=mean_pred,
            diagonal=diag_pred,
            low_rank=low_rank_pred,
        )
        return state_pred

    
    def _update_dlr(self, low_rank_hat):
        singular_vectors, singular_values, _ = jnp.linalg.svd(low_rank_hat, full_matrices=False)

        singular_vectors_drop = singular_vectors[:, self.rank:] # Ut
        singular_values_drop = singular_values[self.rank:] # Λt

        # Update new low rank
        singular_vectors = singular_vectors[:, :self.rank] # Ut
        singular_values = singular_values[:self.rank] # Λt
        low_rank_new = jnp.einsum("Dd,d->Dd", singular_vectors, singular_values)

        # Obtain additive term for diagonal
        lr_drop = jnp.einsum("Dd,d->Dd", singular_vectors_drop, singular_values_drop)
        diag_drop = jnp.einsum("ij,ij->i", lr_drop, lr_drop)

        return low_rank_new, diag_drop
                

    # @partial(jax.jit, static_argnums=(0,))
    def _update(self, bel_pred, x, y):
        eta = self.link_fn(bel_pred.mean, x).astype(float)
        yhat = self.mean(eta)
        yobs = self.suff_statistic(y)
        Rt = jnp.atleast_2d(self.covariance(eta))
        Ht = self.grad_link_fn(bel_pred.mean, x)

        At = jnp.linalg.inv(jnp.linalg.cholesky(Rt))
        memory_entry = Ht.T @ At.T
        _, n_out = memory_entry.shape
        low_rank_hat = jnp.c_[bel_pred.low_rank, memory_entry]
        Gt = jnp.linalg.pinv(
            jnp.eye(self.rank + n_out) + 
            jnp.einsum("ji,j,jk->ik", low_rank_hat, bel_pred.diagonal, low_rank_hat)
        )
        Ct = Ht.T @ At.T @ At

        # Kalman gain
        K1 = jnp.einsum("i,ij->ij", 1 / bel_pred.diagonal, Ct)
        K2 = jnp.einsum(
            "i,ij,jk,lk,lm->im",
            1 / bel_pred.diagonal ** 2, low_rank_hat, Gt,
            low_rank_hat, Ct
        )
        Kt = K1 - K2
        
        mean_new = bel_pred.mean + Kt @ (yobs - yhat)
        low_rank_new, diag_drop = self._update_dlr(low_rank_hat)
        diag_new = bel_pred.diagonal + diag_drop

        bel_new = bel_pred.replace(
            mean=mean_new,
            diagonal=diag_new,
            low_rank=low_rank_new,
        )
        return bel_new


    def step(self, bel, xs, callback_fn):
        xt, yt = xs
        bel_pred = self._predict(bel)
        bel_update = self._update(bel_pred, xt, yt)

        output = callback_fn(bel_update, bel_pred, xt, yt)
        return bel_update, output


class BernoulliFilter(ExpfamFilter):
    def __init__(self, apply_fn, dynamics_covariance, rank):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat,
            dynamics_covariance, rank
        )

    @partial(jax.jit, static_argnums=(0,))
    def _log_partition(self, eta):
        return jnp.log1p(jnp.exp(eta)).sum()

    @partial(jax.jit, static_argnums=(0,))
    def _suff_stat(self, y):
        return y


class MultinomialFilter(ExpfamFilter):
    def __init__(self, apply_fn, dynamics_covariance, rank):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat,
            dynamics_covariance, rank
        )

    @partial(jax.jit, static_argnums=(0,))
    def _log_partition(self, eta):
        eta = jnp.append(eta, 0.0)
        return jax.nn.logsumexp(eta).sum()

    @partial(jax.jit, static_argnums=(0,))
    def _suff_stat(self, y):
        return y
