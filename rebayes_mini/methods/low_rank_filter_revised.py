import jax
import chex
import jax.numpy as jnp
from functools import partial
from rebayes_mini.methods.base_filter import BaseFilter
from rebayes_mini.methods import gauss_filter as kf

@chex.dataclass
class LowRankState:
    """State of the Low Rank Filter"""
    mean: chex.Array
    low_rank: chex.Array

def orthogonal(key, n, m):
    """
    https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L2041-L2095
    """
    z = jax.random.normal(key, (max(n, m), min(n, m)))
    q, r = jnp.linalg.qr(z)
    d = jnp.linalg.diagonal(r)
    x = q * jnp.expand_dims(jnp.sign(d), -2)
    return x.T


class LowRankCovarianceFilter(BaseFilter):
    def __init__(
        self, mean_fn, cov_fn, dynamics_covariance, rank
    ):
        self.mean_fn_tree = mean_fn
        self.cov_fn = cov_fn
        self.dynamics_covariance = dynamics_covariance
        self.rank = rank

    def _init_low_rank(self, key, nparams, cov, diag):
        if diag:
            loading_hidden = cov * jnp.fill_diagonal(jnp.zeros((self.rank, nparams)), jnp.ones(nparams), inplace=False)
        else:
            loading_hidden = cov * orthogonal(key, self.rank, nparams)

        return loading_hidden

    def sample_params(self, key, bel, shape=None):
        """
        TODO: Double check!!
        """
        dim_full = len(bel.mean)
        shape = shape if shape is not None else (1,)
        shape_sub = (*shape, self.rank)
        eps = jax.random.normal(key, shape_sub)

        # params = jnp.einsum("ji,sj->si", bel.low_rank, eps) + eps_full * jnp.sqrt(self.dynamics_covariance) + bel.mean
        params = jnp.einsum("ji,sj->si", bel.low_rank, eps) + bel.mean
        return params

    def sample_fn(self, key, bel):
        params = self.sample_params(key, bel).squeeze()
        def fn(x): return self.mean_fn(params, x).squeeze()
        return fn

    def init_bel(self, params, cov=1.0, low_rank_diag=True, key=314):
        key = jax.random.PRNGKey(key) if isinstance(key, int) else key
        self.rfn, self.mean_fn, init_params = self._initialise_flat_fn(self.mean_fn_tree, params)
        self.grad_mean_fn = jax.jacrev(self.mean_fn)
        nparams = len(init_params)
        low_rank = self._init_low_rank(key, nparams, cov, low_rank_diag)

        return LowRankState(
            mean=init_params,
            low_rank=low_rank,
        )

    def project(self, *matrices):
        """
        Create rank-d matrix P such that
        P^T P approx A + B
        """
        Z = jnp.vstack(matrices)
        ZZ = jnp.einsum("ij,kj->ik", Z, Z)
        singular_vectors, singular_values, _ = jnp.linalg.svd(ZZ, hermitian=True, full_matrices=False)
        singular_values = jnp.sqrt(singular_values) # square root of eigenvalues

        P = jnp.einsum("i,ji,jk->ik", 1 / singular_values, singular_vectors, Z)
        P = jnp.einsum("d,dD->dD", singular_values[:self.rank], P[:self.rank])
        return P

    def predict(self, bel):
        mean_pred = bel.mean
        low_rank_pred = bel.low_rank

        bel_pred = bel.replace(
            mean=mean_pred,
            low_rank=low_rank_pred,
        )
        return bel_pred
    
    def _innovation_and_gain(self, bel, y, x):
        yhat = self.mean_fn(bel.mean, x).astype(float)
        Rt_half = jnp.linalg.cholesky(jnp.atleast_2d(self.cov_fn(yhat)), upper=True)
        Ht = self.grad_mean_fn(bel.mean, x)
        W = bel.low_rank

        C = jnp.r_[W @ Ht.T, jnp.sqrt(self.dynamics_covariance) * Ht.T, Rt_half]
        # C = jnp.r_[W @ Ht.T, Rt_half]
        S_half = jnp.linalg.qr(C, mode="r") # Squared-root of innovation

        # transposed Kalman gain and innovation
        Mt = jnp.linalg.solve(S_half, jnp.linalg.solve(S_half.T, Ht))
        Kt_T = Mt @ W.T @ W + Mt * self.dynamics_covariance
        err = y - yhat
        return Kt_T, err, Rt_half, Ht
    
    def update(self, bel, y, x):
        Kt_T, err, Rt_half, Ht = self._innovation_and_gain(bel, y, x)
        mean_update = bel.mean + jnp.einsum("ij,i->j", Kt_T, err)
        low_rank_update = self.project(
            bel.low_rank - bel.low_rank @ Ht.T @ Kt_T, Rt_half @ Kt_T
        )

        bel = bel.replace(
            mean=mean_update,
            low_rank=low_rank_update
        )
        return bel
