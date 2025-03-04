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
    diagonal: chex.Array

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
        shape = shape if shape is not None else (1,)
        shape = (*shape, self.rank)
        eps = jax.random.normal(key, shape)
            # Compute QR decomposition of W
        Q, R = jnp.linalg.qr(bel.low_rank.T)  # QR decomposition of W^T
        
        # Compute covariance decomposition using QR trick
        A = R @ Q.T + jnp.sqrt(bel.diagonal)  # Approximate Cholesky-like factorization

        params = jnp.einsum("ji,sj->si", A, eps) + bel.mean
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

        diag = jnp.ones(nparams) * cov

        return LowRankState(
            mean=init_params,
            low_rank=low_rank,
            diagonal=diag
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
            diagonal=bel.diagonal + self.dynamics_covariance
        )
        return bel_pred
    
    def _innovation_and_gain(self, bel, y, x):
        yhat = self.mean_fn(bel.mean, x)
        Rt_half = jnp.linalg.cholesky(jnp.atleast_2d(self.cov_fn(yhat)), upper=True)
        Ht = self.grad_mean_fn(bel.mean, x)
        W = bel.low_rank

        C = jnp.r_[W @ Ht.T, jnp.sqrt(self.dynamics_covariance) * Ht.T, Rt_half]
        # C = jnp.r_[W @ Ht.T, Rt_half]
        S_half = jnp.linalg.qr(C, mode="r") # Squared-root of innovation

        # transposed Kalman gain and innovation
        Mt = jnp.linalg.solve(S_half, jnp.linalg.solve(S_half.T, Ht))
        Kt_T = Mt @ W.T @ W + Mt * bel.diagonal
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


class ExpfamFilter(kf.ExpfamFilter):
    def __init__(
        self, apply_fn, log_partition, suff_statistic, dynamics_covariance, rank
    ):
        """
        Moment-matched Low-rank Extended Kalman filter

        Parameters
        ----------
        apply_fn: function
            Conditional expectation for a measurement
        log_partition: function
            [to be deprecated]
        suff_statistic: function
            Sufficient statistic given an observation
        dynamics_covariance: float
            Additive dynamics covariance to correct for model misspecification
        rank: int
            Dimension of low-rank component
        """
        super().__init__(apply_fn, log_partition, suff_statistic, dynamics_covariance)
        self.rank = rank

    def init_bel(self, params, cov=1.0):
        self.rfn, self.link_fn, init_params = self._initialise_link_fn(self.apply_fn, params)
        self.grad_link_fn = jax.jacrev(self.link_fn)
        nparams = len(init_params)

        low_rank = jnp.fill_diagonal(jnp.zeros((self.rank, nparams)), jnp.ones(nparams), inplace=False)
        diag = jnp.ones(nparams) * cov

        return LowRankState(
            mean=init_params,
            low_rank=low_rank,
            diagonal=diag
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


    def predict(self, state):
        mean_pred = state.mean
        low_rank_pred = state.low_rank

        state_pred = state.replace(
            mean=mean_pred,
            low_rank=low_rank_pred,
            diagonal=state.diagonal + self.dynamics_covariance
        )

        return state_pred
    
    def _innovation_and_gain(self, state, y, x):
        eta = self.link_fn(state.mean, x).astype(float)
        yhat = self.mean(eta)
        yobs = self.suff_statistic(y)
        Rt_half = jnp.linalg.cholesky(jnp.atleast_2d(self.covariance(eta)), upper=True)
        Ht = self.grad_link_fn(state.mean, x)
        W = state.low_rank

        # C = jnp.r_[W @ Ht.T, jnp.sqrt(self.dynamics_covariance) * Ht.T, Rt_half]
        C = jnp.r_[W @ Ht.T, Rt_half]
        S_half = jnp.linalg.qr(C, mode="r") # Squared-root of innovation

        # transposed Kalman gain and innovation
        Mt = jnp.linalg.solve(S_half, jnp.linalg.solve(S_half.T, Ht))
        Kt_T = Mt @ W.T @ W + Mt * state.diagonal
        err = yobs - yhat
        return Kt_T, err, Rt_half, Ht
    
    def update(self, state, y, x):
        Kt_T, err, Rt_half, Ht = self._innovation_and_gain(state, y, x)
        mean_update = state.mean + jnp.einsum("ij,i->j", Kt_T, err)
        low_rank_update = self.project(
            state.low_rank - state.low_rank @ Ht.T @ Kt_T, Rt_half @ Kt_T
        )

        state = state.replace(
            mean=mean_update,
            low_rank=low_rank_update
        )
        return state

    def step(self, bel, y, x, callback_fn):
        bel_pred = self.predict(bel)
        bel_update = self.update(bel_pred, y, x) 
        output = callback_fn(bel_update, bel_pred, y, x)
        return bel_update, output


class BernoulliFilter(ExpfamFilter):
    def __init__(self, apply_fn, dynamics_covariance, rank, ):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat,
            dynamics_covariance, rank, 
        )

    @partial(jax.jit, static_argnums=(0,))
    def _log_partition(self, eta):
        return jnp.log1p(jnp.exp(eta)).sum()

    @partial(jax.jit, static_argnums=(0,))
    def _suff_stat(self, y):
        return y


class MultinomialFilter(ExpfamFilter):
    def __init__(self, apply_fn, dynamics_covariance, rank, eps=0.1, ):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat,
            dynamics_covariance, rank, 
        )
        self.eps = eps

    @partial(jax.jit, static_argnums=(0,))
    def _log_partition(self, eta):
        eta = jnp.append(eta, 0.0)
        Z =  jax.nn.logsumexp(eta)
        return Z.sum()

    @partial(jax.jit, static_argnums=(0,))
    def _suff_stat(self, y):
        return y

    def mean(self, eta):
        return jax.nn.softmax(eta)

    def covariance(self, eta):
        mean = self.mean(eta)
        return jnp.diag(mean) - jnp.outer(mean, mean) + jnp.eye(len(eta)) * self.eps


class GaussianFilter(ExpfamFilter):
    def __init__(self, apply_fn, dynamics_covariance, rank, variance=1.0, ):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat, dynamics_covariance, rank, 
        )
        self.variance = variance

    def mean(self, eta):
        return eta

    def covariance(self, eta):
        return self.variance * jnp.eye(1)

    def _suff_stat(self, y):
        return y

    def _log_partition(self, eta):
        return (eta ** 2 / 2).sum()