import jax
import chex
import distrax
import jax.numpy as jnp
from functools import partial
from rebayes_mini.methods import gauss_filter as kf

@chex.dataclass
class LoFiState:
    """State of the Low Rank Filter"""
    mean: chex.Array
    low_rank: chex.Array


class ExpfamFilter(kf.ExpfamFilter):
    def __init__(
        self, apply_fn, log_partition, suff_statistic, dynamics_covariance,
        rank, inflate_diag,
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
        inflate_diag: bool
            Inflate diagonal term based on unaccounted singular components
        """
        super().__init__(apply_fn, log_partition, suff_statistic, dynamics_covariance)
        self.rank = rank
        self.inflate_diag = inflate_diag

    def init_bel(self, key, params, cov=1.0):
        self.rfn, self.link_fn, init_params = self._initialise_link_fn(self.apply_fn, params)
        self.grad_link_fn = jax.jacrev(self.link_fn)
        nparams = len(init_params)

        low_rank = jnp.fill_diagonal(jnp.zeros((self.rank, nparams)), jnp.ones(nparams), inplace=False) * cov
        # low_rank = jax.random.normal(key, (self.rank, nparams))
        # low_rank = jnp.fill_diagonal(low_rank, jnp.ones(nparams) * cov, inplace=False)
        # low_rank = jnp.ones((self.rank, nparams))

        return LoFiState(
            mean=init_params,
            low_rank=low_rank,
        )

    def project(self, A, B):
        """
        Create rank-d matrix P such that
        P^T P approx A + B
        """
        Z = jnp.r_[A, B]
        singular_vectors, singular_values, _ = jnp.linalg.svd(Z @ Z.T, hermitian=True, full_matrices=False)
        singular_values = jnp.sqrt(singular_values + self.dynamics_covariance) # square root of eigenvalues

        P = jnp.einsum("i,ji,jk->ik", 1 / singular_values, singular_vectors, Z)
        P = jnp.einsum("d,dD->dD", singular_values[:self.rank], P[:self.rank])
        return P


    def _predict(self, state):
        mean_pred = state.mean
        low_rank_pred = state.low_rank

        # L = jnp.eye(len(mean_pred)) * self.dynamics_covariance
        # L = L[:self.rank]
        # low_rank_pred = self.project(low_rank_pred, L)

        # Z = state.low_rank
        # singular_vectors, singular_values, _ = jnp.linalg.svd(Z @ Z.T, hermitian=True, full_matrices=False)
        # singular_values = jnp.sqrt(singular_values) # square root of eigenvalues

        # P = jnp.einsum("i,ji,jk->ik", 1 / singular_values, singular_vectors, Z)
        # P = jnp.einsum("d,dD->dD", singular_values[:self.rank], P[:self.rank])


        state_pred = state.replace(
            mean=mean_pred,
            low_rank=low_rank_pred,
        )

        return state_pred
    
    def _innovation_and_gain(self, state, y, x):
        eta = self.link_fn(state.mean, x).astype(float)
        yhat = self.mean(eta)
        yobs = self.suff_statistic(y)
        Rt_half = jnp.linalg.cholesky(jnp.atleast_2d(self.covariance(eta)), upper=True)
        Ht = self.grad_link_fn(state.mean, x)
        W = state.low_rank

        C = jnp.r_[W @ Ht.T, Rt_half]
        S_half = jnp.linalg.qr(C, mode="r") # Squared-root of innovation

        # Kalman gain and innovation
        Kt = jnp.linalg.solve(S_half, jnp.linalg.solve(S_half.T, Ht))
        Kt = (Kt @ W.T @ W + Kt * self.dynamics_covariance).T
        err = yobs - yhat
        return Kt, err, Rt_half, Ht
    
    def _update(self, state, Kt, err, Rt_half, Ht):
        I = jnp.eye(len(Kt))

        mean_update = state.mean + Kt @ err
        low_rank_update = self.project(state.low_rank @ (I - Kt @ Ht).T, Rt_half @ Kt.T)

        state = state.replace(
            mean=mean_update,
            low_rank=low_rank_update
        )
        return state

    def step(self, bel, y, x, callback_fn):
        bel_pred = self._predict(bel)
        Kt, err, Rt_half, Ht = self._innovation_and_gain(bel_pred, y, x)
        bel_update = self._update(bel_pred, Kt, err, Rt_half, Ht)

        output = callback_fn(bel_update, bel_pred, y, x)
        return bel_update, output


class BernoulliFilter(ExpfamFilter):
    def __init__(self, apply_fn, dynamics_covariance, rank, inflate_diag=True):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat,
            dynamics_covariance, rank, inflate_diag
        )

    @partial(jax.jit, static_argnums=(0,))
    def _log_partition(self, eta):
        return jnp.log1p(jnp.exp(eta)).sum()

    @partial(jax.jit, static_argnums=(0,))
    def _suff_stat(self, y):
        return y


class MultinomialFilter(ExpfamFilter):
    def __init__(self, apply_fn, dynamics_covariance, rank, eps=0.1, inflate_diag=True):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat,
            dynamics_covariance, rank, inflate_diag
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
    def __init__(self, apply_fn, dynamics_covariance, rank, variance=1.0, inflate_diag=True):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat, dynamics_covariance, rank, inflate_diag
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