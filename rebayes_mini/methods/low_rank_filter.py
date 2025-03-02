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
    diagonal: chex.Array
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

    def _sample_lr_params(self, key, bel):
        """
        TODO(?): refactor code into jax.vmap. (It faster?
        Sample parameters from a low-rank variational Gaussian approximation.
        This implementation avoids the explicit construction of the
        (D x D) covariance matrix.

        We take s ~ N(0, W W^T + Psi I)

        Implementation based on §4.2.2 of the L-RVGA paper [1].

        [1] Lambert, Marc, Silvère Bonnabel, and Francis Bach.
        "The limited-memory recursive variational Gaussian approximation (L-RVGA).
         Statistics and Computing 33.3 (2023): 70.
        """
        key_x, key_eps = jax.random.split(key)
        dim_full, dim_latent = bel.low_rank.shape
        Psi_inv = 1 / bel.diagonal

        eps_sample = jax.random.normal(key_eps, (dim_latent,))
        x_sample = jax.random.normal(key_x, (dim_full,)) * jnp.sqrt(Psi_inv)

        I_full = jnp.eye(dim_full)
        I_latent = jnp.eye(dim_latent)
        # M = I + W^T Psi^{-1} W
        M = I_latent + jnp.einsum("ji,j,jk->ik", bel.low_rank, Psi_inv, bel.low_rank)
        # L = Psi^{-1} W^T M^{-1}
        L_tr = jnp.linalg.solve(M.T, jnp.einsum("i,ij->ji", Psi_inv, bel.low_rank))

        # samples = (I - LW^T)x + Le
        term1 = I_full - jnp.einsum("ji,kj->ik", L_tr, bel.low_rank)
        x_transform = jnp.einsum("ij,j->i", term1, x_sample)
        eps_transform = jnp.einsum("ji,j->i", L_tr, eps_sample)
        samples = x_transform + eps_transform
        return samples + bel.mean


    def log_predictive_density_exact(self, y, X, bel):
        """
        Equation (59) - (61)
        """
        eta = self.link_fn(bel.mean, X).astype(float)
        mean = self.mean(eta)
        Rt = jnp.atleast_2d(self.covariance(eta))

        Ht = self.grad_link_fn(bel.mean, X)

        diag_inverse = 1 / bel.diagonal
        C1 = jnp.einsum("ji,j,jk->ik", bel.low_rank, diag_inverse, bel.low_rank)
        C1 = jnp.linalg.inv(jnp.eye(self.rank) + C1)
        C2 = jnp.einsum("i,ij,jk,lk,l->il", diag_inverse, bel.low_rank, C1, bel.low_rank, diag_inverse)
        C3 = jnp.eye(len(bel.mean)) * diag_inverse  - C2
        covariance = jnp.einsum("ij,jk,lk->il", Ht, C3, Ht) + Rt

        log_p_pred = distrax.MultivariateNormalFullCovariance(mean, covariance).log_prob(y)
        return log_p_pred


    def predict(self, bel):
        I_lr = jnp.eye(self.rank)
        mean_pred = bel.mean
        diag_pred = 1 / (1 / bel.diagonal + self.dynamics_covariance)

        C = jnp.einsum("ji,j,jk->ik",
            bel.low_rank, (1 / bel.diagonal - diag_pred / bel.diagonal ** 2), bel.low_rank
        )
        C = jnp.linalg.inv(I_lr + C)
        cholC = jnp.linalg.cholesky(C)

        low_rank_pred = jnp.einsum(
            "i,i,ij,jk->ik",
            diag_pred, 1 / bel.diagonal, bel.low_rank, cholC
        )

        bel_pred = bel.replace(
            mean=mean_pred,
            diagonal=diag_pred,
            low_rank=low_rank_pred,
        )
        return bel_pred


    def _svd(self, W):
        """
        Fast implementation of reduced SVD

        See: https://math.stackexchange.com/questions/3685997/how-do-you-compute-the-reduced-svd
        """
        singular_vectors, singular_values, _ = jnp.linalg.svd(W.T @ W, full_matrices=False, hermitian=True)
        singular_values = jnp.sqrt(singular_values)
        singular_values_inv = jnp.where(singular_values != 0.0, 1 / singular_values, 0.0)
        singular_vectors = jnp.einsum("ij,jk,k->ik", W, singular_vectors, singular_values_inv)
        return singular_values, singular_vectors


    def _update_dlr(self, low_rank_hat):
        singular_values, singular_vectors = self._svd(low_rank_hat)

        singular_vectors_drop = singular_vectors[:, self.rank:] # Ut
        singular_values_drop = singular_values[self.rank:] # Λt

        # Update new low rank
        singular_vectors = singular_vectors[:, :self.rank] # Ut
        singular_values = singular_values[:self.rank] # Λt
        low_rank_new = jnp.einsum("Dd,d->Dd", singular_vectors, singular_values)

        # Obtain additive term for diagonal
        diag_drop = jnp.einsum(
            "ij,j,ij,j->i",
            singular_vectors_drop, singular_values_drop,
            singular_vectors_drop, singular_values_drop
        )

        return low_rank_new, diag_drop


    def update(self, bel_pred, y, x):
        eta = self.link_fn(bel_pred.mean, x).astype(float)
        yhat = self.mean(eta)
        yobs = self.suff_statistic(y)
        Rt = jnp.atleast_2d(self.covariance(eta))
        Ht = self.grad_link_fn(bel_pred.mean, x)

        At = jnp.linalg.inv(jnp.linalg.cholesky(Rt))
        memory_entry = Ht.T @ At.T
        _, n_out = memory_entry.shape

        low_rank_hat = jnp.c_[bel_pred.low_rank, memory_entry]
        inverse_diag = 1 / bel_pred.diagonal
        Gt = jnp.linalg.pinv(
            jnp.eye(self.rank + n_out) +
            jnp.einsum("ji,j,jk->ik", low_rank_hat, inverse_diag, low_rank_hat)
        )

        err = yobs - yhat

        # LoFi gain times innovation
        K1 = jnp.einsum(
            "i,ji,kj,kl,l->i",
            inverse_diag, Ht, At, At, err
        )
        K2 = jnp.einsum(
            "i,ij,jk,lk,l,ml,nm,no,o->i",
            inverse_diag, low_rank_hat, Gt,
            low_rank_hat, inverse_diag,
            Ht, At, At, err
        )
        Kt_err = K1 - K2

        mean_new = bel_pred.mean + Kt_err
        low_rank_new, diag_drop = self._update_dlr(low_rank_hat)
        diag_new = bel_pred.diagonal + diag_drop * self.inflate_diag

        bel_new = bel_pred.replace(
            mean=mean_new,
            low_rank=low_rank_new,
            diagonal=diag_new,
        )
        return bel_new


    def step(self, bel, y, x, callback_fn):
        bel_pred = self.predict(bel)
        bel_update = self.update(bel_pred, y, x)

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