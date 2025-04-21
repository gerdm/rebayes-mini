import jax
import chex
import distrax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular
from rebayes_mini.methods.base_filter import BaseFilter

@chex.dataclass
class LLLRState:
    """State of the online last-layer low-rank inference machine"""
    mean_last: chex.Array
    loading_last: chex.Array
    mean_hidden: chex.Array
    loading_hidden: chex.Array


def orthogonal(key, n, m):
    """
    https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L2041-L2095
    """
    z = jax.random.normal(key, (max(n, m), min(n, m)))
    q, r = jnp.linalg.qr(z)
    d = jnp.linalg.diagonal(r)
    x = q * jnp.expand_dims(jnp.sign(d), -2)
    return x.T


class LowRankLastLayer(BaseFilter):
    def __init__(self, mean_fn, cov_fn, rank, dynamics_hidden, dynamics_last, rank_last=None):
        self.mean_fn_tree = mean_fn
        self.covariance = cov_fn
        self.rank = rank
        self.dynamics_hidden = dynamics_hidden
        self.dynamics_last = dynamics_last
        self.rank_last = rank_last if rank_last is not None else -1


    def _initialise_flat_fn(self, apply_fn, params):
        """
        Initialize ravelled function and gradients
        """
        last_layer_params = params["params"]["last_layer"]
        dim_last_layer_params = len(ravel_pytree(last_layer_params)[0])

        flat_params, rfn = ravel_pytree(params)
        flat_params_last = flat_params[-dim_last_layer_params:]
        flat_params_hidden = flat_params[:-dim_last_layer_params]

        @jax.jit
        def mean_fn(params_hidden, params_last, x):
            params = jnp.concat([params_hidden, params_last])
            return apply_fn(rfn(params), x)


        return rfn, mean_fn, flat_params_hidden, flat_params_last


    def _init_low_rank(self, key, nparams, cov, diag, rank):
        if diag:
            loading_hidden = cov * jnp.fill_diagonal(jnp.zeros((rank, nparams)), jnp.ones(nparams), inplace=False)
        else:
            print("Using QR decomposition to initialize low-rank matrix")
            key_Q, key_R = jax.random.split(key)
            A = jax.random.normal(key_Q, (rank, rank))
            Q, _ = jnp.linalg.qr(A)
            
            P = jax.random.normal(key_R, (rank, nparams))
            loading_hidden = Q @ P
            loading_hidden = loading_hidden / jnp.linalg.norm(loading_hidden, axis=-1, keepdims=True) * jnp.sqrt(cov)
            # loading_hidden = cov * orthogonal(key, rank, nparams)

        return loading_hidden


    def init_bel(
        self, params, cov_hidden=1.0, cov_last=1.0, low_rank_diag=True, low_rank_diag_last=True, key=314
    ):
        self.rfn, self.mean_fn, init_params_hidden, init_params_last = self._initialise_flat_fn(self.mean_fn_tree, params)
        self.jac_hidden = jax.jacrev(self.mean_fn, argnums=0)
        self.jac_last = jax.jacrev(self.mean_fn, argnums=1)
        nparams_hidden = len(init_params_hidden)
        nparams_last = len(init_params_last)

        key = jax.random.PRNGKey(key) if isinstance(key, int) else key
        key_hidden, key_last = jax.random.split(key)
        loading_hidden = self._init_low_rank(key_hidden, nparams_hidden, cov_hidden, low_rank_diag, self.rank)

        self.rank_last = loading_hidden.shape[0] if self.rank_last == -1 else self.rank_last
        loading_last = self._init_low_rank(key_last, nparams_last, cov_last, low_rank_diag_last, self.rank_last)
        # loading_last = cov_last * jnp.eye(nparams_last) # TODO: make it low rank as well?
        # loading_last  = orthogonal(key, nparams_last, nparams_last) * cov_last

        return LLLRState(
            mean_hidden=init_params_hidden,
            mean_last=init_params_last,
            loading_hidden=loading_hidden,
            loading_last=loading_last
        )


    def sample_params(self, key, bel, shape=None):
        shape = shape if shape is not None else (1,)
        n_params_last = len(bel.mean_last)
        shape = (*shape, n_params_last)
        eps = jax.random.normal(key, shape)
        params = jnp.einsum("ji,sj->si", bel.loading_last, eps) + bel.mean_last
        return params
    
    def sample_params_hidden(self, key, bel, shape=None):
        shape = shape if shape is not None else (1,)
        shape = (*shape, self.rank)
        eps = jax.random.normal(key, shape)
        params = jnp.einsum("ji,sj->si", bel.loading_hidden, eps) + bel.mean_hidden
        return params

    def sample_fn(self, key, bel):
        params = self.sample_params(key, bel).squeeze()
        # params_hidden = self.sample_params_hidden(key, bel).squeeze()
        params_hidden = bel.mean_hidden
        def fn(x): return self.mean_fn(params_hidden, params, x).squeeze()
        return fn


    def add_sqrt(self, matrices):
        """
        Obtain an upper-triangular matrix C such that
        C^T C = A1 + A2 + ... + Ak
        for (A1, A2, ..., Ak) = matrices
        """
        C_half = jnp.vstack(matrices)
        C_half = jnp.linalg.qr(C_half, mode="r") # Squared-root of innovation
        return C_half


    def add_project(self, matrices, rank, diag_dynamics=0.0):
        """
        Obtain rank-d matrix P such that
        P^T P approx A1 + A2 + ... + Ak
        """
        Z = jnp.vstack(matrices)
        ZZ = jnp.einsum("ij,kj->ik", Z, Z)
        singular_vectors, singular_values, _ = jnp.linalg.svd(ZZ, hermitian=True, full_matrices=False)
        singular_values = jnp.sqrt(singular_values + diag_dynamics) # square root of eigenvalues
        singular_values_inv = jnp.where(singular_values != 0.0, 1 / singular_values, 0.0)

        P = jnp.einsum("i,ji,jk->ik", singular_values_inv, singular_vectors, Z)
        P = jnp.einsum("d,dD->dD", singular_values[:rank], P[:rank])
        return P


    def predict(self, bel):
        return bel


    def predictive_density(self, bel, x):
        yhat = self.mean_fn(bel.mean_hidden, bel.mean_last, x).astype(float)
        Rt = jnp.atleast_2d(self.covariance(yhat))
        # Jacobian for hidden and last layer
        J_hidden = self.jac_hidden(bel.mean_hidden, bel.mean_last, x)
        J_last = self.jac_last(bel.mean_hidden, bel.mean_last, x)

        # Upper-triangular cholesky decomposition of the innovation
        # S_half = jnp.r_[bel.loading_hidden @ J_hidden.T, bel.loading_last @ J_last.T, R_half]
        C = jnp.r_[
            bel.loading_hidden @ J_hidden.T, jnp.sqrt(self.dynamics_hidden) * J_hidden.T,
            bel.loading_last @ J_last.T, jnp.sqrt(self.dynamics_last) * J_last.T,
        ]
        S = jnp.einsum("ji,jk->ik", C, C) + Rt
        # dist = distrax.Normal(loc=yhat, scale=S_half.T @ S_half)
        dist = distrax.MultivariateNormalFullCovariance(loc=yhat, covariance_matrix=S)
        return dist


    def sample_predictive(self, key, bel, x):
        dist = self.predictive_density(bel, x)
        sample = dist.sample(seed=key)
        return sample


    def innovation_and_gain(self, bel, y, x):
        yhat = self.mean_fn(bel.mean_hidden, bel.mean_last, x)
        R_half = jnp.linalg.cholesky(jnp.atleast_2d(self.covariance(yhat)), upper=True)
        # Jacobian for hidden and last layer
        J_hidden = self.jac_hidden(bel.mean_hidden, bel.mean_last, x)
        J_last = self.jac_last(bel.mean_hidden, bel.mean_last, x)

        # Innovation
        err = y - yhat

        # Upper-triangular cholesky decomposition of the innovation
        S_half = self.add_sqrt([
            bel.loading_hidden @ J_hidden.T, jnp.sqrt(self.dynamics_hidden) * J_hidden.T,
            bel.loading_last @ J_last.T, jnp.sqrt(self.dynamics_last) * J_last.T,
            R_half
        ])

        # Transposed gain matrices
        M_hidden = solve_triangular(S_half, solve_triangular(S_half.T, J_hidden, lower=True), lower=False)
        M_last = solve_triangular(S_half, solve_triangular(S_half.T, J_last, lower=True), lower=False)

        gain_hidden = M_hidden @ bel.loading_hidden.T @ bel.loading_hidden + M_hidden * self.dynamics_hidden
        gain_last = M_last @ bel.loading_last.T @ bel.loading_last + M_last * self.dynamics_last

        return err, gain_hidden, gain_last, J_hidden, J_last, R_half


    def _update_hidden(self, bel, J, gain, R_half, err):
        mean_hidden = bel.mean_hidden + jnp.einsum("ij,i->j", gain, err)
        loading_hidden = self.add_project([
            bel.loading_hidden - bel.loading_hidden @ J.T @ gain, R_half @ gain
        ], rank=self.rank, diag_dynamics=self.dynamics_hidden)
        return mean_hidden, loading_hidden


    def _update_last(self, bel, J, gain, R_half, err):
        dim_params = bel.mean_last.shape[0]
        mean_last = bel.mean_last + jnp.einsum("ij,i->j", gain, err)
        loading_last = self.add_project([
            bel.loading_last - bel.loading_last @ J.T @ gain, R_half @ gain
        ], rank=self.rank_last, diag_dynamics=self.dynamics_last)
        return mean_last, loading_last
    
    def predict_fn(self, bel, X):
        """
        Similar to self.mean_fn, but we pass the belief state (non-differentiable).
        This is useful for the case when we want to predict using different agents.
        """
        return self.mean_fn(bel.mean_hidden, bel.mean_last, X)


    def update(self, bel, y, x):
        err, gain_hidden, gain_last, J_hidden, J_last, R_half = self.innovation_and_gain(bel, y, x)

        mean_hidden, loading_hidden = self._update_hidden(bel, J_hidden, gain_hidden, R_half, err)
        mean_last, loading_last = self._update_last(bel, J_last, gain_last, R_half, err)

        bel = bel.replace(
            mean_hidden=mean_hidden,
            mean_last=mean_last,
            loading_hidden=loading_hidden,
            loading_last=loading_last,
        )
        return bel

