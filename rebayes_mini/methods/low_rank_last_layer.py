import jax
import chex
import jax.numpy as jnp
from functools import partial
from jax.flatten_util import ravel_pytree

@chex.dataclass
class OLLIState:
    """State of the online last-layer low-rank inference machine"""
    mean_last: chex.Array
    loading_last: chex.Array
    mean_hidden: chex.Array
    loading_hidden: chex.Array


class LowRankLastLayer:
    def __init__(self, apply_fn, rank, dynamics_hidden):
        self.apply_fn = apply_fn
        self.rank = rank
        self.dynamics_hidden = dynamics_hidden

    def _initialise_fn(self, apply_fn, params):
        """
        Initialize ravelled function and gradients
        """
        last_layer_params = params["params"]["last_layer"]
        dim_last_layer_params = len(ravel_pytree(last_layer_params)[0])

        flat_params, rfn = ravel_pytree(params)
        flat_params_last = flat_params[-dim_last_layer_params:]
        flat_params_hidden = flat_params[:-dim_last_layer_params]

        @jax.jit
        def link_fn(params_hidden, params_last, x):
            params = jnp.concat([params_hidden, params_last])
            return apply_fn(rfn(params), x)


        return rfn, link_fn, flat_params_hidden, flat_params_last

    def init_bel(self, params, cov_hidden=1.0, cov_last=1.0):
        self.rfn, self.mean_fn, init_params_hidden, init_params_last = self._initialise_fn(self.apply_fn, params)
        self.jac_hidden = jax.jacrev(self.mean_fn, argnums=0)
        self.jac_last = jax.jacrev(self.mean_fn, argnums=1)
        nparams_hidden = len(init_params_hidden)
        nparams_last = len(init_params_last)

        loading_hidden = cov_hidden * jnp.fill_diagonal(jnp.zeros((self.rank, nparams_hidden)), jnp.ones(nparams_hidden), inplace=False)
        loading_last = cov_last * jnp.eye(nparams_last) # TODO: make it low rank as well? 

        return OLLIState(
            mean_hidden=init_params_hidden,
            mean_last=init_params_last,
            loading_hidden=loading_hidden,
            loading_last=loading_last
        )

    def add_sqrt(self, *matrices):
        """
        Obtain an upper-triangular matrix C such that
        C^T C = A1 + A2 + ... + Ak
        for (A1, A2, ..., Ak) = matrices
        """
        C_half = jnp.vstack(matrices)
        C_half = jnp.linalg.qr(C_half, mode="r") # Squared-root of innovation
        return C_half

    def add_project(self, *matrices):
        """
        Obtain rank-d matrix P such that
        P^T P approx A1 + A2 + ... + Ak
        """
        Z = jnp.vstack(matrices)
        ZZ = jnp.einsum("ij,kj->ik", Z, Z)
        singular_vectors, singular_values, _ = jnp.linalg.svd(ZZ, hermitian=True, full_matrices=False)
        singular_values = jnp.sqrt(singular_values) # square root of eigenvalues

        P = jnp.einsum("i,ji,jk->ik", 1 / singular_values, singular_vectors, Z)
        P = jnp.einsum("d,dD->dD", singular_values[:self.rank], P[:self.rank])
        return P
    

    def predict(self, bel):
        ...
    
    def innovation_and_gain(self, bel, y, x):
        yhat = self.mean_fn(bel.mean_hidden, bel.mean_last, x)
        # TODO: *** Define self.covariance ***
        R_half = jnp.linalg.cholesky(jnp.atleast_2d(self.covariance(yhat)), upper=True)
        # Jacobian for hidden and last layer
        J_hidden = self.jac_hidden(bel.mean_hidden, bel.mean_last, x)
        J_last = self.jac_last(bel.mean_hidden, bel.mean_last, x)

        # Innovation
        err = y - yhat

        # Upper-triangular cholesky decomposition of the innovation
        S_half = self.add_sqrt(bel.loading_hidden @ J_hidden.T, bel.loading_last @ J_last.T, R_half)

        # Transposed gain matrices
        M_hidden = jnp.linalg.solve(S_half, jnp.linalg.solve(S_half.T, J_hidden))
        M_last = jnp.linalg.solve(S_half, jnp.linalg.solve(S_half.T, J_last))

        gain_hidden = M_hidden @ bel.loading_hidden.T @ bel.loading_hiden + M_hidden * self.dynamics_hidden
        gain_last = M_last @ bel.loading_last.T @ bel.loading_last

        return err, gain_hidden, gain_last, J_hidden, J_last, R_half
    
    def update(self, bel, y, x):
        err, gain_hidden, gain_last, J_hidden, J_last, R_half = self.innovation_and_gain(bel, y, x)

