import jax
import chex
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from functools import partial

@chex.dataclass
class EKFBel:
    """State of the EKF."""
    mean: chex.Array
    cov: chex.Array


class ExpfamFilter:
    def __init__(
        self, apply_fn, log_partition, suff_statistic, dynamics_covariance,
    ):
        """
        apply_fn: function
            Maps state and observation to the natural parameters
        """
        self.apply_fn = apply_fn
        self.log_partition = log_partition
        self.suff_statistic = suff_statistic
        self.dynamics_covariance = dynamics_covariance

    def init_bel(self, params, cov=1.0):
        self.rfn, self.link_fn = self._initialise_link_fn(self.apply_fn, params)
        self.grad_link_fn = jax.jacfwd(self.link_fn)

        flat_params, _ = ravel_pytree(params)
        nparams = len(flat_params)

        return EKFBel(
            mean=flat_params,
            cov=jnp.eye(nparams) * cov,
        )

    def _initialise_link_fn(self, apply_fn, params):
        _, rfn = ravel_pytree(params)

        @jax.jit
        def link_fn(params, x):
            return apply_fn(rfn(params), x)

        return rfn, link_fn

    @partial(jax.jit, static_argnums=(0,))
    def mean(self, eta):
        return jax.grad(self.log_partition)(eta)

    @partial(jax.jit, static_argnums=(0,))
    def covariance(self, eta):
        return jax.hessian(self.log_partition)(eta).squeeze()

    def step(self, bel, xs):
        xt, yt = xs
        pcov_pred = bel.cov + self.dynamics_covariance
        pmean_pred = bel.mean
        nparams = len(pmean_pred)
        I = jnp.eye(nparams)
        
        eta = self.link_fn(bel.mean, xt).astype(float)
        yhat = self.mean(eta)
        err = self.suff_statistic(yt) - yhat
        Rt = self.covariance(eta)
        
        Ht = self.grad_link_fn(pmean_pred, xt)
        Kt = jnp.linalg.solve(Ht @ pcov_pred @ Ht.T + Rt, Ht @ pcov_pred).T
        
        pcov = (I - Kt @ Ht) @ pcov_pred
        pmean = pmean_pred + (Kt @ err).squeeze()
        
        bel = bel.replace(mean=pmean, cov=pcov)
        return bel, bel.replace(cov=0.0) # Save memory
    
    def scan(self, bel, X, y):
        xs = (X, y)
        bels, hist = jax.lax.scan(self.step, bel, xs)
        return bels, hist


    def predict_obs(self):
        ... 
    
    def predict_bel(self):
        ...

    def update_bel(self):
        ...
    