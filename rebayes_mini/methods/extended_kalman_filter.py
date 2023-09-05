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


class ExpfamExtendedKalmanFilter:
    def __init__(
        self, link_fn, log_partition, suff_statistic, dynamics_covariance,
    ):
        self.link_fn = link_fn
        self.log_partition = log_partition
        self.suff_statistic = suff_statistic
        self.dynamics_covariance = dynamics_covariance

    def init_bel(self, params, cov=1.0):
        self.rfn, self.apply_fn = self._initialise_link_fn(self.link_fn, params)
        self.grad_apply_fn = jax.jacfwd(self.apply_fn)

        flat_params, _ = ravel_pytree(params)
        nparams = len(flat_params)

        return EKFBel(
            mean=flat_params,
            cov=jnp.eye(nparams) * cov,
        )

    def _initialise_link_fn(self, link_fn, params):
        _, rfn = ravel_pytree(params)

        @jax.jit
        def apply_fn(params, x):
            return link_fn(rfn(params), x)

        return rfn, apply_fn

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
        
        eta = self.apply_fn(bel.mean, xt).astype(float)
        yhat = self.mean(eta)
        err = self.suff_statistic(yt) - yhat
        Rt = self.covariance(eta)
        
        Ht = self.grad_apply_fn(pmean_pred, xt)
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
    