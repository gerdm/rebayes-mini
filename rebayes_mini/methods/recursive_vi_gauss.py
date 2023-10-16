# R-VGA method
import jax
import chex
import jax.numpy as jnp
from functools import partial
from jax.flatten_util import ravel_pytree

@chex.dataclass
class RVGAState:
    mean: chex.Array
    precision: chex.Array


class RVGA:
    """
    Recursive Variational Gaussian Approximation
    for members of the exponential family
    """
    def __init__(
        self, apply_fn, log_partition, suff_statistic, n_inner=1, n_samples=10,
    ):
        self.apply_fn = apply_fn
        self.log_partition = log_partition
        self.suff_statistic = suff_statistic
        self.n_inner = n_inner
        self.n_samples = n_samples


    def init_bel(self, params, cov=1.0):
        self.rfn, self.link_fn = self._initialise_link_fn(self.apply_fn, params)
        self.grad_log_prob = jax.jacrev(self.log_probability, argnums=0)
        self.hessian_log_prob = jax.jacfwd(self.grad_log_prob, argnums=0)


        flat_params, _ = ravel_pytree(params)
        nparams = len(flat_params)

        return RVGAState(
            mean=flat_params,
            precision=jnp.eye(nparams) / cov,
        )

    def _initialise_link_fn(self, apply_fn, params):
        _, rfn = ravel_pytree(params)

        @jax.jit
        def link_fn(params, x):
            return apply_fn(rfn(params), x)

        return rfn, link_fn
   
    def log_probability(self, params, y, x):
        """
        Log probability up to a constant that does not
        depend on the paramters
        """
        natural_parameters = self.link_fn(params, x)
        lprob = self.suff_statistic(y).T @ natural_parameters - self.log_partition(natural_parameters)
        return lprob.squeeze()

    @partial(jax.jit, static_argnums=(0,))
    def mean(self, eta):
        return jax.grad(self.log_partition)(eta)

    @partial(jax.jit, static_argnums=(0,))
    def covariance(self, eta):
        return jax.hessian(self.log_partition)(eta).squeeze()
    
    def _step_inner(self, bel, key, x, y):
        bel_covariance = jnp.linalg.inv(bel.precision)
        params_sample = jax.random.multivariate_normal(key, bel.mean, bel_covariance, (self.n_samples,))
        mean_grad_logprob = jax.vmap(self.grad_log_prob, in_axes=(0, None, None))(params_sample, y, x).mean(axis=0)
        mean_hessian_logprob = jax.vmap(self.hessian_log_prob, in_axes=(0, None, None))(params_sample, y, x).mean(axis=0)

        mean_new = bel.mean + bel_covariance @ mean_grad_logprob
        prec_new = bel.precision - mean_hessian_logprob

        bel = bel.replace(
            mean=mean_new,
            precision=prec_new,
        )
        return bel, None

    
    def step(self, bel, xs):
        key, x, y = xs
        keys = jax.random.split(key, self.n_inner)
        _inner = partial(self._step_inner, x=x, y=y)
        bel, _ = jax.lax.scan(_inner, bel, keys)

        return bel, bel.mean


    def scan(self, key, bel, y, X):
        n_timesteps = len(X)
        keys = jax.random.split(key, n_timesteps)
        D = (keys, X, y)
        bel, hist = jax.lax.scan(self.step, bel, D)
        return bel, hist
