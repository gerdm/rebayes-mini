# R-VGA method
import jax
import chex
import jax.numpy as jnp
from abc import ABC, abstractmethod
from rebayes_mini import callbacks
from functools import partial
from jax.flatten_util import ravel_pytree

@chex.dataclass
class RVGAState:
    mean: chex.Array
    precision: chex.Array


class RVGA(ABC):
    """
    Recursive Variational Gaussian Approximation.

    Lambert, Marc, Silvère Bonnabel, and Francis Bach.
    "The recursive variational Gaussian approximation (R-VGA).
    Statistics and Computing 32.1 (2022): 10.
    """
    def __init__(
        self, apply_fn, n_inner, n_samples,
    ):
        self.apply_fn = apply_fn
        self.n_inner = n_inner
        self.n_samples = n_samples

    @abstractmethod
    def log_partition(self, eta):
        ...

    @abstractmethod
    def suff_statistic(self, y):
        ...

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
        log_proba = self.suff_statistic(y).T @ natural_parameters - self.log_partition(natural_parameters)
        return log_proba.squeeze()

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

    
    def step(self, bel, xs, callback_fn):
        key, x, y = xs
        keys = jax.random.split(key, self.n_inner)
        _inner = partial(self._step_inner, x=x, y=y)
        bel_update, _ = jax.lax.scan(_inner, bel, keys)

        output = callback_fn(bel_update, bel, y, x)
        return bel_update, output


    def scan(self, key, bel, y, X, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        n_timesteps = len(X)
        keys = jax.random.split(key, n_timesteps)
        D = (keys, X, y)
        _step = partial(self.step, callback_fn=callback_fn)
        bel, hist = jax.lax.scan(_step, bel, D)
        return bel, hist


class BernoulliRVGA(RVGA):
    def __init__(self, apply_fn, n_inner, n_samples):
        super().__init__(apply_fn, n_inner, n_samples)

    def log_partition(self, eta):
        return jnp.log1p(jnp.exp(eta)).sum()

    def suff_statistic(self, y):
        return y
