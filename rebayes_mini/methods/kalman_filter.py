import jax
import chex
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from rebayes_mini import callbacks
from functools import partial

@chex.dataclass
class KFState:
    """State of the Kalman Filter"""
    mean: chex.Array
    cov: chex.Array


class LinearFilter:
    def __init__(
        self, transition_matrix, transition_covariance, observation_covariance,
    ):
        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
    
    def init_bel(self, mean, cov=1.0):
        return KFState(
            mean=mean,
            cov=jnp.eye(len(mean)) * cov,
        )
    
    def step(self, bel, y, obs_matrix, callback_fn):
        # Predict step
        mean_pred = self.transition_matrix @ bel.mean
        cov_pred = self.transition_matrix @ bel.cov @ self.transition_matrix.T + self.transition_covariance

        # Update step
        err = y - obs_matrix @ mean_pred
        S = obs_matrix @ cov_pred @ obs_matrix.T + self.observation_covariance
        K = jnp.linalg.solve(S, obs_matrix @ cov_pred).T
        mean = mean_pred + K @ err
        cov = cov_pred - K @ S @ K.T

        bel_update = bel.replace(mean=mean, cov=cov)
        output = callback_fn(bel_update, bel, y, mean_pred)
        return bel_update, output
    

    def initialise_active_step(self, y, X, callback_fn):
        """
        Create a step function that makes either
        the observation matrix fixed or part of the 
        filtering process as a function of t
        """
        if X.shape[0] != y.shape[0]:
            def _step(bel, y):
                bel, output = self.step(bel, y, X, callback_fn)
                return bel, output
            xs = y
        else:
            def _step(bel, xs):
                x, y = xs
                bel, output = self.step(bel, y, x, callback_fn)
                return bel, output
            xs = (y, X)
        
        return _step, xs

    
    def scan(self, bel, y, X, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        _step, xs = self.initialise_active_step(y, X, callback_fn)
        bel, hist = jax.lax.scan(_step, bel, xs)
        return bel, hist



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
        self.rfn, self.link_fn, init_params = self._initialise_link_fn(self.apply_fn, params)
        self.grad_link_fn = jax.jacfwd(self.link_fn)

        nparams = len(init_params)
        return KFState(
            mean=init_params,
            cov=jnp.eye(nparams) * cov,
        )

    def _initialise_link_fn(self, apply_fn, params):
        flat_params, rfn = ravel_pytree(params)

        @jax.jit
        def link_fn(params, x):
            return apply_fn(rfn(params), x)

        return rfn, link_fn, flat_params

    @partial(jax.jit, static_argnums=(0,))
    def mean(self, eta):
        return jax.grad(self.log_partition)(eta)

    @partial(jax.jit, static_argnums=(0,))
    def covariance(self, eta):
        return jax.hessian(self.log_partition)(eta).squeeze()

    def step(self, bel, xs, callback_fn):
        xt, yt = xs
        pmean_pred = bel.mean
        nparams = len(pmean_pred)
        I = jnp.eye(nparams)
        pcov_pred = bel.cov + self.dynamics_covariance * I

        eta = self.link_fn(bel.mean, xt).astype(float)
        yhat = self.mean(eta)
        err = self.suff_statistic(yt) - yhat
        Rt = self.covariance(eta)

        Ht = self.grad_link_fn(pmean_pred, xt)
        Kt = jnp.linalg.solve(Ht @ pcov_pred @ Ht.T + Rt, Ht @ pcov_pred).T

        pcov = (I - Kt @ Ht) @ pcov_pred
        pmean = pmean_pred + (Kt @ err).squeeze()

        bel_new = bel.replace(mean=pmean, cov=pcov)
        output = callback_fn(bel_new, bel, xt, yt)
        return bel_new, output

    def scan(self, bel, y, X, callback=None):
        xs = (X, y)
        callback = callbacks.get_null if callback is None else callback
        _step = partial(self.step, callback_fn=callback)
        bels, hist = jax.lax.scan(_step, bel, xs)
        return bels, hist


class GaussianFilter(ExpfamFilter):
    def __init__(self, apply_fn, dynamics_covariance, variance=1.0):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat, dynamics_covariance
        )
        self.variance = variance

    @partial(jax.jit, static_argnums=(0,))
    def _log_partition(self, eta):
        return (eta ** 2 / 2).sum()

    @partial(jax.jit, static_argnums=(0,))
    def _suff_stat(self, y):
        return y / jnp.sqrt(self.variance)


class BernoulliFilter(ExpfamFilter):
    def __init__(self, apply_fn, dynamics_covariance):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat, dynamics_covariance
        )

    @partial(jax.jit, static_argnums=(0,))
    def _log_partition(self, eta):
        return jnp.log1p(jnp.exp(eta)).sum()

    @partial(jax.jit, static_argnums=(0,))
    def _suff_stat(self, y):
        return y


class MultinomialFilter(ExpfamFilter):
    def __init__(self, apply_fn, dynamics_covariance):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat, dynamics_covariance
        )

    @partial(jax.jit, static_argnums=(0,))
    def _log_partition(self, eta):
        eta = jnp.append(eta, 0.0)
        return jax.nn.logsumexp(eta).sum()

    @partial(jax.jit, static_argnums=(0,))
    def _suff_stat(self, y):
        return y


class HeteroskedasticGaussianFilter(ExpfamFilter):
    def __init__(self, apply_fn, dynamics_covariance):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat, dynamics_covariance
        )

    @partial(jax.jit, static_argnums=(0,))
    def _log_partition(self, eta):
        eta1, eta2 = eta
        return -eta1 ** 2 / (4 * eta2) - jnp.log(-2 * eta2) / 2

    @partial(jax.jit, static_argnums=(0,))
    def _suff_stat(self, y):
        return jnp.array([y, y ** 2])
