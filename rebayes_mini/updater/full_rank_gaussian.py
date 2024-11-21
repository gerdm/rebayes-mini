import jax
import distrax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from jax.flatten_util import ravel_pytree
from rebayes_mini import callbacks
from rebayes_mini.states.gaussian import Gauss

class BaseLinearGaussian(ABC):
    def __init__(self, apply_fn):
        """
        apply_fn: function
            Maps state and observation to output parameters
        """
        self.apply_fn = apply_fn

    def init_bel(self, params, cov=1.0):
        self.rfn, self.predict_fn, init_params = self._initialise_predict_fn(self.apply_fn, params)
        self.grad_predict_fn = jax.jacrev(self.predict_fn)

        nparams = len(init_params)
        return Gauss(
            mean=init_params,
            cov=jnp.eye(nparams) * cov,
        )

    def _initialise_predict_fn(self, apply_fn, params):
        flat_params, rfn = ravel_pytree(params)

        @jax.jit
        def predict_fn(params, x):
            return apply_fn(rfn(params), x)

        return rfn, predict_fn, flat_params

    @abstractmethod
    def mean(self, eta):
        ...

    @abstractmethod
    def covariance(self, eta):
        ...

    def predictive_density(self, bel, X):
        eta = self.predict_fn(bel.mean, X).astype(float)
        mean = self.mean(eta)
        Rt = jnp.atleast_2d(self.covariance(eta))
        Ht = self.grad_predict_fn(bel.mean, X)
        covariance = Ht @ bel.cov @ Ht.T + Rt
        mean = jnp.atleast_1d(mean)
        dist = distrax.MultivariateNormalFullCovariance(mean, covariance)
        return dist

    def log_predictive_density(self, y, X, bel):
        """
        compute the log-posterior predictive density
        of the moment-matched Gaussian
        """
        log_p_pred = self.predictive_density(bel, X).log_prob(y)
        return log_p_pred


    def update(self, bel, y, x):
        eta = self.predict_fn(bel.mean, x).astype(float)
        yhat = self.mean(eta)
        err = y - yhat

        Rt = jnp.atleast_2d(self.covariance(eta))
        Ht = self.grad_predict_fn(bel.mean, x)
        St = Ht @ bel.cov @ Ht.T + Rt
        Kt = jnp.linalg.solve(St, Ht @ bel.cov).T

        mean_update = bel.mean + Kt @ err

        I = jnp.eye(len(bel.mean))
        cov_update = (I - Kt @ Ht) @ bel.cov @ (I - Kt @ Ht).T + Kt @ Rt @ Kt.T
        bel = bel.replace(mean=mean_update, cov=cov_update)
        return bel


    def step(self, bel, y, x, callback_fn):
        bel_update = self.update(bel, y, x)
        output = callback_fn(bel_update, bel, y, x, self)
        return bel_update, output


    def scan(self, bel, y, X, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        def _step(bel, yX):
            y, x = yX
            bel, out = self.step(bel, y, x, callback_fn)
            return bel, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist


class LinearGaussianFilter(BaseLinearGaussian):
    def __init__(self, apply_fn, variance=1.0):
        super().__init__(apply_fn)
        self.variance = variance

    def mean(self, eta):
        return eta

    def covariance(self, eta):
        return self.variance * jnp.eye(1)


class MultinomialFilter(BaseLinearGaussian):
    def __init__(self, apply_fn, eps=0.1):
        super().__init__(apply_fn)
        self.eps = eps

    def mean(self, eta):
        return jax.nn.softmax(eta)

    def covariance(self, eta):
        mean = self.mean(eta)
        return jnp.diag(mean) - jnp.outer(mean, mean) + jnp.eye(len(eta)) * self.eps
