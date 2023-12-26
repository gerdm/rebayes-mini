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


class KalmanFilter:
    def __init__(
        self, transition_matrix, dynamics_covariance, observation_covariance,
    ):
        self.transition_matrix = transition_matrix
        self.dynamics_covariance = dynamics_covariance
        self.observation_covariance = observation_covariance

    def init_bel(self, mean, cov=1.0):
        return KFState(
            mean=mean,
            cov=jnp.eye(len(mean)) * cov,
        )

    def step(self, bel, y, obs_matrix, callback_fn):
        # Predict step
        mean_pred = self.transition_matrix @ bel.mean
        cov_pred = self.transition_matrix @ bel.cov @ self.transition_matrix.T + self.dynamics_covariance

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


class ExtendedKalmanFilter:
    def __init__(
        self, fn_latent, fn_obs, dynamics_covariance, observation_covariance,
    ):
        self.fn_latent = fn_latent
        self.fn_obs = fn_obs
        self.dynamics_covariance = dynamics_covariance
        self.observation_covariance = observation_covariance

    def _initalise_vector_fns(self, latent):
        vlatent, rfn = ravel_pytree(latent)

        @jax.jit # ht(z)
        def vobs_fn(latent, x):
            latent = rfn(latent)
            return self.fn_obs(latent, x)

        @jax.jit # ft(z, u)
        def vlatent_fn(latent):
            return self.fn_latent(latent)

        return rfn, vlatent_fn, vobs_fn, vlatent

    def _init_components(self, mean, cov):
        self.rfn, self.vlatent_fn, self.vobs_fn, vlatent = self._initalise_vector_fns(mean)
        self.jac_latent = jax.jacrev(self.vlatent_fn) # Ft
        self.jac_obs = jax.jacrev(self.vobs_fn) # Ht
        dim_latent = len(vlatent)

        cov = jnp.eye(dim_latent) * cov
        return vlatent, cov, dim_latent


    def init_bel(self, mean, cov=1.0):
        mean, cov, dim_latent = self._init_components(mean, cov)

        return KFState(
            mean=mean,
            cov=cov,
        )

    def _predict_step(self, bel):
        Ft = self.jac_latent(bel.mean)
        mean_pred = self.vlatent_fn(bel.mean)
        cov_pred = Ft @ bel.cov @ Ft.T + self.dynamics_covariance
        bel = bel.replace(mean=mean_pred, cov=cov_pred)
        return bel

    def _update_step(self, bel, y, x):
        Ht = self.jac_obs(bel.mean, x)
        Rt_inv = jnp.linalg.inv(self.observation_covariance)
        yhat = self.vobs_fn(bel.mean, x)
        prec_update = jnp.linalg.inv(bel.cov) + Ht.T @ Rt_inv @ Ht
        cov_update = jnp.linalg.inv(prec_update)
        Kt = cov_update @ Ht.T @ Rt_inv
        mean_update = bel.mean + Kt @ (y - yhat)

        bel = bel.replace(mean=mean_update, cov=cov_update)
        return bel

    def step(self, bel, xs, callback_fn):
        xt, yt = xs
        bel_pred = self._predict_step(bel)
        bel_update = self._update_step(bel_pred, yt, xt)

        output = callback_fn(bel_update, bel_pred, xt, yt)
        return bel_update, output

    def scan(self, bel, y, X, callback_fn=None):
        xs = (X, y)
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        _step = partial(self.step, callback_fn=callback_fn)
        bels, hist = jax.lax.scan(_step, bel, xs)
        return bels, hist


class ExpfamFilter:
    def __init__(self, apply_fn, log_partition, suff_statistic, dynamics_covariance):
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
        self.grad_link_fn = jax.jacrev(self.link_fn)

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
        Rt = jnp.atleast_2d(self.covariance(eta))

        Ht = Rt @ self.grad_link_fn(pmean_pred, xt)
        St = Ht @ pcov_pred @ Ht.T + Rt
        Kt = jnp.linalg.solve(St, Ht @ pcov_pred).T

        pcov = (I - Kt @ Ht) @ pcov_pred
        pmean = pmean_pred + (Kt @ err).squeeze()

        bel_new = bel.replace(mean=pmean, cov=pcov)
        output = callback_fn(bel_new, bel, xt, yt)
        return bel_new, output

    def scan(self, bel, y, X, callback_fn=None):
        xs = (X, y)
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        _step = partial(self.step, callback_fn=callback_fn)
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
        return jax.nn.logsumexp(eta).sum() * 2

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

