import jax
import distrax
import jax.numpy as jnp
from functools import partial
from rebayes_mini import callbacks
from rebayes_mini.states import GaussState
from jax.flatten_util import ravel_pytree


class KalmanFilter:
    def __init__(
        self, transition_matrix, dynamics_covariance, observation_covariance,
    ):
        self.transition_matrix = transition_matrix
        self.dynamics_covariance = dynamics_covariance
        self.observation_covariance = observation_covariance

    def init_bel(self, mean, cov=1.0):
        return GaussState(
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
                y, x = xs
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

        return GaussState(
            mean=mean,
            cov=cov,
        )

    def _predict(self, bel):
        Ft = self.jac_latent(bel.mean)
        mean_pred = self.vlatent_fn(bel.mean)
        cov_pred = Ft @ bel.cov @ Ft.T + self.dynamics_covariance
        bel = bel.replace(mean=mean_pred, cov=cov_pred)
        return bel

    def _update(self, bel, y, x):
        Ht = self.jac_obs(bel.mean, x)
        Rt = self.observation_covariance

        St = Ht @ bel.cov @ Ht.T + Rt
        Kt = jnp.linalg.solve(St, Ht @ bel.cov).T

        err = y - self.vobs_fn(bel.mean, x)
        mean_update = bel.mean + Kt @ err
        cov_update = bel.cov - Kt @ St @ Kt.T

        bel = bel.replace(mean=mean_update, cov=cov_update)
        return bel

    def step(self, bel, xs, callback_fn):
        xt, yt = xs
        bel_pred = self._predict(bel)
        bel_update = self._update(bel_pred, yt, xt)

        output = callback_fn(bel_update, bel_pred, yt, xt)
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
        return GaussState(
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


    @partial(jax.jit, static_argnums=(0,))
    def log_predictive_density(self, y, X, bel):
        """
        compute the log-posterior predictive density
        of the moment-matched Gaussian
        """
        eta = self.link_fn(bel.mean, X).astype(float)
        mean = self.mean(eta)
        Rt = jnp.atleast_2d(self.covariance(eta))
        # Ht = Rt @ self.grad_link_fn(bel.mean, X)
        Ht = self.grad_link_fn(bel.mean, X)
        covariance = Ht @ bel.cov @ Ht.T + Rt
        mean = jnp.atleast_1d(mean)
        log_p_pred = distrax.MultivariateNormalFullCovariance(mean, covariance).log_prob(y)
        return log_p_pred


    def _predict(self, bel):
        # TODO: add dynamics' function
        nparams = len(bel.mean)
        I = jnp.eye(nparams)
        pmean_pred = bel.mean
        pcov_pred = bel.cov + self.dynamics_covariance * I
        bel = bel.replace(mean=pmean_pred, cov=pcov_pred)
        return bel

    def _update(self, bel, y, x):
        eta = self.link_fn(bel.mean, x).astype(float)
        yhat = self.mean(eta)
        y = self.suff_statistic(y)
        err = y - yhat

        Rt = jnp.atleast_2d(self.covariance(eta))
        Ht = self.grad_link_fn(bel.mean, x)
        St = Ht @ bel.cov @ Ht.T + Rt
        Kt = jnp.linalg.solve(St, Ht @ bel.cov).T

        mean_update = bel.mean + Kt @ err
        cov_update = bel.cov - Kt @ St @ Kt.T
        bel = bel.replace(mean=mean_update, cov=cov_update)
        return bel

    def step(self, bel, xs, callback_fn):
        x, y = xs
        bel_pred = self._predict(bel)
        bel_update = self._update(bel_pred, y, x)

        output = callback_fn(bel_update, bel_pred, y, x)
        return bel_update, output

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
    
    def mean(self, eta):
        return eta
    
    def covariance(self, eta):
        return self.variance * jnp.eye(1)
    
    def _suff_stat(self, y):
        return y

    def _log_partition(self, eta):
        return (eta ** 2 / 2).sum()


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
    def __init__(self, apply_fn, dynamics_covariance, eps=0.1):
        super().__init__(
            apply_fn, self._log_partition, self._suff_stat, dynamics_covariance
        )
        self.eps = eps

    @partial(jax.jit, static_argnums=(0,))
    def _log_partition(self, eta):
        eta = jnp.append(eta, 0.0)
        return jax.nn.logsumexp(eta).sum()

    @partial(jax.jit, static_argnums=(0,))
    def _suff_stat(self, y):
        return y


    def mean(self, eta):
        return jax.nn.softmax(eta)

    def covariance(self, eta):
        mean = self.mean(eta)
        return jnp.diag(mean) - jnp.outer(mean, mean) + jnp.eye(len(eta)) * self.eps


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
