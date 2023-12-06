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



class RobustKalmanFilter(KalmanFilter):
    """
    See:
    G. Agamennoni, J. I. Nieto and E. M. Nebot,
    "Approximate Inference in State-Space Models With Heavy-Tailed Noise," 
    in IEEE Transactions on Signal Processing, vol. 60, no. 10, pp. 5024-5037, Oct. 2012,
    doi: 10.1109/TSP.2012.2208106.
    """
    def __init__(
        self, transition_matrix, dynamics_covariance, prior_observation_covariance, n_inner,
        noise_scaling
    ):
        super().__init__(transition_matrix, dynamics_covariance, prior_observation_covariance)
        self.n_inner = n_inner
        self.noise_scaling = noise_scaling

    def _predict(self, bel):
        mean_update = self.transition_matrix @ bel.mean
        cov_update = self.transition_matrix @ bel.cov @ self.transition_matrix.T + self.dynamics_covariance
        bel_predict = bel.replace(mean=mean_update, cov=cov_update)
        return bel_predict

    def _update(self, _, bel, bel_pred, x, y):
        I = jnp.eye(len(bel.mean))
        S = (y - x @ bel.mean) @ (y - x @ bel.mean).T + x @ bel.cov @ x.T
        Lambda = (self.noise_scaling * self.observation_covariance + S) / (self.noise_scaling + 1)

        Kt = jnp.linalg.solve(x @ bel_pred.cov @ x.T + Lambda, x @ bel_pred.cov)
        mean_new = bel_pred.mean + Kt.T @ (y - x @ bel_pred.mean)
        cov_new = Kt.T @ Lambda @ Kt + (I - x.T @ Kt).T @ bel_pred.cov @ (I - x.T @ Kt)

        bel = bel.replace(mean=mean_new, cov=cov_new)
        return bel

    def step(self, bel, y, x, callback_fn):
        bel_pred = self._predict(bel)
        partial_update = partial(self._update, bel_pred=bel_pred, x=x, y=y)
        bel_update = jax.lax.fori_loop(0, self.n_inner, partial_update, bel_pred)
        output = callback_fn(bel_update, bel_pred, y, x)

        return bel_update, output



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

