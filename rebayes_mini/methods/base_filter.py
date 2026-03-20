import jax
import chex
import distrax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from abc import ABC, abstractmethod
from jax.flatten_util import ravel_pytree
from rebayes_mini import callbacks


@chex.dataclass
class GaussState:
    mean: chex.Array
    cov: chex.Array

@chex.dataclass
class GaussStateSqr:
    mean: chex.Array
    W: chex.Array


@chex.dataclass
class GaussUVState:
    mean: chex.Array
    cov: chex.Array
    alpha: chex.Array  # InvGamma shape  (requires alpha > 1 for finite mean)
    beta: chex.Array   # InvGamma scale  (beta > 0)


class BaseFilter(ABC):
    def __init__(self, mean_fn, cov_fn):
        self.mean_fn = mean_fn
        self.cov_fn = cov_fn

    def _initialise_flat_fn(self, apply_fn, params):
        flat_params, rfn = ravel_pytree(params)

        @jax.jit
        def mean_fn(params, x):
            return apply_fn(rfn(params), x)

        return rfn, mean_fn, flat_params

    @abstractmethod
    def predict_fn(self, bel, x):
        ...

    @abstractmethod
    def init_bel(self):
        ...

    @abstractmethod
    def predict(self, bel):
        ...

    @abstractmethod
    def update(self, bel, y, x):
        ...

    @abstractmethod
    def sample_fn(self, key, bel):
        ...

    def step(self, bel, y, x, callback_fn):
        bel_pred = self.predict(bel)
        bel_update = self.update(bel_pred, y, x)

        output = callback_fn(bel_update, bel_pred, y, x)
        return bel_update, output

    def scan(self, bel, y, X, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        def _step(bel, yX):
            y, x = yX
            bel, out = self.step(bel, y, x, callback_fn)
            return bel, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist


class ExtendedFilter(BaseFilter):
    def __init__(self, mean_fn, cov_fn, dynamics_covariance, n_inner=1):
        """
        apply_fn: function
            Maps state and observation to the natural parameters
        """
        self.mean_fn_og = mean_fn
        self.cov_fn = cov_fn
        self.dynamics_covariance = dynamics_covariance
        self.n_inner = n_inner

    def init_bel(self, params, cov=1.0):
        self.rfn, self.mean_fn, init_params = self._initialise_flat_fn(self.mean_fn_og, params)
        self.grad_mean = jax.jacrev(self.mean_fn)

        nparams = len(init_params)
        return GaussState(
            mean=init_params,
            cov=jnp.eye(nparams) * cov,
        )


    def predictive_density(self, bel, X):
        mean = self.mean_fn(bel.mean, X).astype(float)
        # mean = self.mean(eta)
        Rt = jnp.atleast_2d(self.cov_fn(mean))
        Ht = self.grad_mean(bel.mean, X)
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

    def sample_params(self, key, bel, shape=None):
        shape = shape if shape is not None else (1,)
        L = jnp.linalg.cholesky(bel.cov)
        eps = jax.random.normal(key, (*shape, len(bel.mean)))
        params = jnp.einsum("ji,sj->si", L, eps) + bel.mean
        return params

    def sample_fn(self, key, bel):
        params = self.sample_params(key, bel).squeeze()
        def fn(x): return self.mean_fn(params, x).squeeze()
        return fn

    def predict(self, bel):
        nparams = len(bel.mean)
        I = jnp.eye(nparams)
        pmean_pred = bel.mean
        pcov_pred = bel.cov + self.dynamics_covariance * I
        bel = bel.replace(mean=pmean_pred, cov=pcov_pred)
        return bel

    def sample_predictive(self, key, bel, x):
        dist = self.predictive_density(bel, x)
        sample = dist.sample(seed=key)
        return sample
    
    def predict_fn(self, bel, x):
        return self.mean_fn(bel.mean, x).astype(float)

    def _update(self, bel, bel_pred, y, x):
        yhat = self.predict_fn(bel, x)
        Rt = jnp.atleast_2d(self.cov_fn(yhat))

        Ht = self.grad_mean(bel.mean, x)
        St = Ht @ bel_pred.cov @ Ht.T + Rt
        Kt = jnp.linalg.solve(St, Ht @ bel_pred.cov).T

        err = y - yhat - Ht @ (bel_pred.mean - bel.mean)

        mean_update = bel_pred.mean + Kt @ err
        I = jnp.eye(len(bel.mean))
        cov_update = (I - Kt @ Ht) @ bel_pred.cov @ (I - Kt @ Ht).T + Kt @ Rt @ Kt.T
        bel = bel.replace(mean=mean_update, cov=cov_update)
        return bel
    
    def update(self, bel, y, x):
        bel_pred = bel
        _update = lambda _, bel: self._update(bel, bel_pred, y, x)
        bel_update = jax.lax.fori_loop(0, self.n_inner, _update, bel_pred, unroll=self.n_inner)
        return bel_update

    def step(self, bel, y, x, callback_fn):
        bel_pred = self.predict(bel)
        bel_update = self.update(bel_pred, y, x)
        
        output = callback_fn(bel_update, bel_pred, y, x)
        return bel_update, output

    def scan(self, bel, y, X, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        def _step(bel, yX):
            y, x = yX
            bel, out = self.step(bel, y, x, callback_fn)
            return bel, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist


class ExtendedFilterUV(ExtendedFilter):
    """
    Iterated EKF with unknown scalar observation variance σ².

    Models σ² ~ InvGamma(alpha, beta), so E[σ²] = beta / (alpha - 1).
    The linearised observation covariance is R_t = sigma2 * I.
    After the inner linearisation iterations, alpha and beta receive a
    single conjugate update from the residual at the final mean estimate:

        alpha <- alpha + n_obs / 2
        beta  <- beta  + (err · err) / 2

    Parameters
    ----------
    alpha0 : float
        Initial InvGamma shape.  Must be > 1 so that E[σ²] is finite.
    beta0 : float
        Initial InvGamma scale (> 0).
    """

    def __init__(self, mean_fn, dynamics_covariance, alpha0=3.0, beta0=1.0, n_inner=1):
        super().__init__(mean_fn, lambda yhat: jnp.eye(jnp.atleast_1d(yhat).shape[0]), dynamics_covariance, n_inner)
        self.alpha0 = jnp.asarray(alpha0, dtype=float)
        self.beta0 = jnp.asarray(beta0, dtype=float)

    def _obs_cov(self, bel, yhat):
        """Observation covariance from E[sigma^2 | bel] under InvGamma(alpha, beta)."""
        sigma2 = bel.beta / (bel.alpha - 1)
        dim_obs = jnp.atleast_1d(yhat).shape[0]
        return sigma2 * jnp.eye(dim_obs)

    def init_bel(self, params, cov=1.0):
        self.rfn, self.mean_fn, init_params = self._initialise_flat_fn(self.mean_fn_og, params)
        self.grad_mean = jax.jacrev(self.mean_fn)
        nparams = len(init_params)
        return GaussUVState(
            mean=init_params,
            cov=jnp.eye(nparams) * cov,
            alpha=self.alpha0,
            beta=self.beta0,
        )

    def predict(self, bel):
        nparams = len(bel.mean)
        pcov_pred = bel.cov + self.dynamics_covariance * jnp.eye(nparams)
        return bel.replace(cov=pcov_pred)

    def predictive_density(self, bel, X):
        mean = self.mean_fn(bel.mean, X).astype(float)
        mean = jnp.atleast_1d(mean)
        Rt = self._obs_cov(bel, mean)
        Ht = self.grad_mean(bel.mean, X)
        covariance = Ht @ bel.cov @ Ht.T + Rt
        dist = distrax.MultivariateNormalFullCovariance(mean, covariance)
        return dist

    def _multivariate_student_t_logpdf(self, y, mean, scale, dof):
        """
        Log density for multivariate Student-t with location `mean`,
        scale matrix `scale`, and degrees of freedom `dof`.

        Formula:
        log p(y) = lgamma((nu + d)/2) - lgamma(nu/2)
                   - 0.5 * (d * log(nu * pi) + log|Sigma|)
                   - 0.5 * (nu + d) * log(1 + delta / nu)
        where delta = (y - mean)^T Sigma^{-1} (y - mean).
        """
        y = jnp.atleast_1d(y)
        mean = jnp.atleast_1d(mean)
        d = y.shape[0]
        err = y - mean

        sign, logdet = jnp.linalg.slogdet(scale)
        # Numerical guard; scale should be SPD, so sign is expected to be +1.
        logdet = jnp.where(sign > 0, logdet, jnp.inf)
        maha = err @ jnp.linalg.solve(scale, err)

        log_norm = (
            gammaln((dof + d) / 2)
            - gammaln(dof / 2)
            - 0.5 * (d * jnp.log(dof * jnp.pi) + logdet)
        )
        log_kernel = -0.5 * (dof + d) * jnp.log1p(maha / dof)
        return log_norm + log_kernel

    def log_posterior_predictive(self, y, X, bel):
        """
        Log posterior predictive after marginalizing unknown variance sigma^2.

        Assumptions and form
        --------------------
        Uses the Normal-Inverse-Gamma conjugate result:

            y | x, bel, sigma^2 ~ Normal(m(x), sigma^2 I)
            sigma^2 | bel ~ InvGamma(alpha, beta)

        which implies

            y | x, bel ~ StudentT_nu(loc=m(x), scale=(beta/alpha) I),
            nu = 2 * alpha.

        For consistency with the filter's linearized predictive uncertainty,
        we add the projected parameter uncertainty H P H^T to the Student-t
        scale matrix.

        References
        ----------
        - Murphy, K. P. (2007), Conjugate Bayesian analysis of the Gaussian.
        - Normal-inverse-gamma distribution, marginal is Student-t
          (Wikipedia: Normal-inverse-gamma distribution).
        """
        mean = self.mean_fn(bel.mean, X).astype(float)
        mean = jnp.atleast_1d(mean)
        Ht = self.grad_mean(bel.mean, X)

        dof = 2.0 * bel.alpha
        ig_scale = (bel.beta / bel.alpha) * jnp.eye(mean.shape[0])
        scale = Ht @ bel.cov @ Ht.T + ig_scale

        return self._multivariate_student_t_logpdf(y, mean, scale, dof)

    def log_predictive_density(self, y, X, bel):
        return self.log_posterior_predictive(y, X, bel)

    def _update_state(self, bel, bel_pred, y, x):
        """Update mean and cov using the current sigma2 estimate; alpha/beta unchanged."""
        yhat = self.predict_fn(bel, x)
        Rt = self._obs_cov(bel, yhat)

        Ht = self.grad_mean(bel.mean, x)
        St = Ht @ bel_pred.cov @ Ht.T + Rt
        Kt = jnp.linalg.solve(St, Ht @ bel_pred.cov).T

        err = y - yhat - Ht @ (bel_pred.mean - bel.mean)
        mean_update = bel_pred.mean + Kt @ err

        I = jnp.eye(len(bel.mean))
        cov_update = (I - Kt @ Ht) @ bel_pred.cov @ (I - Kt @ Ht).T + Kt @ Rt @ Kt.T
        return bel.replace(mean=mean_update, cov=cov_update)

    def _update_variance(self, bel, y, x):
        """Single conjugate InvGamma update from the residual at the current mean."""
        err = (y - self.predict_fn(bel, x)).ravel()
        n_obs = jnp.size(y)
        return bel.replace(
            alpha=bel.alpha + n_obs / 2,
            beta=bel.beta + jnp.dot(err, err) / 2,
        )

    def update(self, bel, y, x):
        bel_pred = bel
        _iter = lambda _, b: self._update_state(b, bel_pred, y, x)
        bel_update = jax.lax.fori_loop(0, self.n_inner, _iter, bel_pred, unroll=self.n_inner)
        bel_update = self._update_variance(bel_update, y, x)
        return bel_update
    
    def step(self, bel, y, x, callback_fn):
        bel_pred = self.predict(bel)
        bel_update = self.update(bel_pred, y, x)
        output = callback_fn(bel_update, bel_pred, y, x)
        return bel_update, output


class SquareRootFilter(BaseFilter):
    """
    Linearised Iterated Square root filter
    """
    def __init__(self, mean_fn, cov_fn, dynamics_covariance, n_inner=1):
        """
        apply_fn: function
            Maps state and observation to the natural parameters
        """
        self.mean_fn_og = mean_fn
        self.cov_fn = cov_fn
        self.dynamics_covariance = dynamics_covariance
        self.n_inner = n_inner

    def init_bel(self, params, cov=1.0):
        self.rfn, self.mean_fn, init_params = self._initialise_flat_fn(self.mean_fn_og, params)
        self.grad_mean = jax.jacrev(self.mean_fn)

        nparams = len(init_params)
        return GaussStateSqr(
            mean=init_params,
            W=jnp.linalg.cholesky(jnp.eye(nparams) * cov, upper=True),
        )
    
    def sample_params(self, key, bel, shape=None):
        shape = shape if shape is not None else (1,)
        dim = len(bel.mean)
        shape = (*shape, dim)
        eps = jax.random.normal(key, shape)
        params = jnp.einsum("ji,..j->...i", bel.W, eps) + bel.mean
        return params
    
    def sample_fn(self, key, bel):
        params = self.sample_params(key, bel).squeeze()
        def fn(x): return self.mean_fn(params, x).squeeze()
        return fn

    def add_sqrt(self, matrices):
        """
        Obtain an upper-triangular matrix C such that
        C^T C = A1 + A2 + ... + Ak
        for (A1, A2, ..., Ak) = matrices
        """
        C_half = jnp.vstack(matrices)
        C_half = jnp.linalg.qr(C_half, mode="r") # Squared-root of innovation
        return C_half
    
    def sample_params(self, key, bel, shape=None):
        shape = shape if shape is not None else (1,)
        dim_params = len(bel.mean)
        eps = jax.random.normal(key, (*shape, dim_params))
        sample_params = jnp.einsum("ji,sj->si", bel.W, eps) + bel.mean
        return sample_params

    def predict(self, bel):
        nparams = len(bel.mean)
        I = jnp.eye(nparams)
        pmean_pred = bel.mean
        Q_half = jnp.sqrt(self.dynamics_covariance) * I
        W_pred = self.add_sqrt([bel.W, Q_half])
        bel = bel.replace(mean=pmean_pred, W=W_pred)
        return bel

    def predict_fn(self, bel, x):
        return self.mean_fn(bel.mean, x).astype(float)

    def update(self, bel, bel_pred, y, x):
        yhat = self.predict_fn(bel, x)
        Rt = jnp.atleast_2d(self.cov_fn(yhat))
        R_half = jnp.linalg.cholesky(Rt)

        # Innovation update
        Ht = self.grad_mean(bel.mean, x)

        S_half = self.add_sqrt([bel_pred.W @ Ht.T, R_half])
        M = jnp.linalg.solve(S_half, jnp.linalg.solve(S_half.T, Ht))
        K_transposed = M @ bel_pred.W.T @ bel_pred.W

        err = y - yhat - Ht @ (bel_pred.mean - bel.mean)

        mean_update = bel_pred.mean + jnp.einsum("ij,i->j", K_transposed, err)
        W_update = self.add_sqrt([
            bel_pred.W - bel_pred.W @ Ht.T @ K_transposed, R_half @ K_transposed
        ])

        bel = bel.replace(mean=mean_update, W=W_update)
        return bel

    def step(self, bel, y, x, callback_fn):
        bel_pred = self.predict(bel)
        _update = lambda _, bel: self.update(bel, bel_pred, y, x)
        bel_update = jax.lax.fori_loop(0, self.n_inner, _update, bel_pred, unroll=self.n_inner)

        output = callback_fn(bel_update, bel_pred, y, x, self)
        return bel_update, output
