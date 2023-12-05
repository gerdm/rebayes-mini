import jax
import chex
import jax.numpy as jnp
from functools import partial
from jax.flatten_util import ravel_pytree
from rebayes_mini.methods import gauss_filter as kf
from rebayes_mini import callbacks

@chex.dataclass
class GBState:
    mean: chex.Array
    covariance: chex.Array
    weighting_term: float = 1.0

@chex.dataclass
class RGBState:
    mean: chex.Array
    covariance: chex.Array
    key: chex.Array = None
    weighting_term: float = 1.0


class WSMFilter:
    """
    Weighted score-matching filter
    """
    def __init__(
        self, apply_fn, suff_stat, log_base_measure, dynamics_covariance,
        weighting_function, transition_matrix=None,
    ):
        self.apply_fn = apply_fn
        self.suff_stat = suff_stat
        self.grad_sstat = jax.jacfwd(suff_stat)
        self.dynamics_covariance = dynamics_covariance
        self.weighting_function = weighting_function
        self.log_base_measure = log_base_measure
        self.grad_log_base_measure = jax.grad(log_base_measure)
        self.transition_matrix = transition_matrix
        

    def init_bel(self, params, cov=1.0):
        self.rfn, self.link_fn = self._initialise_link_fn(self.apply_fn, params)
        self.grad_link = jax.jacrev(self.link_fn)
        self.m = partial(self.weighting_function, linkfn=self.link_fn)

        flat_params, _ = ravel_pytree(params)
        nparams = len(flat_params)

        if self.transition_matrix is None:
            self.transition_matrix = jnp.eye(nparams)

        return GBState(
            mean=flat_params,
            covariance=jnp.eye(nparams) * cov,
        )


    def _initialise_link_fn(self, apply_fn, params):
        _, rfn = ravel_pytree(params)

        @jax.jit
        def link_fn(params, x):
            return apply_fn(rfn(params), x)

        return rfn, link_fn
        

    def C(self, y, m, x):
        # Output: (P,)
        correction = self.link_fn(m, x) - self.grad_link(m, x) @ m
        return self.grad_sstat(y).T @ correction  + self.grad_log_base_measure(y)


    def Lambda(self, y, mean, x):
        mval = self.m(y, mean, x)
        gradr = self.grad_sstat(y)
        gradl = self.grad_link(mean, x)
        
        out = jnp.einsum(
            "ji,jk,kl,ml,nm,no->io",
            gradl, gradr, mval, mval, gradr, gradl
        )
        return out


    def divterm(self, y, mean, x):
        mval = self.m(y, mean, x)
        gradr = self.grad_sstat(y)
        return jnp.einsum("ij,kj,lk->il", mval, mval, gradr)


    def nu(self, y, mean, x):
        mval = self.m(y, mean, x)
        gradr = self.grad_sstat(y)
        gradl = self.grad_link(mean, x)

        Cv = jnp.atleast_1d(self.C(y, mean, x))

        term1 = jnp.einsum(
            "ij,jk,lk,l->i",
            gradr, mval, mval, Cv
        )

        # TODO: clean this mess
        term2 = jax.jacrev(self.divterm)(y, mean, x).sum(axis=0)#sum(axis=-1)
        if len(jnp.atleast_1d(y)) > 1:
            term2 = term2.sum(axis=-1)

        return gradl.T @ (term1 + term2)
    
    def step(self, bel, D, learning_rate, callback_fn):
        y, x = D
        hat_cov = self.transition_matrix @ bel.covariance @ self.transition_matrix.T + self.dynamics_covariance
        hat_prec = jnp.linalg.inv(hat_cov)
        hat_mean = self.transition_matrix @ bel.mean
        
        Lambda = self.Lambda(y, hat_mean, x)
        nu = self.nu(y, hat_mean, x)
        
        prec = hat_prec + 2 * learning_rate * Lambda
        cov = jnp.linalg.inv(prec)
        mean = cov @ (hat_prec @ hat_mean - 2 * learning_rate * nu)
        
        bel_update = bel.replace(
            mean=mean,
            covariance=cov
        )
        output = callback_fn(bel_update, bel, y, x)
        return bel_update, output
    
    
    def scan(self, bel, y, x=None, learning_rate=1.0, callback_fn=None):
        D = (y, x)
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        _step = partial(self.step, learning_rate=learning_rate, callback_fn=callback_fn)
        bel, hist = jax.lax.scan(_step, bel, D)
        return bel, hist


class WeightedObsCovFilter(kf.KalmanFilter):
    """
    Ting, JA., Theodorou, E., Schaal, S. (2007).
    Learning an Outlier-Robust Kalman Filter.
    In: Kok, J.N., Koronacki, J., Mantaras, R.L.d., Matwin, S., Mladenič, D., Skowron, A. (eds)
    Machine Learning: ECML 2007. ECML 2007. Lecture Notes in Computer Science(),
    vol 4701. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-74958-5_76

    Special case for known SSM hyperparameters --- no M-step
    """
    def __init__(
        self, transition_matrix, dynamics_covariance, observation_covariance,
        prior_shape, prior_rate, n_inner=1, n_samples=10
    ):
        super().__init__(transition_matrix, dynamics_covariance, observation_covariance)
        # Prior gamma parameters per timestep.
        # We assume are fixed through time
        self.prior_shape = prior_shape # alpha term
        self.prior_rate = prior_rate # beta term
        self.n_inner = n_inner
        self.n_samples = n_samples
    
    def init_bel(self, mean, covariance, key=314):
        state = RGBState(
            mean=mean,
            covariance=jnp.eye(len(mean)) * covariance,
            weighting_term=1.0,
            key=jax.random.PRNGKey(key)
        )
        return state
    
    @partial(jax.vmap, in_axes=(None, 0, None, None, None))
    def _err_term(self, mean, y, x, R_inv):
        """
        Error correcting term for the posterior
        mean and covariance
        """
        err = y - x @ self.transition_matrix @ mean
        return err.T @ R_inv @ err

    def _update(self, i, bel, bel_prev, y, x):
        keyt = jax.random.fold_in(bel.key, i)
        mean_samples = jax.random.multivariate_normal(keyt, bel.mean, bel.covariance, (self.n_samples,))
        Rinv = jnp.linalg.inv(self.observation_covariance)
        mean_err = self._err_term(mean_samples, y, x, Rinv).mean()

        yhat = x @ self.transition_matrix @ bel_prev.mean
        weighting_term = (self.prior_shape + 1 / 2) / (self.prior_rate + mean_err / 2)
        pprec = jnp.linalg.inv(self.dynamics_covariance + bel_prev.covariance) + weighting_term * x.T @ Rinv @ x
        pcov = jnp.linalg.inv(pprec)
        pmean = self.transition_matrix @ bel_prev.mean + weighting_term * pcov @ x.T @ Rinv @ (y - yhat)

        bel = bel.replace(
            mean=pmean,
            covariance=pcov,
            weighting_term=weighting_term,
            key=keyt
        )
        return bel
    
    def step(self, bel, y, x, callback_fn):
        partial_update = partial(self._update, y=y, x=x, bel_prev=bel)
        bel_update = jax.lax.fori_loop(0, self.n_inner, partial_update, bel)
        output = callback_fn(bel_update, bel, y, x)

        return bel_update, output


class IMQFilter:
    """
    Inverse-Multi-Quadratic filter
    for a Gaussian state space model with 
    known observation covariance
    """
    def __init__(
        self, apply_fn, dynamics_covariance, observation_covariance, soft_threshold,
        transition_matrix=None,
        adaptive_dynamics=False,
    ):
        """
        apply_fn: function
            Maps state and observation to the natural parameters
        """
        self.apply_fn = apply_fn
        self.dynamics_covariance = dynamics_covariance
        self.observation_covariance = observation_covariance
        self.soft_threshold = soft_threshold
        self.transition_matrix = transition_matrix
        self.adaptive_dynamics = adaptive_dynamics

    def init_bel(self, params, cov=1.0):
        self.rfn, self.link_fn, init_params = self._initialise_link_fn(self.apply_fn, params)
        self.grad_link_fn = jax.jacrev(self.link_fn)

        nparams = len(init_params)
        if self.transition_matrix is None:
            self.transition_matrix = jnp.eye(nparams)

        return GBState(
            mean=init_params,
            covariance=jnp.eye(nparams) * cov,
        )

    def _initialise_link_fn(self, apply_fn, params):
        flat_params, rfn = ravel_pytree(params)

        @jax.jit
        def link_fn(params, x):
            return apply_fn(rfn(params), x)

        return rfn, link_fn, flat_params

    def step(self, bel, xs, callback_fn):
        xt, yt = xs
        dynamics_covariance = jax.lax.cond(
            self.adaptive_dynamics,
            lambda: self.dynamics_covariance * (1 - bel.weighting_term) ** 2,
            lambda: self.dynamics_covariance,
        )
        pmean_pred = self.transition_matrix @ bel.mean
        pcov_pred = self.transition_matrix @ bel.covariance @ self.transition_matrix.T + dynamics_covariance

        yhat = self.link_fn(pmean_pred, xt)
        err = yt - yhat
        Rt = jnp.atleast_2d(self.observation_covariance)
        weighting_term = self.soft_threshold ** 2 / (self.soft_threshold ** 2 + jnp.inner(err, err))

        Rt_inv = jnp.linalg.inv(Rt)
        Ht = self.grad_link_fn(pmean_pred, xt)
        pprec = jnp.linalg.inv(pcov_pred) + weighting_term * Ht.T @ Rt_inv @ Ht
        pcov = jnp.linalg.inv(pprec)
        pmean = pmean_pred + weighting_term * pcov @ Ht.T @ Rt_inv @ err

        bel_new = bel.replace(mean=pmean, covariance=pcov, weighting_term=weighting_term)
        output = callback_fn(bel_new, bel, xt, yt)
        return bel_new, output

    def scan(self, bel, y, X, callback_fn=None):
        xs = (X, y)
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        _step = partial(self.step, callback_fn=callback_fn)
        bels, hist = jax.lax.scan(_step, bel, xs)
        return bels, hist
