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


class IMQFilter:
    """
    Inverse-Multi-Quadratic filter
    for a Gaussian state space model with
    known observation covariance
    TODO: deprecate
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
            lambda: self.dynamics_covariance * -jnp.log(bel.weighting_term) / 2,
            # lambda: self.dynamics_covariance * 1 / bel.weighting_term,
            # lambda: self.dynamics_covariance * (1 - bel.weighting_term) ** 2,
            lambda: self.dynamics_covariance,
        )
        pmean_pred = self.transition_matrix @ bel.mean
        pcov_pred = self.transition_matrix @ bel.covariance @ self.transition_matrix.T + dynamics_covariance

        yhat = self.link_fn(pmean_pred, xt)
        err = yt - yhat
        Rt = jnp.atleast_2d(self.observation_covariance)
        weighting_term = self.soft_threshold ** 2 / (self.soft_threshold ** 2 + jnp.inner(err, err))

        Ht = self.grad_link_fn(pmean_pred, xt)
        St = Ht @ pcov_pred @ Ht.T + Rt / weighting_term
        Kt = jnp.linalg.solve(St, Ht @ pcov_pred).T

        mean_update = pmean_pred + Kt @ err
        cov_update = pcov_pred - Kt @ St @ Kt.T
        bel_new = bel.replace(
            mean=mean_update,
            covariance=cov_update,
            weighting_term=weighting_term,
        )

        output = callback_fn(bel_new, bel, yt, xt)
        return bel_new, output

    def scan(self, bel, y, X, callback_fn=None):
        xs = (X, y)
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        _step = partial(self.step, callback_fn=callback_fn)
        bels, hist = jax.lax.scan(_step, bel, xs)
        return bels, hist
