import jax
import jax.numpy as jnp
from typing import Callable
from rebayes_mini import callbacks
from rebayes_mini.states import PULSEGaussState
from jax.flatten_util import ravel_pytree


def find_key_value_and_path(d, target_key, path=None):
    if path is None:
        path = []

    for key, value in d.items():
        current_path = path + [key]

        if key == target_key:
            return value, current_path
        elif isinstance(value, dict):
            result, result_path = find_key_value_and_path(value, target_key, current_path)
            if result is not None:
                return result, result_path
            
    return None, None


def update_nested_dict(d, keys, new_value):
    if len(keys) == 1:
        d[keys[0]] = new_value
    else:
        key = keys[0]
        rest_of_keys = keys[1:]

        if key not in d:
            d[key] = {}

        update_nested_dict(d[key], rest_of_keys, new_value)


class SubspaceLastLayerFilter:
    """
    PULSE
    """
    def __init__(
        self,
        apply_fn_hidden: Callable,
        apply_fn_last: Callable,
        dynamics_covariance_hidden: float,
        dynamics_covariance_last: float,
    ):
        self.apply_fn_hidden = apply_fn_hidden # subcified model
        self.apply_fn_last = apply_fn_last # any function congruent with hidden space
        self.dynamics_covariance_hidden = dynamics_covariance_hidden
        self.dynamics_covariance_last = dynamics_covariance_last

    def _suff_stat(self, y):
        return y

    def init_bel(self, params_hidden, params_last, cov_hidden=1.0, cov_last=1.0):
        out = self._initialise_link_fn(
            self.apply_fn_hidden, self.apply_fn_last, params_hidden, params_last
        )
        self.rfn_hidden, self.rfn_last, self.link_fn, init_params_hidden, init_params_last = out

        self.grad_hidden = jax.jacrev(self.link_fn, argnums=0)
        self.grad_last = jax.jacrev(self.link_fn, argnums=1)

        nparams_hidden = len(init_params_hidden)
        nparams_last = len(init_params_last)
        return PULSEGaussState(
            mean_hidden=init_params_hidden,
            cov_hidden=jnp.eye(nparams_hidden) * cov_hidden,
            mean_last=init_params_last,
            cov_last=jnp.eye(nparams_last) * cov_last,
        )
   
    def _initialise_link_fn(self, apply_fn_hidden, apply_fn_last, params_hidden, params_last):
        """
        Initialize hidden and last-layer parameters
        """
        # Hidden-layer parameters
        params_fixed = params_hidden["fixed"]
        P, path = find_key_value_and_path(params_fixed, "P")
        P = P / jnp.linalg.norm(P, axis=0)
        update_nested_dict(params_fixed, path, P)
        params_train_hidden = params_hidden["params"]
        flat_params_hidden, rfn_hidden = ravel_pytree(params_train_hidden)

        # Last-layer parameters
        flat_params_last, rfn_last = ravel_pytree(params_last)

        @jax.jit
        def link_fn(params_hidden, params_last, x):
            p_train_recon = rfn_hidden(params_hidden)
            p_hidden = {
                "fixed": params_fixed,
                "params": p_train_recon
            }
            p_last = rfn_last(params_last)

            x = apply_fn_hidden(p_hidden, x)
            x = apply_fn_last(p_last, x)
            return x

        return rfn_hidden, rfn_last, link_fn, flat_params_hidden, flat_params_last

    def _predict(self, bel):
        # Hidden parameters
        nparams_hidden = len(bel.mean_hidden)
        I_hidden = jnp.eye(nparams_hidden)
        mean_hidden = bel.mean_hidden
        cov_hidden = bel.cov_hidden + self.dynamics_covariance_hidden * I_hidden

        # Last-layer parameters
        nparams_last = len(bel.mean_last)
        I_last = jnp.eye(nparams_last)
        mean_last = bel.mean_last
        cov_last = bel.cov_last + self.dynamics_covariance_last * I_last

        bel = bel.replace(
            mean_hidden=mean_hidden,
            cov_hidden=cov_hidden,
            mean_last=mean_last,
            cov_last=cov_last,
        )
        return bel

    def _update(self, bel, y, x):
        eta = self.link_fn(bel.mean_hidden, bel.mean_last, x).astype(float)
        yhat = self.mean(eta)
        y = self.suff_statistic(y)
        err = y - yhat
        I_obs = jnp.eye(len(y))

        Rt = jnp.atleast_2d(self.covariance(eta))
        Rt_inv = jnp.linalg.inv(Rt)

        Ht_hidden = self.grad_hidden(bel.mean_hidden, bel.mean_last, x)
        Ht_last = self.grad_last(bel.mean_hidden, bel.mean_last, x)
        
        # update hidden parameters
        I_hidden = jnp.eye(len(bel.mean_hidden))
        # St_hidden = Ht_hidden @ bel.cov_hidden @ Ht_hidden.T + Rt
        # Kt_hidden = jnp.linalg.solve(St_hidden, Ht_hidden @ bel.cov_hidden).T
        # cov_hidden = jnp.einsum("ij,jk,lk->il", Kt_hidden, St_hidden, Kt_hidden)
        # cov_hidden = (I_hidden - Kt_hidden @ Ht_hidden) @ bel.cov_hidden @ (I_hidden - Kt_hidden @ Ht_hidden).T + Kt_hidden @ Rt @ Kt_hidden.T

        prec_hidden = jnp.linalg.inv(bel.cov_hidden) + Ht_hidden.T @ Rt_inv @ Ht_hidden
        cov_hidden = jnp.linalg.inv(prec_hidden)

        # update last-layer parameters
        I_last = jnp.eye(len(bel.mean_last))
        # St_last = Ht_last @ bel.cov_last @ Ht_last.T + Rt
        # Kt_last = jnp.linalg.solve(St_last, Ht_last @ bel.cov_last).T
        # cov_last = jnp.einsum("ij,jk,lk->il", Kt_last, St_last, Kt_last)
        # cov_last = (I_last - Kt_last @ Ht_last) @ bel.cov_last @ (I_last - Kt_last @ Ht_last).T + Kt_last @ Rt @ Kt_last.T

        prec_last = jnp.linalg.inv(bel.cov_last) + Ht_last.T @ Rt_inv @ Ht_last
        cov_last = jnp.linalg.inv(prec_last)

        Kt_hidden, *_ = jnp.linalg.lstsq(prec_hidden, Ht_hidden.T @ Rt_inv)
        Kt_last, *_ = jnp.linalg.lstsq(prec_last, Ht_last.T @ Rt_inv)

        # Compute updated gain matrices
        Kt_hidden_prime = I_hidden - Kt_hidden @ Ht_last @ Kt_last @ Ht_hidden
        Kt_hidden_prime, *_ = jnp.linalg.lstsq(Kt_hidden_prime, Kt_hidden @ (I_obs  - Ht_last @ Kt_last) @ err)
        # Kt_hidden_prime = Kt_hidden @ (I_obs  - Ht_last @ Kt_last) @ err

        Kt_last_prime = I_last  - Kt_last @ Ht_hidden @ Kt_hidden @ Ht_last
        Kt_last_prime, *_ = jnp.linalg.lstsq(Kt_last_prime, Kt_last @ (I_obs  - Ht_hidden @ Kt_hidden) @ err)
        # Kt_last_prime = Kt_last @ (I_obs  - Ht_hidden @ Kt_hidden) @ err

        # Update joint mean
        # mean_hidden = bel.mean_hidden + Kt_hidden @ err
        # mean_last = bel.mean_last + Kt_last @ err

        mean_hidden = bel.mean_hidden + Kt_hidden_prime
        mean_last = bel.mean_last + Kt_last_prime


        bel = bel.replace(
            mean_hidden=mean_hidden,
            cov_hidden=cov_hidden,
            mean_last=mean_last,
            cov_last=cov_last,
        )
        return bel

    def step(self, bel, y, x, callback_fn):
        bel_pred = self._predict(bel)
        bel_update = self._update(bel_pred, y, x)

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


class MultinomialFilter(SubspaceLastLayerFilter):
    def __init__(self, apply_fn_hidden, apply_fn_last, dynamics_covariance_hidden, dynamics_covariance_last, eps=1e-7):
        super().__init__(
            apply_fn_hidden, apply_fn_last, dynamics_covariance_hidden, dynamics_covariance_last
        )
        self.eps = eps

    def suff_statistic(self, y):
        return y
    
    def mean(self, eta):
        return jax.nn.softmax(eta)

    def covariance(self, eta):
        mean = self.mean(eta)
        return jnp.diag(mean) - jnp.outer(mean, mean) + jnp.eye(len(eta)) * self.eps

