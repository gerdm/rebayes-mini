import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable
from functools import partial
from rebayes_mini.methods import gauss_filter as kf
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


def subcify(cls):
    class SubspaceModule(nn.Module):
        dim_in: int
        dim_subspace: int
        init_proj: Callable = nn.initializers.normal(stddev=0.1)

        def init(self, rngs, *args, **kwargs):
            r1, r2 = jax.random.split(rngs, 2)
            rngs_dict = {"params": r1, "fixed": r2}

            return nn.Module.init(self, rngs_dict, *args, **kwargs)

        def setup(self):
            key_dummy = jax.random.PRNGKey(0)
            vinit = (1, self.dim_in) if isinstance(self.dim_in, int) else (1, *self.dim_in)
            params = cls().init(key_dummy, jnp.ones(vinit))
            params_all, reconstruct_fn = ravel_pytree(params)

            self.dim_full = len(params_all)
            self.reconstruct_fn = reconstruct_fn

            self.subspace = self.param(
                "subspace",
                self.init_proj,
                (self.dim_subspace,)
            )

            shape = (self.dim_full, self.dim_subspace)
            init_fn = lambda shape: self.init_proj(self.make_rng("fixed"), shape)
            self.projection = self.variable("fixed", "P", init_fn, shape).value

            shape = (self.dim_full,)
            init_fn = lambda shape: self.init_proj(self.make_rng("fixed"), shape)
            self.bias = self.variable("fixed", "b", init_fn, shape).value

        @nn.compact
        def __call__(self, x):
            params = self.projection @ self.subspace  + self.bias
            params = self.reconstruct_fn(params)
            return cls().apply(params, x)

    return SubspaceModule


class SubspaceFilter(kf.ExpfamFilter):
    def __init__(
        self,
        apply_fn: Callable,
        log_partition: Callable,
        suff_statistic: Callable,
        dynamics_covariance: float,
    ):
        super().__init__(
            apply_fn,
            log_partition,
            suff_statistic,
            dynamics_covariance,
        )


    def _initialise_link_fn(self, apply_fn, params):
        params_fixed = params["fixed"]
        P, path = find_key_value_and_path(params_fixed, "P")
        P = P / jnp.linalg.norm(P, axis=0)
        update_nested_dict(params_fixed, path, P)

        params_train = params["params"]

        flat_params, rfn = ravel_pytree(params_train)

        @jax.jit
        def link_fn(params, x):
            p_train_recon = rfn(params)
            p_full = {
                "fixed": params_fixed,
                "params": p_train_recon
            }
            return apply_fn(p_full, x)

        return rfn, link_fn, flat_params


class BernoulliFilter(SubspaceFilter):
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


class MultinomialFilter(SubspaceFilter):
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

