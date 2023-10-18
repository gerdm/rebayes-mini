import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable
from functools import partial
from rebayes_mini.methods import kalman_filter as kf
from jax.flatten_util import ravel_pytree

def subcify(cls):
    class SubspaceModule(nn.Module):
        dim_in: int
        dim_subspace: int
        init_normal: Callable = nn.initializers.normal()
        init_proj: Callable = nn.initializers.normal()

        def init(self, rngs, *args, **kwargs):
            r1, r2 = jax.random.split(rngs, 2)
            rngs_dict = {"params": r1, "fixed": r2}

            return nn.Module.init(self, rngs_dict, *args, **kwargs)

        def setup(self):
            key_dummy = jax.random.PRNGKey(0)
            params = cls().init(key_dummy, jnp.ones((1, self.dim_in)))
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
        P = params_fixed["P"]
        P = P / jnp.linalg.norm(P, axis=0) * 2
        params_fixed["P"] = P

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
