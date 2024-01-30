import jax
import chex
import jax.numpy as jnp
from functools import partial
from rebayes_mini import callbacks

class EnsembleKalmanFilter:
    def __init__(
        self, latent_fn, obs_fn, n_particles
    ):
        self.latent_fn = latent_fn
        self.obs_fn = obs_fn
        self.n_particles = n_particles
        self.matrix_deviation = jnp.eye(n_particles) - jnp.ones((n_particles, n_particles)) / n_particles
    
    def init_bel(self, key, dim_latent):
        particles = jax.random.normal(key, (self.n_particles, dim_latent))
        return particles

    def _predict_step(self, particles, key, X):
        key_latent, key_obs = jax.random.split(key)
        keys_latent = jax.random.split(key_latent, self.n_particles)
        keys_obs = jax.random.split(key_obs, self.n_particles)       

        latent_pred = jax.vmap(self.latent_fn, in_axes=(0, 0, None))(particles, keys_latent, X)
        obs_pred = jax.vmap(self.obs_fn, in_axes=(0, 0, None))(latent_pred, keys_obs, X)

        return latent_pred, obs_pred
    
    def _update_step(self, latent_pred, obs_pred, y):
        latent_pred_hat = jnp.einsum("ji,jk->ki", latent_pred, self.matrix_deviation)
        obs_pred_hat = jnp.einsum("ji,jk->ki", obs_pred, self.matrix_deviation)

        Mk = jnp.einsum("ji,jk->ik", latent_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        Sk = jnp.einsum("ji,jk->ik", obs_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        K = jnp.linalg.solve(Sk, Mk)

        latent = latent_pred + jnp.einsum("ij,kj->ki", K, y - obs_pred)

        return latent
    
    def step(self, particles, obs, key, callback_fn):
        yt, xt, t = obs
        key = jax.random.fold_in(key, t)
        particles_latent_pred, particles_obs_pred = self._predict_step(particles, key, xt)
        particles_latent = self._update_step(particles_latent_pred, particles_obs_pred, yt)

        out = callback_fn(particles_latent, particles_latent_pred, yt, xt)

        return particles_latent, out
    
    def scan(self, particles_init, key, y, X=None, callback_fn=None):
        n_steps = len(y)
        timesteps = jnp.arange(n_steps)
        X = X if X is not None else timesteps
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        _step = partial(self.step, callback_fn=callback_fn, key=key)
        xs = (y, X, timesteps)
        particles, hist = jax.lax.scan(_step, particles_init, xs)
        return particles, hist


class EnsembleKalmanFilterInflation(EnsembleKalmanFilter):
    def __init__(
        self, latent_fn, obs_fn, n_particles, inflation_factor
    ):
        super().__init__(latent_fn, obs_fn, n_particles)
        self.inflation_factor = inflation_factor


    def _update_step(self, latent_pred, obs_pred, y):
        latent_pred_hat = jnp.einsum("ji,jk->ki", latent_pred, self.matrix_deviation)
        obs_pred_hat = jnp.einsum("ji,jk->ki", obs_pred, self.matrix_deviation)

        latent_pred_hat = latent_pred_hat - latent_pred_hat.mean(axis=0, keepdims=True)
        Mk = jnp.einsum("ji,jk->ik", latent_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        Mk = Mk.at[jnp.diag_indices_from(Mk)].set(jnp.diag(Mk) * self.inflation_factor)
        # Sk = jnp.einsum("ji,jk->ik", obs_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        K = jnp.linalg.solve(Mk + jnp.eye(Mk.shape[0]), Mk)


        errs = y - obs_pred
        latent = latent_pred + jnp.einsum("ij,kj->ki", K, errs)

        return latent
    


class WLEnsembleKalmanFilterSoft(EnsembleKalmanFilter):
    """
    Weighted likelihood Ensemble Kalman Filter
    """
    def __init__(
        self, latent_fn, obs_fn, n_particles, c
    ):
        super().__init__(latent_fn, obs_fn, n_particles)
        self.c = c

    def _update_step(self, latent_pred, obs_pred, y):
        latent_pred_hat = jnp.einsum("ji,jk->ki", latent_pred, self.matrix_deviation)
        obs_pred_hat = jnp.einsum("ji,jk->ki", obs_pred, self.matrix_deviation)

        Mk = jnp.einsum("ji,jk->ik", latent_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        Sk = jnp.einsum("ji,jk->ik", obs_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        K = jnp.linalg.solve(Sk, Mk)

        errs = y - obs_pred
        wt = jnp.sqrt(jnp.power(errs, 2)) < self.c
        latent = latent_pred + jnp.einsum("ij,kj->ki", K, wt * errs)

        return latent


class WLEnsembleKalmanFilterHard(EnsembleKalmanFilter):
    """
    Weighted likelihood Ensemble Kalman Filter
    """
    def __init__(
        self, latent_fn, obs_fn, n_particles, c
    ):
        super().__init__(latent_fn, obs_fn, n_particles)
        self.c = c

    def _update_step(self, latent_pred, obs_pred, y):
        latent_pred_hat = jnp.einsum("ji,jk->ki", latent_pred, self.matrix_deviation)
        obs_pred_hat = jnp.einsum("ji,jk->ki", obs_pred, self.matrix_deviation)

        Mk = jnp.einsum("ji,jk->ik", latent_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        Sk = jnp.einsum("ji,jk->ik", obs_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        K = jnp.linalg.solve(Sk, Mk)

        errs = y - obs_pred
        wt = jnp.sqrt(jnp.power(errs, 2).mean(axis=0)) < self.c
        latent = latent_pred + jnp.einsum("ij,kj->ki", K, wt * errs)

        return latent


class WLEnsembleKalmanFilterResample(EnsembleKalmanFilter):
    """
    Weighted likelihood Ensemble Kalman Filter
    """
    def __init__(
        self, latent_fn, obs_fn, n_particles, c
    ):
        super().__init__(latent_fn, obs_fn, n_particles)
        self.c = c

    @partial(jax.vmap, in_axes=(None, 0, 1, 1), out_axes=1)
    def resample_from_column(self, key, particles, weights):
        p_weights = jnp.ones(self.n_particles) / (self.n_particles - weights.sum())
        rval = weights.argmax()
        ix_where = jnp.where(weights, size=self.n_particles, fill_value=rval)
        p_weights = p_weights.at[ix_where].set(0.0)
        particles_resample = jax.random.choice(key, particles, shape=(self.n_particles,), p=p_weights)
        new_particles =  particles * (1 - weights) + particles_resample * weights
        
        return new_particles

    def _update_step(self, key, latent_pred, obs_pred, y):
        latent_pred_hat = jnp.einsum("ji,jk->ki", latent_pred, self.matrix_deviation)
        obs_pred_hat = jnp.einsum("ji,jk->ki", obs_pred, self.matrix_deviation)

        errs = y - obs_pred_hat
        # keys = jax.random.split(key, errs.shape[1])
        wt = jnp.sqrt(jnp.power(errs, 2)) < self.c
        # obs_pred_hat = obs_pred_hat * wt

        # obs_pred_hat = obs_pred_hat * wt
        wt = jnp.mean(wt, axis=0)

        Mk = jnp.einsum("ji,jk->ik", latent_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        Sk = jnp.einsum("ji,jk->ik", obs_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        # K = jnp.linalg.solve(Sk, Mk)
        K = jnp.diag(wt) @ jnp.linalg.inv(Sk) @ Mk @ jnp.diag(wt)


        # obs_pred = self.resample_from_column(keys, obs_pred, wt)
        errs = y - obs_pred_hat
        latent = latent_pred + jnp.einsum("ij,kj->ki", K, errs)

        return latent

    def step(self, particles, obs, key, callback_fn):
        yt, xt, t = obs
        key = jax.random.fold_in(key, t)
        key_pred, key_update = jax.random.split(key)
        particles_latent_pred, particles_obs_pred = self._predict_step(particles, key_pred, xt)
        particles_latent = self._update_step(key_update, particles_latent_pred, particles_obs_pred, yt)

        out = callback_fn(particles_latent, particles_latent_pred, yt, xt)

        return particles_latent, out


class HubEnsembleKalmanFilter(EnsembleKalmanFilter):
    """
    Huberised likelihood Ensemble Kalman Filter
    """
    def __init__(
        self, latent_fn, obs_fn, n_particles, clipping_height
    ):
        super().__init__(latent_fn, obs_fn, n_particles)
        self.clipping_height = clipping_height
    

    def huberisation(self, x):
        return jnp.clip(x, -self.clipping_height, self.clipping_height)

    def _update_step(self, latent_pred, obs_pred, y):
        latent_pred_hat = jnp.einsum("ji,jk->ki", latent_pred, self.matrix_deviation)
        obs_pred_hat = jnp.einsum("ji,jk->ki", obs_pred, self.matrix_deviation)

        Mk = jnp.einsum("ji,jk->ik", latent_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        Sk = jnp.einsum("ji,jk->ik", obs_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        K = jnp.linalg.solve(Sk, Mk)

        errs = y - obs_pred
        errs = self.huberisation(errs)
        latent = latent_pred + jnp.einsum("ij,kj->ki", K, errs)

        return latent
