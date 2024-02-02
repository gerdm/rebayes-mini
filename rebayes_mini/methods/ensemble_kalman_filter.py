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
    

    def _load_gain_errs(self, latent_pred, obs_pred, y):

        latent_pred_hat = latent_pred - latent_pred.mean(axis=0, keepdims=True)
        obs_pred_hat = obs_pred - obs_pred.mean(axis=0, keepdims=True)
        # latent_pred_hat = latent_pred - latent_pred.mean(axis=0, keepdims=True)
        # Estimate of the state covariance matrix
        # Mk = jnp.einsum("ji,jk->ik", latent_pred_hat, obs_pred) / (self.n_particles - 1)
        Mk = jnp.einsum("ji,jk->ik", latent_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        Mk = Mk.at[jnp.diag_indices_from(Mk)].set(jnp.diag(Mk) * self.inflation_factor)
        # Sk = jnp.einsum("ji,jk->ik", obs_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        gain = jnp.linalg.solve(Mk + jnp.eye(Mk.shape[0]), Mk)
        errs = y - obs_pred # innovation

        return gain, errs

    def _update_step(self, latent_pred, obs_pred, y):
        gain, errs = self._load_gain_errs(latent_pred, obs_pred, y)
        latent = latent_pred + jnp.einsum("ij,kj->ki", gain, errs)
        return latent
    


class WLEnsembleKalmanFilterPP(EnsembleKalmanFilterInflation):
    def __init__(
        self, latent_fn, obs_fn, n_particles, inflation_factor, threshold
    ):
        """
        Weighted-likelihood Ensemble Kalman Filter with
        inflation and per-particle weighting
        """
        super().__init__(latent_fn, obs_fn, n_particles, inflation_factor)
        self.threshold = threshold


    def _update_step(self, latent_pred, obs_pred, y):
        gain, errs = self._load_gain_errs(latent_pred, obs_pred, y)
        wt = jnp.sqrt(jnp.power(errs, 2)) < self.threshold
        latent = latent_pred + jnp.einsum("ij,kj->ki", gain, wt * errs)

        return latent


class WLEnsembleKalmanFilterAP(EnsembleKalmanFilterInflation):
    def __init__(
        self, latent_fn, obs_fn, n_particles, inflation_factor, threshold
    ):
        """
        Weighted-likelihood Ensemble Kalman Filter with
        inflation and average particle
        """
        super().__init__(latent_fn, obs_fn, n_particles, inflation_factor)
        self.threshold = threshold


    def _update_step(self, latent_pred, obs_pred, y):
        gain, errs = self._load_gain_errs(latent_pred, obs_pred, y)
        wt = jnp.sqrt(jnp.power(errs, 2).mean(axis=0)) < self.threshold
        latent = latent_pred + jnp.einsum("ij,kj->ki", gain, wt * errs)

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


class HubEnsembleKalmanFilter(EnsembleKalmanFilter):
    """
    Huberised likelihood Ensemble Kalman Filter
    """
    def __init__(
        self, latent_fn, obs_fn, n_particles, inflation_factor, clipping_height
    ):
        super().__init__(latent_fn, obs_fn, n_particles)
        self.clipping_height = clipping_height
        self.inflation_factor = inflation_factor
    

    def huberisation(self, x):
        return jnp.clip(x, -self.clipping_height, self.clipping_height)

    def _update_step(self, latent_pred, obs_pred, y):
        # latent_pred_hat = jnp.einsum("ji,jk->ki", latent_pred, self.matrix_deviation)
        # obs_pred_hat = jnp.einsum("ji,jk->ki", obs_pred, self.matrix_deviation)

        # Mk = jnp.einsum("ji,jk->ik", latent_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        # Sk = jnp.einsum("ji,jk->ik", obs_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        # K = jnp.linalg.solve(Sk, Mk)

        # errs = y - obs_pred
        # errs = self.huberisation(errs)
        # latent = latent_pred + jnp.einsum("ij,kj->ki", K, errs)

        # return latent

        latent_pred_hat = jnp.einsum("ji,jk->ki", latent_pred, self.matrix_deviation)
        obs_pred_hat = jnp.einsum("ji,jk->ki", obs_pred, self.matrix_deviation)

        latent_pred_hat = latent_pred_hat - latent_pred_hat.mean(axis=0, keepdims=True)
        Mk = jnp.einsum("ji,jk->ik", latent_pred_hat, obs_pred_hat) / (self.n_particles - 1)
        Mk = Mk.at[jnp.diag_indices_from(Mk)].set(jnp.diag(Mk) * self.inflation_factor) # covariance inflation
        K = jnp.linalg.solve(Mk + jnp.eye(Mk.shape[0]), Mk)


        errs = y - obs_pred
        errs = self.huberisation(errs)
        latent = latent_pred + jnp.einsum("ij,kj->ki", K, errs)

        return latent