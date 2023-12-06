import jax
import chex
import jax.numpy as jnp
from functools import partial
from jax.flatten_util import ravel_pytree
from rebayes_mini.methods.gauss_filter import KalmanFilter
from rebayes_mini import callbacks


@chex.dataclass
class WOCFState:
    """
    Weighted Obs Cov Filter state
    """
    mean: chex.Array
    covariance: chex.Array
    key: chex.Array = None
    weighting_term: float = 1.0


@chex.dataclass
class RobustStState:
    mean: chex.Array
    covariance: chex.Array
    obs_cov_scale: chex.Array
    obs_cov_dof: float
    dof_a: float
    dof_b: float
    scale_nu: float
    rho: float


class WeightedObsCovFilter(KalmanFilter):
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
        state = WOCFState(
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
        pprec = jnp.linalg.inv(self.dynamics_covariance) + weighting_term * x.T @ Rinv @ x
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


class RobustStFilter:
    """
    Huang2016 with Extended Kalman filter predict and update equations
    """
    def __init__(
        self, fn_latent, fn_obs, dynamics_covariance,
    ):
        self.fn_latent = fn_latent
        self.fn_obs = fn_obs
        self.dynamics_covariance = dynamics_covariance
    

    def init_bel(
        self, mean, covariance, obs_cov_scale, obs_cov_dof, dof_a, dof_b, scale_nu, rho
    ):
        return RobustStState(
            mean=mean,
            covariance=covariance,
            obs_cov_scale=obs_cov_scale,
            obs_cov_dof=obs_cov_dof,
            dof_a=dof_a,
            dof_b=dof_b,
            scale_nu=scale_nu,
            rho=rho,
        )