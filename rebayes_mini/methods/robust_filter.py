import jax
import chex
import jax.numpy as jnp
from functools import partial
from jax.flatten_util import ravel_pytree
from jax.scipy.special import digamma
from rebayes_mini.methods.gauss_filter import KalmanFilter, ExtendedKalmanFilter
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
class KFTState:
    """State of the Kalman Filter"""
    mean: chex.Array
    cov: chex.Array
    err: chex.Array

@chex.dataclass
class RobustStState:
    # State
    mean: chex.Array
    covariance: chex.Array

    # Rt --- Observation covariance
    obs_cov_scale: chex.Array # U
    obs_cov_dof: float # u

    # 1 / lambda --- noise scaling
    weighting_shape: float # lambda-alpha
    weighting_rate: float # lambda-beta

    # nu --- degrees of freedom for noise scaling
    dof_shape: float # nu-a
    dof_rate: float # nu-b

    rho: float
    dim_obs: int


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


class ExtendedRobustKalmanFilter(ExtendedKalmanFilter):
    def __init__(
        self, fn_latent, fn_obs, dynamics_covariance, prior_observation_covariance, n_inner,
        noise_scaling
    ):
        super().__init__(fn_latent, fn_obs, dynamics_covariance, prior_observation_covariance)
        self.n_inner = n_inner
        self.noise_scaling = noise_scaling

    def _update(self, _, bel, bel_pred, x, y):
        Ht = self.jac_obs(bel.mean, x)
        I = jnp.eye(len(bel.mean))
        yhat_corr = self.vobs_fn(bel.mean, x) # + Ht @ (bel.mean - bel_pred.mean)
        S = (y - yhat_corr) @ (y - yhat_corr).T + Ht @ bel.cov @ Ht.T
        Lambda = (self.noise_scaling * self.observation_covariance + S) / (self.noise_scaling + 1)

        Kt = jnp.linalg.solve(Ht @ bel_pred.cov @ Ht.T + Lambda, Ht @ bel_pred.cov)
        mean_new = bel_pred.mean + Kt.T @ (y - self.vobs_fn(bel_pred.mean, x))
        cov_new = Kt.T @ Lambda @ Kt + (I - Ht.T @ Kt).T @ bel_pred.cov @ (I - Ht.T @ Kt)

        bel = bel.replace(mean=mean_new, cov=cov_new)
        return bel

    def step(self, bel, xs, callback_fn):
        x, y = xs
        bel_pred = super()._predict_step(bel)
        partial_update = partial(self._update, bel_pred=bel_pred, x=x, y=y)
        bel_update = jax.lax.fori_loop(0, self.n_inner, partial_update, bel_pred)
        output = callback_fn(bel_update, bel_pred, y, x)

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


class RobustStFilter(ExtendedKalmanFilter):
    """
    Huang2016 modified with Extended Kalman filter predict and update equations
    """
    def __init__(
        self, fn_latent, fn_obs, dynamics_covariance, n_inner
    ):
        self.fn_latent = fn_latent
        self.fn_obs = fn_obs
        self.dynamics_covariance = dynamics_covariance
        self.n_inner = n_inner

    def init_bel(
        self, mean, covariance, obs_cov_scale, obs_cov_dof,
        dof_shape, dof_rate, rho, dim_obs
    ):
        self.rfn, self.vlatent_fn, self.vobs_fn, vlatent = super()._initalise_vector_fns(mean)
        self.jac_latent = jax.jacrev(self.vlatent_fn) # Ft
        self.jac_obs = jax.jacrev(self.vobs_fn) # Ht

        dim_latent = len(vlatent)
        return RobustStState(
            mean=mean,
            covariance=covariance * jnp.eye(dim_latent),
            obs_cov_scale=obs_cov_scale * jnp.eye(dim_obs),
            obs_cov_dof=obs_cov_dof,
            dof_shape=dof_shape,
            dof_rate=dof_rate,
            weighting_shape=1.0,
            weighting_rate=1.0,
            rho=rho,
            dim_obs=dim_obs
        )

    def _ekf_predict_step(self, bel):
        # Refactor EKF to make use of th original function
        Ft = self.jac_latent(bel.mean)
        mean_pred = self.vlatent_fn(bel.mean)
        cov_pred = Ft @ bel.covariance @ Ft.T + self.dynamics_covariance
        bel = bel.replace(mean=mean_pred, covariance=cov_pred)
        return bel

    def _ekf_update_step(self, bel, observation_covariance, y, x):
        Ht = self.jac_obs(bel.mean, x)
        Rt_inv = jnp.linalg.inv(observation_covariance)
        yhat = self.vobs_fn(bel.mean, x)
        prec_update = jnp.linalg.inv(bel.covariance) + Ht.T @ Rt_inv @ Ht
        cov_update = jnp.linalg.inv(prec_update)
        Kt = cov_update @ Ht.T @ Rt_inv
        mean_update = bel.mean + Kt @ (y - yhat)

        bel = bel.replace(mean=mean_update, covariance=cov_update)
        return bel
    
    def _compute_D_term(self, bel, bel_pred, y, x):
        """
        Equation (15) 
        """
        Ht = self.jac_obs(bel_pred.mean, x)
        ht = self.vobs_fn(bel_pred.mean, x)
        yhat_c = ht + Ht @ (bel.mean - bel_pred.mean)
        err = y - yhat_c
        D = jnp.outer(err, err) + Ht @ bel.covariance @ Ht.T
        return D
    
    def _compute_initial_expectations(self, bel, bel_pred, y, x):
        """
        Compute initial expectations using (28) - (32)
        """
        expected_obs_prec = (bel.obs_cov_dof - bel.dim_obs - 1) * jnp.linalg.inv(bel.obs_cov_scale) # (28)
        expected_weighting_term = bel.weighting_shape / bel.weighting_rate # (29)
        expected_dof = bel.dof_shape / bel.dof_rate # (30)
        # D_term = self._compute_D_term(bel, bel_pred, y, x)
        # expected_log_weighting_term = digamma(bel.weighting_shape) - jnp.log(bel.weighting_rate) # (32)

        expectations = expected_obs_prec, expected_weighting_term, expected_dof
        return expectations
    
    def _predict_step(self, bel):
        # Time update
        # bel = super()._predict_step(bel) # EKF predict step
        bel = self._ekf_predict_step(bel)
        obs_cov_dof = bel.rho * (
            bel.obs_cov_dof + bel.dim_obs - 1
        ) + bel.dim_obs + 1
        obs_cov_scale = bel.rho * bel.obs_cov_scale
        dof_shape = bel.rho * bel.dof_shape
        dof_rate = bel.rho * bel.dof_rate

        # Measurement update
        dof_shape = dof_shape + 0.5
        weighting_shape = 0.5 * dof_shape / dof_rate
        weighting_rate = 0.5 * dof_shape / dof_rate 

        bel = bel.replace(
            obs_cov_dof=obs_cov_dof,
            obs_cov_scale=obs_cov_scale,
            dof_shape=dof_shape,
            dof_rate=dof_rate,
            weighting_shape=weighting_shape,
            weighting_rate=weighting_rate
        )

        return bel

    def _update_step(self, i, group, y, x, bel_pred):
        bel, expected_terms = group
        expected_obs_prec, expected_weighting_term, expected_dof = expected_terms
        # Time update
        obs_cov_est = jnp.linalg.inv(expected_obs_prec) / expected_weighting_term # (11)
        bel = self._ekf_update_step(bel, obs_cov_est, y, x)

        D = self._compute_D_term(bel, bel_pred, y, x) # (31)

        weighting_shape = (bel.dim_obs + expected_dof) / 2 #(17)
        weighting_rate = jnp.einsum("ij,ji->", D, obs_cov_est) / 2 + expected_dof / 2 # (18)

        expected_weighting_term = weighting_shape / weighting_rate # (29)
        expected_log_weighting_term = digamma(weighting_shape) - jnp.log(weighting_rate) # (32)

        obs_cov_dof = bel_pred.obs_cov_dof + 1.0 # (21)
        obs_cov_scale =  bel_pred.obs_cov_scale + expected_weighting_term * D # (22)

        # degrees of freedom's shape and rate
        dof_shape = bel_pred.dof_shape + 0.5 # (26)
        dof_rate = bel_pred.dof_rate - 0.5 - 0.5 * expected_log_weighting_term + 0.5 * expected_weighting_term # (27)

        expected_obs_prec = (obs_cov_dof - bel.dim_obs - 1) * jnp.linalg.inv(obs_cov_scale) # (28)
        expected_dof = dof_shape / dof_rate # (30)

        bel = bel.replace(
            obs_cov_dof=obs_cov_dof,
            obs_cov_scale=obs_cov_scale,
            dof_shape=dof_shape,
            dof_rate=dof_rate,
            weighting_shape=weighting_shape,
            weighting_rate=weighting_rate,
        )

        expected_terms = (
            expected_obs_prec, expected_weighting_term, expected_dof
        )

        group = bel, expected_terms
        return group

    def step(self, bel, xs, callback_fn):
        xt, yt = xs
        bel_pred = self._predict_step(bel)
        partial_update = partial(self._update_step, y=yt, x=xt, bel_pred=bel_pred)
        expected_terms = self._compute_initial_expectations(bel_pred, bel_pred, yt, xt)
        group_init = bel_pred, expected_terms
        bel_update, _ = jax.lax.fori_loop(0, self.n_inner, partial_update, group_init)
        output = callback_fn(bel_update, bel_pred, yt, xt)

        return bel_update, output


class ExtendedThresholdedKalmanFilter(ExtendedKalmanFilter):
    """
    Heuristic-based thresholding of the update
    first presented in Ting et al. 2007
    """
    def __init__(
        self, fn_latent, fn_observed, dynamics_covariance, observation_covariance, threshold
    ):
        super().__init__(fn_latent, fn_observed, dynamics_covariance, observation_covariance)
        self.threshold = threshold

    def init_bel(self, mean, cov=1.0):
        mean, cov, dim_latent = self._init_components(mean, cov)

        return KFTState(
            mean=mean,
            cov=cov,
            err=0.0
        )

    def _update_step(self, bel, y, x):
        Ht = self.jac_obs(bel.mean, x)
        Rt_inv = jnp.linalg.inv(self.observation_covariance)
        yhat = self.vobs_fn(bel.mean, x)
        prec_update = jnp.linalg.inv(bel.cov) + Ht.T @ Rt_inv @ Ht
        cov_update = jnp.linalg.inv(prec_update)
        Kt = cov_update @ Ht.T @ Rt_inv
        err = y - yhat
        mean_update = bel.mean + Kt @ err
        bel = bel.replace(mean=mean_update, cov=cov_update)
        return bel, err

    def step(self, bel, xs, callback_fn):
        xt, yt = xs
        bel_pred = self._predict_step(bel)
        bel_update, err = self._update_step(bel_pred, yt, xt)
        err_distance = jnp.sqrt(jnp.einsum("j,jk,k->", err, jnp.linalg.inv(self.observation_covariance), err))

        bel_update = jax.lax.cond(err_distance < self.threshold, lambda: bel_update, lambda: bel_pred)
        bel_update = bel_update.replace(err=err_distance)

        output = callback_fn(bel_update, bel_pred, xt, yt)
        return bel_update, output
