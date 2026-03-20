import jax
import jax.numpy as jnp
from functools import partial
from jax.flatten_util import ravel_pytree
from jax.scipy.special import digamma
from rebayes_mini.methods.replay_sgd import FifoSGD
from rebayes_mini.methods.base_filter import ExtendedFilter
from rebayes_mini.states import OutlierEKFState, WOCFState, KFTState, RobustStState


class WeightedObsCovFilter(ExtendedFilter):
    """
    Ting, JA., Theodorou, E., Schaal, S. (2007).
    Learning an Outlier-Robust Kalman Filter.
    In: Kok, J.N., Koronacki, J., Mantaras, R.L.d., Matwin, S., Mladenič, D., Skowron, A. (eds)
    Machine Learning: ECML 2007. ECML 2007. Lecture Notes in Computer Science(),
    vol 4701. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-74958-5_76

    Special case for known SSM hyperparameters --- no M-step.

    Parameters
    ----------
    mean_fn :
        Callable (params, x) -> y_hat.
    dynamics_covariance :
        Scalar or matrix added to parameter covariance at each predict step.
    observation_covariance :
        Fixed observation noise covariance matrix R.
    prior_shape, prior_rate :
        Gamma prior parameters on the precision weighting term.
    """
    def __init__(
        self, mean_fn, dynamics_covariance, observation_covariance,
        prior_shape, prior_rate, n_inner=1, n_samples=10
    ):
        super().__init__(mean_fn, cov_fn=None, dynamics_covariance=dynamics_covariance, n_inner=n_inner)
        self.observation_covariance = observation_covariance
        self.prior_shape = prior_shape
        self.prior_rate = prior_rate
        self.n_samples = n_samples

    def init_bel(self, params, cov=1.0, key=314):
        bel = super().init_bel(params, cov)
        return WOCFState(
            mean=bel.mean,
            cov=bel.cov,
            weighting_term=1.0,
            key=jax.random.PRNGKey(key),
        )

    @partial(jax.vmap, in_axes=(None, 0, None, None, None))
    def _err_term(self, mean, y, x, R_inv):
        """
        Error correcting term for the posterior mean and covariance.
        """
        err = y - self.mean_fn(mean, x)
        return err.T @ R_inv @ err

    def _inner_update(self, i, bel, bel_prev, y, x):
        keyt = jax.random.fold_in(bel.key, i)
        mean_samples = jax.random.multivariate_normal(keyt, bel.mean, bel.cov, (self.n_samples,))
        Rinv = jnp.linalg.inv(self.observation_covariance)
        mean_err = self._err_term(mean_samples, y, x, Rinv).mean()

        yhat = self.mean_fn(bel_prev.mean, x)
        weighting_term = (self.prior_shape + 0.5) / (self.prior_rate + mean_err / 2)
        Ht = self.grad_mean(bel_prev.mean, x)
        pprec = jnp.linalg.inv(bel_prev.cov) + weighting_term * Ht.T @ Rinv @ Ht
        pcov = jnp.linalg.inv(pprec)
        pmean = bel_prev.mean + weighting_term * pcov @ Ht.T @ Rinv @ (y - yhat)

        bel = bel.replace(
            mean=pmean,
            cov=pcov,
            weighting_term=weighting_term,
            key=keyt,
        )
        return bel

    def update(self, bel, y, x):
        _update = partial(self._inner_update, bel_prev=bel, y=y, x=x)
        bel_update = jax.lax.fori_loop(0, self.n_inner, _update, bel)
        return bel_update


class ExtendedFilterInverseWishart(ExtendedFilter):
    """
    EKF with Inverse-Wishart adaptive observation covariance.

    Agamenoni, G., Nieto, J.I., Nebot, E.M. (2012).
    """
    def __init__(
        self, mean_fn, dynamics_covariance, prior_observation_covariance,
        n_inner, noise_scaling
    ):
        super().__init__(mean_fn, cov_fn=None, dynamics_covariance=dynamics_covariance, n_inner=n_inner)
        self.prior_observation_covariance = prior_observation_covariance
        self.noise_scaling = noise_scaling

    def _obs_cov(self, bel, y, x):
        yhat = self.mean_fn(bel.mean, x)
        Ht = self.grad_mean(bel.mean, x)
        S = jnp.outer(y - yhat, y - yhat) + Ht @ bel.cov @ Ht.T
        return (self.noise_scaling * self.prior_observation_covariance + S) / (self.noise_scaling + 1)

    def _inner_update(self, _, bel, bel_pred, y, x):
        Lambda = self._obs_cov(bel, y, x)
        yhat = self.mean_fn(bel_pred.mean, x)
        Ht = self.grad_mean(bel_pred.mean, x)
        I = jnp.eye(len(bel.mean))
        Kt = jnp.linalg.solve(Ht @ bel_pred.cov @ Ht.T + Lambda, Ht @ bel_pred.cov)
        mean_new = bel_pred.mean + Kt.T @ (y - yhat)
        cov_new = Kt.T @ Lambda @ Kt + (I - Ht.T @ Kt).T @ bel_pred.cov @ (I - Ht.T @ Kt)
        return bel.replace(mean=mean_new, cov=cov_new)

    def update(self, bel, y, x):
        bel_pred = bel
        _iter = partial(self._inner_update, bel_pred=bel_pred, y=y, x=x)
        return jax.lax.fori_loop(0, self.n_inner, _iter, bel_pred)


class RobustStFilter(ExtendedFilter):
    """
    Robust filter with adaptive Student-t observation noise.

    Huang, Y. et al. (2016). A novel robust Student's t-based Kalman filter.
    Ported to the parameter-space / BaseFilter API: the latent state is the
    flattened parameter vector; dynamics are a parameter random walk.

    Parameters
    ----------
    mean_fn :
        Callable (params, x) -> y_hat.
    dynamics_covariance :
        Scalar added to parameter covariance at each predict step (random walk).
    obs_cov_scale :
        Initial scalar for the observation covariance scale matrix U
        (will be multiplied by I_{dim_obs}).
    obs_cov_dof :
        Initial degrees-of-freedom for the Inverse-Wishart on R_t.
    dof_shape, dof_rate :
        Initial Gamma shape/rate for the Student-t degrees-of-freedom nu.
    rho :
        Forgetting factor in (0, 1] for the hyperparameter predict step.
    dim_obs :
        Dimension of the observation vector.
    n_inner :
        Number of inner VI iterations per time step.
    """

    def __init__(
        self,
        mean_fn,
        dynamics_covariance=0.0,
        obs_cov_scale=1.0,
        obs_cov_dof=5.0,
        dof_shape=2.0,
        dof_rate=2.0,
        rho=1.0,
        dim_obs=1,
        n_inner=1,
    ):
        super().__init__(mean_fn, cov_fn=None, dynamics_covariance=dynamics_covariance, n_inner=n_inner)
        self.obs_cov_scale0 = float(obs_cov_scale)
        self.obs_cov_dof0 = float(obs_cov_dof)
        self.dof_shape0 = float(dof_shape)
        self.dof_rate0 = float(dof_rate)
        self.rho = float(rho)
        self.dim_obs = int(dim_obs)

    def init_bel(self, params, cov=1.0):
        # Initialise flat params and Jacobian via base class helper.
        self.rfn, self.mean_fn, init_params = self._initialise_flat_fn(self.mean_fn_og, params)
        self.grad_mean = jax.jacrev(self.mean_fn)
        nparams = len(init_params)
        return RobustStState(
            mean=init_params,
            cov=jnp.eye(nparams) * cov,
            obs_cov_scale=self.obs_cov_scale0 * jnp.eye(self.dim_obs),
            obs_cov_dof=self.obs_cov_dof0,
            dof_shape=self.dof_shape0,
            dof_rate=self.dof_rate0,
            weighting_shape=1.0,
            weighting_rate=1.0,
            rho=self.rho,
            dim_obs=self.dim_obs,
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, bel):
        # Random-walk dynamics on parameter mean/cov (base class logic).
        nparams = len(bel.mean)
        cov_pred = bel.cov + self.dynamics_covariance * jnp.eye(nparams)

        # Hyperparameter time-propagation.
        obs_cov_dof = bel.rho * (bel.obs_cov_dof + bel.dim_obs - 1) + bel.dim_obs + 1
        obs_cov_scale = bel.rho * bel.obs_cov_scale
        dof_shape = bel.rho * bel.dof_shape + 0.5
        dof_rate = bel.rho * bel.dof_rate
        weighting_shape = 0.5 * dof_shape / dof_rate
        weighting_rate = 0.5 * dof_shape / dof_rate

        return bel.replace(
            cov=cov_pred,
            obs_cov_dof=obs_cov_dof,
            obs_cov_scale=obs_cov_scale,
            dof_shape=dof_shape,
            dof_rate=dof_rate,
            weighting_shape=weighting_shape,
            weighting_rate=weighting_rate,
        )

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def _ekf_update(self, bel, observation_covariance, y, x):
        Ht = self.grad_mean(bel.mean, x)
        Rt_inv = jnp.linalg.inv(observation_covariance)
        yhat = self.predict_fn(bel, x)
        prec_update = jnp.linalg.inv(bel.cov) + Ht.T @ Rt_inv @ Ht
        cov_update = jnp.linalg.inv(prec_update)
        Kt = cov_update @ Ht.T @ Rt_inv
        mean_update = bel.mean + Kt @ (y - yhat)
        return bel.replace(mean=mean_update, cov=cov_update)

    def _compute_D_term(self, bel, bel_pred, y, x):
        """
        Equation (31): D = outer(err, err) + H P H^T
        """
        Ht = self.grad_mean(bel_pred.mean, x)
        yhat_c = self.predict_fn(bel_pred, x) + Ht @ (bel.mean - bel_pred.mean)
        err = y - yhat_c
        return jnp.outer(err, err) + Ht @ bel.cov @ Ht.T

    def _compute_initial_expectations(self, bel):
        """
        Compute initial expectations using (28)-(30).
        """
        expected_obs_prec = (bel.obs_cov_dof - bel.dim_obs - 1) * jnp.linalg.inv(bel.obs_cov_scale)  # (28)
        expected_weighting_term = bel.weighting_shape / bel.weighting_rate  # (29)
        expected_dof = bel.dof_shape / bel.dof_rate  # (30)
        return expected_obs_prec, expected_weighting_term, expected_dof

    def _vi_step(self, i, group, y, x, bel_pred):
        bel, expected_terms = group
        expected_obs_prec, expected_weighting_term, expected_dof = expected_terms

        obs_cov_est = jnp.linalg.inv(expected_obs_prec) / expected_weighting_term  # (11)
        bel = self._ekf_update(bel, obs_cov_est, y, x)

        D = self._compute_D_term(bel, bel_pred, y, x)  # (31)

        weighting_shape = (bel.dim_obs + expected_dof) / 2  # (17)
        weighting_rate = jnp.einsum("ij,ji->", D, obs_cov_est) / 2 + expected_dof / 2  # (18)

        expected_weighting_term = weighting_shape / weighting_rate  # (29)
        expected_log_weighting_term = digamma(weighting_shape) - jnp.log(weighting_rate)  # (32)

        obs_cov_dof = bel_pred.obs_cov_dof + 1.0  # (21)
        obs_cov_scale = bel_pred.obs_cov_scale + expected_weighting_term * D  # (22)

        dof_shape = bel_pred.dof_shape + 0.5  # (26)
        dof_rate = bel_pred.dof_rate - 0.5 - 0.5 * expected_log_weighting_term + 0.5 * expected_weighting_term  # (27)

        expected_obs_prec = (obs_cov_dof - bel.dim_obs - 1) * jnp.linalg.inv(obs_cov_scale)  # (28)
        expected_dof = dof_shape / dof_rate  # (30)

        bel = bel.replace(
            obs_cov_dof=obs_cov_dof,
            obs_cov_scale=obs_cov_scale,
            dof_shape=dof_shape,
            dof_rate=dof_rate,
            weighting_shape=weighting_shape,
            weighting_rate=weighting_rate,
        )
        return bel, (expected_obs_prec, expected_weighting_term, expected_dof)

    # ------------------------------------------------------------------
    # BaseFilter API
    # ------------------------------------------------------------------

    def update(self, bel, y, x):
        bel_pred = bel
        expected_terms = self._compute_initial_expectations(bel_pred)
        group_init = bel_pred, expected_terms
        _step = partial(self._vi_step, y=y, x=x, bel_pred=bel_pred)
        bel_update, _ = jax.lax.fori_loop(0, self.n_inner, _step, group_init)
        return bel_update


class ExtendedFilterMD(ExtendedFilter):
    """
    EKF with Mahalanobis-distance hard gating (outlier rejection).

    Observations whose Mahalanobis distance to the prediction exceeds
    `threshold` are silently ignored (weighting term set to 0).
    """
    def __init__(
        self, mean_fn, cov_fn, dynamics_covariance, threshold, n_inner=1
    ):
        super().__init__(mean_fn, cov_fn, dynamics_covariance, n_inner)
        self.threshold = threshold

    def _update(self, bel, bel_pred, y, x):
        yhat = self.predict_fn(bel, x)
        Rt = jnp.atleast_2d(self.cov_fn(yhat))
        Ht = self.grad_mean(bel.mean, x)

        err = y - yhat
        obs_prec = jnp.linalg.inv(Rt)
        mahalanobis = jnp.sqrt(jnp.einsum("j,jk,k->", err, obs_prec, err))
        weighting_term = (mahalanobis < self.threshold).astype(float)

        St = Ht @ bel_pred.cov @ Ht.T + Rt
        Kt = jnp.linalg.solve(St, Ht @ bel_pred.cov).T

        mean_update = bel_pred.mean + weighting_term * Kt @ err
        cov_update = bel_pred.cov - weighting_term * Kt @ St @ Kt.T
        return bel.replace(mean=mean_update, cov=cov_update)


class ExtendedFilterIMQ(ExtendedFilter):
    """
    EKF with inverse multi-quadratic (IMQ) soft outlier weighting.

    The effective observation noise is inflated by 1/w, where
    w = c^2 / (c^2 + ||err||^2) and c = soft_threshold.
    """
    def __init__(
        self, mean_fn, cov_fn, dynamics_covariance, soft_threshold, n_inner=1
    ):
        super().__init__(mean_fn, cov_fn, dynamics_covariance, n_inner)
        self.soft_threshold = soft_threshold

    def _update(self, bel, bel_pred, y, x):
        yhat = self.predict_fn(bel_pred, x)
        err = y - yhat
        weighting_term = self.soft_threshold ** 2 / (self.soft_threshold ** 2 + jnp.inner(err, err))
        Ht = self.grad_mean(bel_pred.mean, x)
        Rt = jnp.atleast_2d(self.cov_fn(yhat)) / weighting_term

        St = Ht @ bel_pred.cov @ Ht.T + Rt
        Kt = jnp.linalg.solve(St, Ht @ bel_pred.cov).T
        I = jnp.eye(len(bel.mean))
        mean_update = bel_pred.mean + Kt @ err
        cov_update = (I - Kt @ Ht) @ bel_pred.cov @ (I - Kt @ Ht).T + Kt @ Rt @ Kt.T
        return bel.replace(mean=mean_update, cov=cov_update)


class ExtendedFilterBernoulli(ExtendedFilter):
    """
    EKF with variational Bernoulli outlier indicator.

    Wang, H., Li, H., Fang, J., & Wang, H. (2018).
    Robust Gaussian Kalman filter with outlier detection.
    IEEE Signal Processing Letters, 25(8), 1236-1240.
    """
    def __init__(
        self, mean_fn, cov_fn, dynamics_covariance,
        alpha, beta, tol_inlier, n_inner
    ):
        super().__init__(mean_fn, cov_fn, dynamics_covariance, n_inner)
        self.alpha0 = float(alpha)
        self.beta0 = float(beta)
        self.tol_inlier = float(tol_inlier)

    def init_bel(self, params, cov=1.0):
        self.rfn, self.mean_fn, init_params = self._initialise_flat_fn(self.mean_fn_og, params)
        self.grad_mean = jax.jacrev(self.mean_fn)
        nparams = len(init_params)
        return OutlierEKFState(
            mean=init_params,
            cov=jnp.eye(nparams) * cov,
            alpha=self.alpha0,
            beta=self.beta0,
            pr_inlier=1.0,
            tau=0.0,
        )

    def _expectation_proba_outlier(self, bel):
        elog_pr = digamma(bel.alpha) - digamma(bel.alpha + bel.beta + 1)   # (29)
        elog_1mpr = digamma(bel.beta + 1) - digamma(bel.alpha + bel.beta + 1)  # (30)
        return elog_pr, elog_1mpr

    def _compute_B_term(self, bel, bel_pred, y, x):
        """Equation (26): B = outer(err, err)."""
        Ht = self.grad_mean(bel_pred.mean, x)
        yhat_c = self.predict_fn(bel_pred, x) + Ht @ (bel.mean - bel_pred.mean)
        err = y - yhat_c
        return jnp.outer(err, err)

    def _update_expectation_inlier(self, bel, bel_pred, y, x):
        """Expectation of the inlier indicator --- eq. (31)."""
        B = self._compute_B_term(bel, bel_pred, y, x)
        Rt = jnp.atleast_2d(self.cov_fn(self.predict_fn(bel_pred, x)))
        Rt_inv = jnp.linalg.inv(Rt)

        elog_pi, elog_1mpi = self._expectation_proba_outlier(bel)
        logpr_inlier = elog_pi - jnp.einsum("ij,ji->", B, Rt_inv) / 2   # (27)
        logpr_outlier = elog_1mpi
        log_norm_cst = -jnp.logaddexp(logpr_inlier, logpr_outlier)  # (28)
        return jnp.exp(logpr_inlier + log_norm_cst)  # (31)

    def _update_with_inlier(self, bel_pred, y, x, e_inlier):
        Ht = self.grad_mean(bel_pred.mean, x)
        Rt = jnp.atleast_2d(self.cov_fn(self.predict_fn(bel_pred, x))) / e_inlier
        St = Ht @ bel_pred.cov @ Ht.T + Rt
        Kt = jnp.linalg.solve(St, Ht @ bel_pred.cov).T
        err = y - self.predict_fn(bel_pred, x)
        mean_update = bel_pred.mean + Kt @ err
        cov_update = bel_pred.cov - Kt @ St @ Kt.T
        return bel_pred.replace(mean=mean_update, cov=cov_update)

    def _inner_update(self, i, bel, bel_pred, y, x):
        e_inlier = bel.pr_inlier
        mean_old = bel.mean
        bel_state = jax.lax.cond(
            e_inlier < self.tol_inlier,
            lambda: bel_pred,
            lambda: self._update_with_inlier(bel_pred, y, x, e_inlier),
        )
        expectation_inlier = self._update_expectation_inlier(bel_state, bel_pred, y, x)
        tau = jnp.linalg.norm(bel_state.mean - mean_old) / jnp.linalg.norm(mean_old)
        return bel_state.replace(
            pr_inlier=expectation_inlier,
            alpha=self.alpha0 + expectation_inlier,
            beta=self.beta0 + 1 - expectation_inlier,
            tau=tau,
        )

    def update(self, bel, y, x):
        bel_pred = bel
        bel = bel_pred.replace(pr_inlier=1.0, tau=1.0)
        _inner = partial(self._inner_update, bel_pred=bel_pred, y=y, x=x)
        return jax.lax.fori_loop(0, self.n_inner, _inner, bel)


class FifoSGDIMQ(FifoSGD):
    def __init__(
        self, apply_fn, tx, buffer_size, dim_features, dim_output, soft_threshold, n_inner=1,
    ):
        super().__init__(apply_fn, self.lossfn, tx, buffer_size, dim_features, dim_output, n_inner)
        self.soft_threshold = soft_threshold

    def lossfn(self, params, counter, x, y, applyfn, weighting_term):
        yhat = applyfn(params, x)
        params_flat = ravel_pytree(params)[0]
        loss = jnp.sum(counter * (y - yhat) ** 2) / counter.sum()
        loss = loss * weighting_term 
        return loss

    def inverse_multi_quad(self, bel, y, x):
        yhat = self.apply_fn(bel.params, x) # prior predictive
        err = y - yhat
        weighting_term = self.soft_threshold ** 2 / (self.soft_threshold ** 2 + jnp.inner(err, err))
        return weighting_term

    @partial(jax.jit, static_argnums=(0,))
    def _train_step(self, bel, weighting_term):
        X, y = bel.buffer_X, bel.buffer_y
        loss, grads = self.loss_grad(bel.params, bel.counter, X, y, bel.apply_fn, weighting_term)
        bel = bel.apply_gradients(grads=grads)
        return loss, bel

    def update_state(self, bel, Xt, yt):
        weighting_term = self.inverse_multi_quad(bel, yt, Xt)
        bel = bel.apply_buffers(Xt, yt)

        def partial_step(_, bel):
            _, bel = self._train_step(bel, weighting_term)
            return bel
        bel = jax.lax.fori_loop(0, self.n_inner - 1, partial_step, bel)
        # Do not count inner steps as part of the outer step
        _, bel = self._train_step(bel, weighting_term)
        return bel

