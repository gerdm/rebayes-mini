import jax
import distrax
import jax.numpy as jnp
from tqdm import tqdm
from abc import ABC, abstractmethod
from functools import partial
from rebayes_mini.states import GaussState


class BayesianOnlineChangepointDetection(ABC):
    def __init__(self, p_change):
        self.p_change = p_change

    @abstractmethod
    def init_bel(self, y_hist, X_hist, bel_init, size_filter):
        ...


    @abstractmethod
    def update_bel_single(self, y, X, bel_prev):
        ...


    @abstractmethod
    def compute_log_posterior_predictive(self, y, X, bel):
        """
        Compute log-posterior predictive for a Gaussian with known variance
        """
        ...


    @partial(jax.jit, static_argnums=(0,))
    def get_ix(self, t, ell):
        # Number of steps to get to first observation at time t
        ix_step = t * (t + 1) // 2
        # Increase runlength
        ix = ix_step + ell
        return ix


    def update_log_joint_reset(self, t, ell, y, X, bel_hist, log_joint_hist):
        bel_prior = jax.tree_map(lambda x: x[0], bel_hist)
        log_p_pred = self.compute_log_posterior_predictive(y, X, bel_prior)

        if t == 0:
            log_joint = log_p_pred + jnp.log(self.p_change)
        else:
            ix_start = self.get_ix(t-1, 0)
            ix_end = self.get_ix(t-1, (t-1) + 1)
            log_joint = log_p_pred + jax.nn.logsumexp(log_joint_hist[ix_start:ix_end] + jnp.log(self.p_change))

        ix = self.get_ix(t, ell)
        log_joint_hist = log_joint_hist.at[ix].set(log_joint.squeeze())
        return log_joint_hist


    @partial(jax.jit, static_argnums=(0,))
    def update_log_joint_increase(self, t, ell, y, X, bel_hist, log_joint_hist):
        ix_update = self.get_ix(t, ell)
        ix_prev = self.get_ix(t-1, ell-1)

        bel_posterior = jax.tree_map(lambda hist: hist[ix_prev], bel_hist)
        log_p_pred = self.compute_log_posterior_predictive(y, X, bel_posterior)

        log_joint = log_p_pred + log_joint_hist[ix_prev] + jnp.log(1 - self.p_change)
        log_joint = log_joint.squeeze()
        return log_joint_hist.at[ix_update].set(log_joint)


    def update_log_joint(self, t, ell, y, X, bel_hist, log_joint_hist):
        if ell == 0:
            log_joint_hist = self.update_log_joint_reset(t, ell, y, X, bel_hist, log_joint_hist)
        else:
            log_joint_hist = self.update_log_joint_increase(t, ell, y, X, bel_hist, log_joint_hist)
        return log_joint_hist



    @partial(jax.jit, static_argnums=(0,))
    def update_bel_increase(self, t, ell, y, X, bel_hist):
        ix_prev = self.get_ix(t-1, ell-1)
        ix_update = self.get_ix(t, ell)

        bel_previous_single = jax.tree_map(lambda hist: hist[ix_prev], bel_hist)
        bel_posterior_single = self.update_bel_single(y, X, bel_previous_single)

        # update belief state
        bel_hist = jax.tree_map(lambda hist, element: hist.at[ix_update].set(element), bel_hist, bel_posterior_single)
        return bel_hist


    @partial(jax.jit, static_argnums=(0,))
    def update_bel_reset(self, t, ell, y, X, bel_hist):
        ix_update = self.get_ix(t, ell)

        bel_init_single = jax.tree_map(lambda x: x[0], bel_hist)
        bel_hist = jax.tree_map(lambda hist, element: hist.at[ix_update].set(element), bel_hist, bel_init_single)

        return bel_hist


    @partial(jax.jit, static_argnums=(0,))
    def update_bel(self, t, ell, y, X, bel_hist):
        """
        Update belief state (posterior) for a given runlength
        """
        params = t, ell, y, X, bel_hist
        bel_hist = jax.lax.cond(
            ell == 0,
            self.update_bel_reset,
            self.update_bel_increase,
            *params
        )

        return bel_hist


    def update_log_posterior(self, t, runlength_log_posterior, marginal, log_joint_hist):
        ix_init = self.get_ix(t, 0)
        ix_end = self.get_ix(t, t+1)

        section = slice(ix_init, ix_end, 1)
        log_joint_sub = log_joint_hist[section]

        marginal_t = jax.nn.logsumexp(log_joint_sub)
        marginal = marginal.at[t].set(marginal_t)

        update = log_joint_sub - marginal_t
        runlength_log_posterior = runlength_log_posterior.at[section].set(update)

        return marginal, runlength_log_posterior


    def scan(self, y, X, bel_prior):
        """
        Bayesian online changepoint detection (BOCD)
        discrete filter
        """
        n_samples, d = X.shape

        size_filter = n_samples * (n_samples + 1) // 2
        marginal = jnp.zeros(n_samples)
        log_joint = jnp.zeros((size_filter,))
        log_cond = jnp.zeros((size_filter,))
        bel_hist = self.init_bel(y, X, bel_prior, size_filter)

        for t in tqdm(range(n_samples)):
            tix = jnp.maximum(0, t-1)
            xt = X[tix].squeeze()
            yt = y[tix].squeeze()

            # Compute log-joint
            for ell in range(t+1):
                log_joint = self.update_log_joint(t, ell, yt, xt, bel_hist, log_joint)

            # compute runlength log-posterior
            marginal, log_cond = self.update_log_posterior(t, log_cond, marginal, log_joint)

            # Update posterior parameters
            for ell in range(t+1):
                if t == 0: # TODO: refactor
                    continue
                bel_hist = self.update_bel(t, ell, yt, xt, bel_hist)

        out = {
            "log_joint": log_joint,
            "log_runlength_posterior": log_cond,
            "marginal": marginal,
            "bel": bel_hist,
        }

        return out


class LM_BOCD(BayesianOnlineChangepointDetection):
    """
    LM-BOCD: Bayesian Online Changepoint Detection for linear model
    with known measurement variance
    For a run with T data points and paramter dimension D,
    the algorithm has memory requirement of O(T * (T + 1) / 2 * D ^ 2)
    """
    def __init__(self, p_change, beta):
        super().__init__(p_change)
        self.beta = beta


    def init_bel(self, y_hist, X_hist, bel_init, size_filter):
        _, d = X_hist.shape
        hist_mean = jnp.zeros((size_filter, d))
        hist_cov = jnp.zeros((size_filter, d, d))

        bel_hist = GaussState(mean=hist_mean, cov=hist_cov)
        bel_hist = jax.tree_map(lambda hist, init: hist.at[0].set(init), bel_hist, bel_init)
        return bel_hist


    @partial(jax.jit, static_argnums=(0,))
    def update_bel_single(self, y, X, bel_prev):
        cov_previous = bel_prev.cov
        mean_previous = bel_prev.mean

        prec_previous = jnp.linalg.inv(cov_previous)

        prec_posterior = prec_previous + self.beta * jnp.outer(X, X)
        cov_posterior = jnp.linalg.inv(prec_posterior)
        mean_posterior = cov_posterior @ (prec_previous @ mean_previous + self.beta * X * y)

        bel = GaussState(mean=mean_posterior, cov=cov_posterior)
        return bel


    @partial(jax.jit, static_argnums=(0,))
    def compute_log_posterior_predictive(self, y, X, bel):
        """
        Compute log-posterior predictive for a Gaussian with known variance
        """
        mean = bel.mean @ X
        scale = 1 / self.beta + X @  bel.cov @ X
        log_p_pred = distrax.Normal(mean, scale).log_prob(y)
        return log_p_pred


class WLLM_BOCD(LM_BOCD):
    """
    Weighted-likelihood LM-BOCD
    """
    def __init__(self, p_change, beta, c):
        super().__init__(p_change, beta)
        self.c = c

    @partial(jax.jit, static_argnums=(0,))
    def imq_kernel(self, residual):
        """
        Inverse multi-quadratic kernel
        """
        return 1 / jnp.sqrt(1 + residual ** 2 / self.c ** 2)


    @partial(jax.jit, static_argnums=(0,))
    def update_bel_single(self, y, X, bel):
        cov_previous = bel.cov
        mean_previous = bel.mean

        prec_previous = jnp.linalg.inv(cov_previous)

        Wt = self.imq_kernel(y - X @ mean_previous)
        prec_posterior = prec_previous + Wt ** 2 * self.beta * jnp.outer(X, X)
        cov_posterior = jnp.linalg.inv(prec_posterior)
        mean_posterior = cov_posterior @ (prec_previous @ mean_previous + Wt ** 2 * self.beta * X * y)

        bel = GaussState(mean=mean_posterior, cov=cov_posterior)
        return bel


    @partial(jax.jit, static_argnums=(0,))
    def compute_log_posterior_predictive(self, y, X, bel):
        """
        Compute log-posterior predictive for a Gaussian with known variance
        """
        mean = bel.mean @ X
        residual = y - mean
        Wt = self.imq_kernel(residual)

        scale = 1 / (self.beta * Wt ** 2) + X @ bel.cov @ X
        log_p_pred = distrax.Normal(mean, scale).log_prob(y)
        return log_p_pred


class AWLLM_BOCD(LM_BOCD):
    """
    Adaptive, Weighted-likelihood LM-BOCD
    """
    def __init__(self, p_change, beta, c, shock_val):
        super().__init__(p_change, beta)
        self.c = c
        self.shock_val = shock_val

    @partial(jax.jit, static_argnums=(0,))
    def imq_kernel(self, residual):
        """
        Inverse multi-quadratic kernel
        """
        return 1 / jnp.sqrt(1 + residual ** 2 / self.c ** 2)


    @partial(jax.jit, static_argnums=(0,))
    def update_bel_single(self, y, X, bel):
        cov_previous = bel.cov
        mean_previous = bel.mean

        prec_previous = jnp.linalg.inv(cov_previous)

        Wt = self.imq_kernel(y - X @ mean_previous)
        prec_posterior = prec_previous + Wt ** 2 * self.beta * jnp.outer(X, X)
        cov_posterior = jnp.linalg.inv(prec_posterior)
        mean_posterior = cov_posterior @ (prec_previous @ mean_previous + Wt ** 2 * self.beta * X * y)

        bel = GaussState(mean=mean_posterior, cov=cov_posterior)
        return bel


    @partial(jax.jit, static_argnums=(0,))
    def compute_log_posterior_predictive(self, y, X, bel):
        """
        Compute log-posterior predictive for a Gaussian with known variance
        """
        mean = bel.mean @ X
        residual = y - mean
        Wt = self.imq_kernel(residual)

        scale = 1 / (self.beta * Wt ** 2) + X @ bel.cov @ X
        log_p_pred = distrax.Normal(mean, scale).log_prob(y)
        return log_p_pred

    @partial(jax.jit, static_argnums=(0,))
    def update_bel_reset(self, t, ell, y, X, bel_hist):
        shift = jnp.maximum(0, t - 1)
        ix_prev = self.get_ix(t - 1, shift)
        ix_update = self.get_ix(t, ell)

        bel_previous_single = jax.tree_map(lambda hist: hist[ix_prev], bel_hist)
        prev_mean = bel_previous_single.mean
        prev_cov = bel_previous_single.cov / self.shock_val
        bel_reset = GaussState(mean=prev_mean, cov=prev_cov)
        bel_posterior = self.update_bel_single(y, X, bel_reset)

        # update belief state
        bel_hist = jax.tree_map(lambda hist, element: hist.at[ix_update].set(element), bel_hist, bel_posterior)
        return bel_hist

    def update_log_joint_reset(self, t, ell, y, X, bel_hist, log_joint_hist):
        ix_prev = self.get_ix(t - 1, t - 1)
        bel_prior = jax.tree_map(lambda x: x[ix_prev], bel_hist)
        log_p_pred = self.compute_log_posterior_predictive(y, X, bel_prior)

        if t == 0:
            log_joint = log_p_pred + jnp.log(self.p_change)
        else:
            ix_start = self.get_ix(t-1, 0)
            ix_end = self.get_ix(t-1, (t-1) + 1)
            log_joint = log_p_pred + jax.nn.logsumexp(log_joint_hist[ix_start:ix_end] + jnp.log(self.p_change))

        ix = self.get_ix(t, ell)
        log_joint_hist = log_joint_hist.at[ix].set(log_joint.squeeze())
        return log_joint_hist


class LowMemoryBayesianOnlineChangepoint(ABC):
    def __init__(self, p_change, K, beta):
        self.p_change = p_change
        self.K = K
        self.beta = beta

    @partial(jax.vmap, in_axes=(None, None, None, 0))
    def update_log_joint_increase(self, y, X, bel):
        log_p_pred = self.compute_log_posterior_predictive(y, X, bel)
        log_joint = log_p_pred + bel.log_joint + jnp.log(1 - self.p_change)
        return log_joint

    def update_log_joint_reset(self, y, X, bel, bel_prior):
        log_p_pred = self.compute_log_posterior_predictive(y, X, bel_prior)
        log_joint = log_p_pred + jax.nn.logsumexp(bel.log_joint) + jnp.log(self.p_change)
        return jnp.atleast_1d(log_joint)
    
    # @partial(jax.jit, static_argnums=(0,))
    def update_log_joint(self, y, X, bel, bel_prior):
        log_joint_increase = self.update_log_joint_increase(y, X, bel)
        log_joint_reset = self.update_log_joint_reset(y, X, bel, bel_prior)
        # Expand log-joint
        log_joint = jnp.concatenate([log_joint_reset, log_joint_increase])
        log_joint = jnp.nan_to_num(log_joint, nan=-jnp.inf, neginf=-jnp.inf)
        # reduce to K values --- index 0 is a changepoint
        log_joint, top_indices = jax.lax.top_k(log_joint, k=self.K)
        return log_joint, top_indices


    # @partial(jax.jit, static_argnums=(0,))
    def update_bel(self, y, X, bel, bel_prior, top_indices):
        """
        Update belief state (posterior) for a given runlength
        """
        # Update all belief states
        bel = self.update_bel_single(y, X, bel)
        # Increment belief state by adding bel_prior and keeping top_indices
        bel = jax.tree_map(lambda beliefs, prior: jnp.concatenate([prior[None, ...], beliefs]), bel, bel_prior)
        bel = jax.tree_map(lambda param: jnp.take(param, top_indices, axis=0), bel)
        return bel
    
    def step(self, y, X, bel, bel_prior):
        """
        Update belief state and log-joint for a single observation
        """
        log_joint, top_indices = self.update_log_joint(y, X, bel, bel_prior)
        bel = self.update_bel(y, X, bel, bel_prior, top_indices)
        bel = bel.replace(log_joint=log_joint)
        return bel, top_indices



    @partial(jax.jit, static_argnums=(0,))
    def compute_log_posterior_predictive(self, y, X, bel):
        """
        Compute log-posterior predictive for a Gaussian with known variance
        TODO: remove after refactor
        """
        mean = bel.mean @ X
        scale = 1 / self.beta + X @  bel.cov @ X
        log_p_pred = distrax.Normal(mean, scale).log_prob(y)
        return log_p_pred

    @partial(jax.vmap, in_axes=(None, None, None, 0))
    def update_bel_single(self, y, X, bel):
        cov_previous = bel.cov
        mean_previous = bel.mean

        prec_previous = jnp.linalg.inv(cov_previous)

        prec_posterior = prec_previous + self.beta * jnp.outer(X, X)
        cov_posterior = jnp.linalg.inv(prec_posterior)
        mean_posterior = cov_posterior @ (prec_previous @ mean_previous + self.beta * X * y)

        bel = bel.replace(mean=mean_posterior, cov=cov_posterior)
        return bel