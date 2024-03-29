import jax
import einops
import distrax
import jax.numpy as jnp
from tqdm import tqdm
from abc import ABC, abstractmethod
from functools import partial
from rebayes_mini import states
from rebayes_mini import callbacks


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

        bel_hist = states.GaussState(mean=hist_mean, cov=hist_cov)
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

        bel = states.GaussState(mean=mean_posterior, cov=cov_posterior)
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

        bel = states.GaussState(mean=mean_posterior, cov=cov_posterior)
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

        bel = states.GaussState(mean=mean_posterior, cov=cov_posterior)
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
        bel_reset = states.GaussState(mean=prev_mean, cov=prev_cov)
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
    def __init__(self, p_change, K):
        self.p_change = p_change
        self.K = K


    @abstractmethod
    def init_bel(self, y, X, bel_init):
        ...


    @abstractmethod
    def compute_log_posterior_predictive(self, y, X, bel):
        ...


    @abstractmethod
    def vmap_update_bel(self, y, X, bel):
        """
        Vmap over bel state. y and X are single observations
        """
        ...


    @partial(jax.vmap, in_axes=(None, None, None, 0))
    def update_log_joint_increase(self, y, X, bel):
        log_p_pred = self.compute_log_posterior_predictive(y, X, bel)
        log_joint = log_p_pred + bel.log_joint + jnp.log(1 - self.p_change)
        return log_joint


    def update_log_joint_reset(self, y, X, bel, bel_prior):
        log_p_pred = self.compute_log_posterior_predictive(y, X, bel_prior)
        log_joint = log_p_pred + jax.nn.logsumexp(bel.log_joint) + jnp.log(self.p_change)
        return jnp.atleast_1d(log_joint)


    def update_log_joint(self, y, X, bel, bel_prior):
        log_joint_increase = self.update_log_joint_increase(y, X, bel)
        log_joint_reset = self.update_log_joint_reset(y, X, bel, bel_prior)
        # Expand log-joint
        log_joint = jnp.concatenate([log_joint_reset, log_joint_increase])
        log_joint = jnp.nan_to_num(log_joint, nan=-jnp.inf, neginf=-jnp.inf)
        # reduce to K values --- index 0 is a changepoint
        log_joint, top_indices = jax.lax.top_k(log_joint, k=self.K)
        return log_joint, top_indices


    def update_bel(self, y, X, bel, bel_prior, top_indices):
        """
        Update belief state (posterior) for a given runlength
        """
        # Update all belief states
        bel = self.vmap_update_bel(y, X, bel)
        # Increment belief state by adding bel_prior and keeping top_indices
        bel = jax.tree_map(lambda beliefs, prior: jnp.concatenate([prior[None], beliefs]), bel, bel_prior)
        bel = jax.tree_map(lambda param: jnp.take(param, top_indices, axis=0), bel)
        return bel


    def update_runlengths(self, bel, top_indices):
        """
        Update runlengths
        """
        runlengths = bel.runlength + 1
        runlengths = jnp.concatenate([jnp.array([0]), runlengths])
        runlengths = jnp.take(runlengths, top_indices, axis=0)
        return runlengths


    def step(self, y, X, bel, bel_prior, callback_fn):
        """
        Update belief state and log-joint for a single observation
        """
        log_joint, top_indices = self.update_log_joint(y, X, bel, bel_prior)
        bel = self.update_bel(y, X, bel, bel_prior, top_indices)
        runlengths = self.update_runlengths(bel, top_indices)
        bel = bel.replace(log_joint=log_joint, runlength=runlengths)

        out = callback_fn(bel, bel_prior, y, X, top_indices)

        return bel, out


    def scan(self, y, X, bel, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        bel_prior = jax.tree_map(lambda x: x[0], bel)
        def _step(bel, yX):
            y, X = yX
            bel, out = self.step(y, X, bel, bel_prior, callback_fn)
            return bel, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist


class LM_LMBOCD(LowMemoryBayesianOnlineChangepoint):
    """
    Low-memory LM-BOCD
    """
    def __init__(self, p_change, K, beta):
        super().__init__(p_change, K)
        self.beta = beta


    def init_bel(self, mean, cov, log_joint_init):
        """
        Initialize belief state
        """
        d, *_ = mean.shape
        bel = states.BOCDGaussState(
            mean=jnp.zeros((self.K, d)),
            cov=jnp.zeros((self.K, d, d)),
            log_joint=jnp.ones((self.K,)) * -jnp.inf,
            runlength=jnp.zeros(self.K)
        )

        bel_init = states.BOCDGaussState(
            mean=mean,
            cov=cov,
            log_joint=log_joint_init,
            runlength=jnp.array(0)
        )

        bel = jax.tree_map(lambda param_hist, param: param_hist.at[0].set(param), bel, bel_init)

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


    @partial(jax.vmap, in_axes=(None, None, None, 0))
    def vmap_update_bel(self, y, X, bel):
        cov_previous = bel.cov
        mean_previous = bel.mean

        prec_previous = jnp.linalg.inv(cov_previous)

        prec_posterior = prec_previous + self.beta * jnp.outer(X, X)
        cov_posterior = jnp.linalg.inv(prec_posterior)
        mean_posterior = cov_posterior @ (prec_previous @ mean_previous + self.beta * X * y)

        bel = bel.replace(mean=mean_posterior, cov=cov_posterior)
        return bel


class BernoulliRegimeChange:
    """
    Bernoulli regime change based on the
    variational beam search (VBS) algorithm
    TODO: split into base class and LM (linear model) class.
    """
    def __init__(self, p_change, K, beta, shock):
        self.p_change = p_change
        self.K = K
        self.beta = beta
        self.shock = shock

    def init_bel(self, mean, cov, log_weight):
        """
        Initialize belief state
        """
        d, *_ = mean.shape
        bel = states.BernoullChangeGaussState(
            mean=jnp.zeros((self.K, d)),
            cov=jnp.zeros((self.K, d, d)),
            log_weight=jnp.ones((self.K,)) * -jnp.inf,
            segment=jnp.zeros(self.K)
        )

        bel_init = states.BernoullChangeGaussState(
            mean=mean,
            cov=cov,
            log_weight=log_weight,
            segment=jnp.array(0)
        )

        bel = jax.tree_map(lambda param_hist, param: param_hist.at[0].set(param), bel, bel_init)

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

    def update_bel(self, y, X, bel, has_changepoint):
        cov_previous = bel.cov
        mean_previous = bel.mean

        shock = self.shock ** has_changepoint
        prec_previous = shock * jnp.linalg.inv(cov_previous)

        prec_posterior = prec_previous + self.beta * jnp.outer(X, X)
        cov_posterior = jnp.linalg.inv(prec_posterior)
        mean_posterior = cov_posterior @ (prec_previous @ mean_previous + self.beta * X * y)

        bel = bel.replace(
            mean=mean_posterior,
            cov=cov_posterior,
            segment=bel.segment + has_changepoint
        )
        return bel

    def split_and_update(self, y, X, bel):
        """
        Update belief state and log-joint for a single observation
        """
        bel_up = self.update_bel(y, X, bel, has_changepoint=1.0)
        bel_down = self.update_bel(y, X, bel, has_changepoint=0.0)

        log_pp_up = self.compute_log_posterior_predictive(y, X, bel_up)
        log_pp_down = self.compute_log_posterior_predictive(y, X, bel_down)

        log_odds = log_pp_up - log_pp_down
        log_odds = log_odds + jnp.log(self.p_change / (1 - self.p_change))

        log_prob_up_conditional = -jnp.log1p(jnp.exp(-log_odds))
        log_prob_down_conditional = -log_odds - jnp.log1p(jnp.exp(-log_odds))

        # Compute log-hypothesis weight
        log_weight_up = bel_up.log_weight + log_prob_up_conditional
        log_weight_down = bel_down.log_weight + log_prob_down_conditional
        log_weight_up = jnp.nan_to_num(log_weight_up, nan=-jnp.inf, neginf=-jnp.inf)
        log_weight_down = jnp.nan_to_num(log_weight_down, nan=-jnp.inf, neginf=-jnp.inf)
        # update
        bel_up = bel_up.replace(log_weight=log_weight_up)
        bel_down = bel_down.replace(log_weight=log_weight_down)

        # Combine
        bel_combined = jax.tree_map(lambda x, y: jnp.stack([x, y]), bel_up, bel_down)
        return bel_combined


    def step(self, y, X, bel):
        # from K to 2K belief states
        vmap_split_and_update = jax.vmap(self.split_and_update, in_axes=(None, None, 0))
        bel = vmap_split_and_update(y, X, bel)
        bel = jax.tree_map(lambda x: einops.rearrange(x, "a b ... -> (a b) ..."), bel)
        # from 2K to K belief states — the 'beam search'
        _, top_indices = jax.lax.top_k(bel.log_weight, k=self.K)
        bel = jax.tree_map(lambda x: jnp.take(x, top_indices, axis=0), bel)
        # renormalise weights
        log_weights = bel.log_weight - jax.nn.logsumexp(bel.log_weight)
        log_weights = jnp.nan_to_num(log_weights, nan=-jnp.inf, neginf=-jnp.inf)

        bel = bel.replace(log_weight=log_weights)

        return bel

    def scan(self, y, X, bel, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        def _step(bel, yX):
            y, X = yX
            bel_posterior = self.step(y, X, bel)
            out = callback_fn(y, X, bel_posterior, bel)

            return bel_posterior, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist


class GammaFilter:
    def __init__(self, n_inner, ebayes_lr, beta, state_drift):
        self.n_inner = n_inner
        self.ebayes_lr = ebayes_lr # empirical bayes learning rate
        self.beta = beta
        self.state_drift = state_drift
    
    def predict_bel(self, eta, bel):
        gamma = jnp.exp(-eta / 2)
        dim = bel.mean.shape[0]

        mean = gamma * bel.mean
        cov = gamma ** 2 * bel.cov + (1 - gamma ** 2) * jnp.eye(dim) * self.beta
        bel = bel.replace(mean=mean, cov=cov)
        return bel

    def log_predict_density(self, eta, y, X, bel):
        bel = self.predict_bel(eta, bel)
        mean = bel.mean @ X
        cov = X.T @ bel.cov @ X + self.beta
        log_p_pred = distrax.Normal(mean, cov).log_prob(y)
        return log_p_pred
    
    def update_bel(self, eta, y, X, bel):
        bel_pred = self.predict_bel(eta, bel)
        Kt = bel_pred.cov @ X / (X.T @ bel_pred.cov @ X + self.beta)

        mean = bel_pred.mean + Kt * (y - X.T @ bel_pred.mean)
        cov = bel_pred.cov - Kt * X.T @ bel_pred.cov
        bel = bel.replace(mean=mean, cov=cov)
    
    def step(self, y, X, bel):
        grad_log_predict_density = jax.grad(self.log_predict_density, argnums=0)

        def _inner_pred(i, bel):
            grad = grad_log_predict_density(eta, y, X, bel)
            eta = eta + self.ebayes_lr * grad
            bel = bel.replace(eta=eta)
            return bel
        
        bel = jax.lax.fori_loop(0, self.n_inner, _inner_pred, bel)
        bel = self.update_bel(bel.eta, y, X, bel)
        