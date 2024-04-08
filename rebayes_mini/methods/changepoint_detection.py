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
        bel_prior = jax.tree.map(lambda x: x[0], bel_hist)
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

        bel_posterior = jax.tree.map(lambda hist: hist[ix_prev], bel_hist)
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

        bel_previous_single = jax.tree.map(lambda hist: hist[ix_prev], bel_hist)
        bel_posterior_single = self.update_bel_single(y, X, bel_previous_single)

        # update belief state
        bel_hist = jax.tree.map(lambda hist, element: hist.at[ix_update].set(element), bel_hist, bel_posterior_single)
        return bel_hist


    @partial(jax.jit, static_argnums=(0,))
    def update_bel_reset(self, t, ell, y, X, bel_hist):
        ix_update = self.get_ix(t, ell)

        bel_init_single = jax.tree.map(lambda x: x[0], bel_hist)
        bel_hist = jax.tree.map(lambda hist, element: hist.at[ix_update].set(element), bel_hist, bel_init_single)

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
        bel_hist = jax.tree.map(lambda hist, init: hist.at[0].set(init), bel_hist, bel_init)
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
    def update_bel(self, y, X, bel):
        """
        Update belief state (posterior)
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
        log_joint_reset = self.update_log_joint_reset(y, X, bel, bel_prior)
        log_joint_increase = self.update_log_joint_increase(y, X, bel)
        # Expand log-joint
        log_joint = jnp.concatenate([log_joint_reset, log_joint_increase])
        log_joint = jnp.nan_to_num(log_joint, nan=-jnp.inf, neginf=-jnp.inf)
        # reduce to K values --- index 0 is a changepoint
        log_joint, top_indices = jax.lax.top_k(log_joint, k=self.K)
        return log_joint, top_indices


    def update_bel_indices(self, y, X, bel, bel_prior, top_indices):
        """
        Update belief state (posterior) for the chosen indices
        """
        # Update all belief states when a changepoint did not happen
        vmap_update_bel = jax.vmap(self.update_bel, in_axes=(None, None, 0))
        bel = vmap_update_bel(y, X, bel)
        # Increment belief state by adding bel_prior and keeping top_indices
        bel = jax.tree.map(lambda prior, beliefs: jnp.concatenate([prior[None], beliefs]), bel_prior, bel)
        bel = jax.tree.map(lambda param: jnp.take(param, top_indices, axis=0), bel)
        return bel


    def update_runlengths(self, bel, top_indices):
        """
        Update runlengths
        """
        runlengths = bel.runlength + 1
        runlengths = jnp.concatenate([jnp.array([0]), runlengths])
        runlengths = jnp.take(runlengths, top_indices, axis=0)
        bel = bel.replace(runlength=runlengths)
        return bel


    def step(self, y, X, bel, bel_prior, callback_fn):
        """
        Update belief state and log-joint for a single observation
        """
        log_joint, top_indices = self.update_log_joint(y, X, bel, bel_prior)
        bel_posterior = self.update_bel_indices(y, X, bel, bel_prior, top_indices)

        bel_posterior = self.update_runlengths(bel_posterior, top_indices)
        bel_posterior = bel_posterior.replace(log_joint=log_joint)

        out = callback_fn(bel_posterior, bel, y, X, top_indices)

        return bel_posterior, out


    def scan(self, y, X, bel, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        bel_prior = jax.tree.map(lambda x: x[0], bel)
        def _step(bel, yX):
            y, X = yX
            bel, out = self.step(y, X, bel, bel_prior, callback_fn)
            return bel, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist


class BayesianOnlineChangepointHazardDetection(ABC):
    def __init__(self, K, b):
        self.K = K
        self.b = b

    @abstractmethod
    def init_bel(self, y, X, bel_init):
        ...


    @abstractmethod
    def compute_log_posterior_predictive(self, y, X, bel):
        ...


    @abstractmethod
    def update_bel(self, y, X, bel):
        """
        Update belief state (posterior)
        """
        ...

    
    def p_change(self, changepoints, t):
        alpha = changepoints
        beta = t - alpha
        proba = (alpha + 1) / (alpha + beta + 2)
        return proba

    @partial(jax.vmap, in_axes=(None, None, None, None, 0))
    def update_log_joint_increase(self, y, X, t, bel):
        log_p_pred = self.compute_log_posterior_predictive(y, X, bel)
        log_joint = log_p_pred + bel.log_joint + jnp.log(1 - self.p_change(bel.changepoints, t))
        return log_joint

    @partial(jax.vmap, in_axes=(None, None, None, None, 0, None))
    def update_log_joint_reset(self, y, X, t, bel, bel_prior):
        log_p_pred = self.compute_log_posterior_predictive(y, X, bel_prior)
        log_joint = log_p_pred + jax.nn.logsumexp(bel.log_joint) + jnp.log(self.p_change(bel.changepoints, t))
        return log_joint

    def update_runlengths(self, bel):
        """
        Update runlengths and number of changepoints
        """
        runlengths = bel.runlength + 1 # increase runlength by one
        changepoints = bel.changepoints # no change in changepoints

        bel = bel.replace(changepoints=changepoints, runlength=runlengths)
        return bel

    def update_changepoints(self, bel):
        changepoints = bel.changepoints + 1 # increase changepoints by one
        runlengths = bel.runlength * 0
        bel = bel.replace(changepoints=changepoints, runlength=runlengths)
        return bel

    def update_log_joint(self, y, X, t, bel, bel_prior):
        log_joint_reset = self.update_log_joint_reset(y, X, t, bel, bel_prior)
        log_joint_increase = self.update_log_joint_increase(y, X, t, bel)
        # Expand log-joint
        log_joint = jnp.concatenate([log_joint_reset, log_joint_increase])
        log_joint = jnp.nan_to_num(log_joint, nan=-jnp.inf, neginf=-jnp.inf)
        # sekect top K values
        log_joint, top_indices = jax.lax.top_k(log_joint, k=self.K)
        return log_joint, top_indices

    def update_bel_increase(self, y, X, bel, bel_prior):
        bel_update = self.update_bel(y, X, bel)
        bel_update = self.update_runlengths(bel_update)
        return bel_update
    
    def update_bel_reset(self, y, X, bel, bel_prior):
        bel_update = bel_prior
        bel_update = self.update_changepoints(bel_update)
        return bel_update

    def update_bel_batch(self, y, X, bel, bel_prior, is_changepoint):
        vmap_update_bel_increase = jax.vmap(self.update_bel_increase, in_axes=(None, None, 0, None))
        vmap_update_bel_reset = jax.vmap(self.update_bel_reset, in_axes=(None, None, 0, None))

        bel = jax.lax.cond(
            is_changepoint,
            lambda: vmap_update_bel_reset(y, X, bel, bel_prior),
            lambda: vmap_update_bel_increase(y, X, bel, bel_prior),
        )
        return bel

    def step(self, y, X, t, bel, bel_prior, callback_fn):
        """
        Update belief state and log-joint for a single observation
        """
        # From K to 2K belief states
        log_joint, top_indices = self.update_log_joint(y, X, t, bel, bel_prior)

        # Update runlengths and changepoints
        bel_posterior_reset = self.update_bel_batch(y, X, bel, bel_prior, is_changepoint=True)
        bel_posterior_increase = self.update_bel_batch(y, X, bel, bel_prior, is_changepoint=False)
        bel_posterior = jax.tree.map(lambda bchange, bupdate: jnp.concatenate([bchange, bupdate]), bel_posterior_reset, bel_posterior_increase)
        bel_posterior = bel_posterior.replace(log_joint=log_joint)

        # from 2K to K belief states
        bel_posterior = jax.tree.map(lambda param: jnp.take(param, top_indices, axis=0), bel_posterior)

        # callback and finish update
        out = callback_fn(bel_posterior, bel, y, X, top_indices, t)
        return bel_posterior, out

    def scan(self, y, X, bel, callback_fn=None):
        timesteps = jnp.arange(y.shape[0])
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        bel_prior = jax.tree.map(lambda x: x[0], bel)
        def _step(bel, yXt):
            y, X, t = yXt
            bel, out = self.step(y, X, t, bel, bel_prior, callback_fn)
            return bel, out

        bel, hist = jax.lax.scan(_step, bel, (y, X, timesteps))
        return bel, hist


class AdaptiveBayesianOnlineChangepoint(LowMemoryBayesianOnlineChangepoint):
    def __init__(self, p_change, K, shock=0.0):
        super().__init__(p_change, K)
        self.shock = shock

    def step(self, y, X, bel, _, callback_fn):
        """
        Update belief state and log-joint for a single observation
        """
        ix_max = jnp.nanargmax(bel.log_joint)
        bel_prior = jax.tree.map(lambda x: x[ix_max], bel)
        dim = bel_prior.mean.shape[0]
        new_cov = jax.lax.cond(self.shock > 0, lambda S: jnp.eye(dim) / self.shock, lambda S: _.cov, bel_prior.cov)
        new_mean = bel_prior.mean / jnp.sqrt(jnp.linalg.norm(bel_prior.mean))
        bel_prior = bel_prior.replace(
            mean=new_mean,
            cov=new_cov,
            log_joint=_.log_joint, runlength=_.runlength
        )

        log_joint, top_indices = self.update_log_joint(y, X, bel, bel_prior)
        bel_posterior = self.update_bel_indices(y, X, bel, bel_prior, top_indices)

        bel_posterior = self.update_runlengths(bel_posterior, top_indices)
        bel_posterior = bel_posterior.replace(log_joint=log_joint)

        out = callback_fn(bel_posterior, bel, y, X, top_indices)

        return bel_posterior, out


class BernoulliRegimeChange(ABC):
    """
    Bernoulli regime change based on the
    variational beam search (VBS) algorithm
    """
    def __init__(self, p_change, K, shock, inflate_prior_covariance):
        self.p_change = p_change
        self.K = K
        self.shock = shock
        self.inflate_prior_covariance = inflate_prior_covariance


    @abstractmethod
    def init_bel(self, mean, cov, log_weight):
        """
        Initialize belief state
        """
        ...


    @abstractmethod
    def compute_log_posterior_predictive(self, y, X, bel):
        ...


    @abstractmethod
    def update_bel():
        ...

    def predict_cov_changepoint(self, bel):
        cov_previous = bel.cov
        dim = cov_previous.shape[0]
        cov_if_changepoint = jax.lax.cond(
            self.inflate_prior_covariance,
            lambda: cov_previous / self.shock,
            lambda: jnp.eye(dim) / self.shock,
        )
        return cov_if_changepoint


    def predict_bel(self, bel, has_changepoint):
        mean_previous = bel.mean
        cov_previous = bel.cov
        cov_changepoint = self.predict_cov_changepoint(bel)

        cov_pred = jax.lax.cond(
            has_changepoint,
            lambda: cov_changepoint,
            lambda: cov_previous,
        )
        
        bel = bel.replace(cov=cov_pred)
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
        bel_combined = jax.tree.map(lambda x, y: jnp.stack([x, y]), bel_up, bel_down)
        return bel_combined


    def step(self, y, X, bel):
        # from K to 2K belief states
        vmap_split_and_update = jax.vmap(self.split_and_update, in_axes=(None, None, 0))
        bel = vmap_split_and_update(y, X, bel)
        bel = jax.tree.map(lambda x: einops.rearrange(x, "a b ... -> (a b) ..."), bel)
        # from 2K to K belief states — the 'beam search'
        _, top_indices = jax.lax.top_k(bel.log_weight, k=self.K)
        bel = jax.tree.map(lambda x: jnp.take(x, top_indices, axis=0), bel)
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
            out = callback_fn(bel_posterior, bel, y, X)

            return bel_posterior, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist



class KalmanFilterAdaptiveDynamics(ABC):
    def __init__(self, n_inner, ebayes_lr, state_drift, deflate_mean=True):
        self.n_inner = n_inner
        self.ebayes_lr = ebayes_lr # empirical bayes learning rate
        self.state_drift = state_drift
        self.deflate_mean = deflate_mean * 1.0

    @abstractmethod
    def init_bel(self):
        """
        Initialize belief state
        """
        ...

    @abstractmethod
    def log_posterior_predictive(self, eta, y, X, bel):
        ...


    @abstractmethod
    def update_bel(self, y, X, bel):
        ...

    def predict_bel(self, eta, bel):
        gamma = jnp.exp(-eta / 2)
        dim = bel.mean.shape[0]

        mean = (gamma ** self.deflate_mean) * bel.mean
        cov = gamma ** 2 * bel.cov + (1 - gamma ** 2) * jnp.eye(dim) * self.state_drift
        bel = bel.replace(mean=mean, cov=cov)
        return bel


    def step(self, y, X, bel):
        grad_log_predict_density = jax.grad(self.log_posterior_predictive, argnums=0)

        def _inner_pred(i, bel):
            eta = bel.eta
            grad = grad_log_predict_density(eta, y, X, bel)
            eta = eta + self.ebayes_lr * grad
            eta = eta * (eta > 0) # hard threshold
            bel = bel.replace(eta=eta)
            return bel

        bel = jax.lax.fori_loop(0, self.n_inner, _inner_pred, bel)
        bel = self.update_bel(y, X, bel)
        return bel


    def scan(self, y, X, bel, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        def _step(bel, yX):
            y, X = yX
            bel_posterior = self.step(y, X, bel)
            out = callback_fn(bel_posterior, bel, y, X)

            return bel_posterior, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist


class KalmanFilterBetaAdaptiveDynamics(ABC):
    def __init__(self, n_inner, ebayes_lr, state_drift=1.0):
        self.n_inner = n_inner
        self.ebayes_lr = ebayes_lr # empirical bayes learning rate
        self.state_drift = state_drift

    @abstractmethod
    def init_bel(self):
        """
        Initialize belief state
        """
        ...

    @abstractmethod
    def log_posterior_predictive(self, eta, y, X, bel):
        ...


    @abstractmethod
    def update_bel(self, y, X, bel):
        ...

    def predict_bel(self, eta, bel):
        gamma = jax.nn.sigmoid(eta)
        dim = bel.mean.shape[0]

        mean = bel.mean
        cov = gamma * bel.cov + (1 - gamma) * jnp.eye(dim) * self.state_drift
        bel = bel.replace(mean=mean, cov=cov)
        return bel


    def step(self, y, X, bel):
        grad_log_predict_density = jax.grad(self.log_posterior_predictive, argnums=0)

        def _inner_pred(i, bel):
            eta = bel.eta
            grad = grad_log_predict_density(eta, y, X, bel)
            eta = eta + self.ebayes_lr * grad
            bel = bel.replace(eta=eta)
            return bel

        bel = jax.lax.fori_loop(0, self.n_inner, _inner_pred, bel)
        bel = self.update_bel(y, X, bel)
        return bel


    def scan(self, y, X, bel, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        def _step(bel, yX):
            y, X = yX
            bel_posterior = self.step(y, X, bel)
            out = callback_fn(bel_posterior, bel, y, X)

            return bel_posterior, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist


class LinearModelBOCD(LowMemoryBayesianOnlineChangepoint):
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

        bel = jax.tree.map(lambda param_hist, param: param_hist.at[0].set(param), bel, bel_init)

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


    def update_bel(self, y, X, bel):
        cov_previous = bel.cov
        mean_previous = bel.mean

        prec_previous = jnp.linalg.inv(cov_previous)

        prec_posterior = prec_previous + self.beta * jnp.outer(X, X)
        cov_posterior = jnp.linalg.inv(prec_posterior)
        mean_posterior = cov_posterior @ (prec_previous @ mean_previous + self.beta * X * y)

        bel = bel.replace(mean=mean_posterior, cov=cov_posterior)
        return bel
    

class LinearModelABOCD(AdaptiveBayesianOnlineChangepoint):
    def __init__(self, p_change, K, shock, beta):
        super().__init__(p_change, K, shock)
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

        bel = jax.tree.map(lambda param_hist, param: param_hist.at[0].set(param), bel, bel_init)

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


    def update_bel(self, y, X, bel):
        cov_previous = bel.cov
        mean_previous = bel.mean

        prec_previous = jnp.linalg.inv(cov_previous)

        prec_posterior = prec_previous + self.beta * jnp.outer(X, X)
        cov_posterior = jnp.linalg.inv(prec_posterior)
        mean_posterior = cov_posterior @ (prec_previous @ mean_previous + self.beta * X * y)

        bel = bel.replace(mean=mean_posterior, cov=cov_posterior)
        return bel


class LinearModelBRC(BernoulliRegimeChange):
    def __init__(self, p_change, K, beta, shock, inflate_prior_covariance):
        """
        Bernoulli regime change with adaptive dynamics
        Parameters
        ----------
        p_change : float
            Probability of change
        K : int
            Number of belief states in buffer
        beta : float
            measurement variance
        shock : float
            shock parameter to the covariance matrix on a changepoint
        inflate_prior_covariance : bool
            inflate prior covariance on changepoint or default to identity times 1 / shock
        """
        super().__init__(p_change, K, shock, inflate_prior_covariance)
        self.beta = beta


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

        bel = jax.tree.map(lambda param_hist, param: param_hist.at[0].set(param), bel, bel_init)

        return bel


    def update_bel(self, y, X, bel, has_changepoint):
        bel = self.predict_bel(bel, has_changepoint)
        cov_previous = bel.cov
        mean_previous = bel.mean

        prec_previous = jnp.linalg.inv(cov_previous)

        prec_posterior = prec_previous + self.beta * jnp.outer(X, X)
        cov_posterior = jnp.linalg.inv(prec_posterior)
        mean_posterior = cov_posterior @ (prec_previous @ mean_previous + self.beta * X * y)

        bel = bel.replace(
            mean=mean_posterior,
            cov=cov_posterior,
            segment=bel.segment + has_changepoint
        )
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



class LinearModelKFA(KalmanFilterAdaptiveDynamics):
    def __init__(self, n_inner, ebayes_lr, beta, state_drift, deflate_mean=True):
        super().__init__(n_inner, ebayes_lr, state_drift, deflate_mean)
        self.beta = 1/beta # variance to precision

    def init_bel(self, mean, cov, eta=0.0):
        """
        Initialize belief state
        """
        bel = states.GammaFilterState(
            mean=mean,
            cov=cov,
            eta=eta
        )
        return bel


    def log_posterior_predictive(self, eta, y, X, bel):
        bel = self.predict_bel(eta, bel)
        mean = bel.mean @ X
        cov = X.T @ bel.cov @ X + self.beta
        log_p_pred = distrax.Normal(mean, cov).log_prob(y)
        return log_p_pred


    def update_bel(self, y, X, bel):
        bel_pred = self.predict_bel(bel.eta, bel)
        Kt = bel_pred.cov @ X / (X.T @ bel_pred.cov @ X + self.beta)

        mean = bel_pred.mean + Kt * (y - X.T @ bel_pred.mean)
        cov = bel_pred.cov - Kt * X.T @ bel_pred.cov
        bel = bel.replace(mean=mean, cov=cov)
        return bel


class LinearModelKFBA(KalmanFilterBetaAdaptiveDynamics):
    def __init__(self, n_inner, ebayes_lr, beta, state_drift, a, b):
        super().__init__(n_inner, ebayes_lr, state_drift)
        self.beta = 1/beta # variance to precision
        self.a = a
        self.b = b

    def init_bel(self, mean, cov, eta=0.0):
        """
        Initialize belief state
        """
        bel = states.GammaFilterState(
            mean=mean,
            cov=cov,
            eta=eta
        )
        return bel


    def log_posterior_predictive(self, eta, y, X, bel):
        bel = self.predict_bel(eta, bel)
        mean = bel.mean @ X
        cov = X.T @ bel.cov @ X + self.beta
        log_p_pred = distrax.Normal(mean, cov).log_prob(y)
        gamma = jax.nn.sigmoid(eta)
        log_p_pred = log_p_pred + distrax.Beta(self.a, self.b).log_prob(gamma)
        return log_p_pred


    def update_bel(self, y, X, bel):
        bel_pred = self.predict_bel(bel.eta, bel)
        Kt = bel_pred.cov @ X / (X.T @ bel_pred.cov @ X + self.beta)

        mean = bel_pred.mean + Kt * (y - X.T @ bel_pred.mean)
        cov = bel_pred.cov - Kt * X.T @ bel_pred.cov
        bel = bel.replace(mean=mean, cov=cov)
        return bel


class LinearModelBOCHD(BayesianOnlineChangepointHazardDetection):
    def __init__(self, K, b, beta):
        super().__init__(K, b)
        self.beta = beta


    def init_bel(self, mean, cov, log_weight):
        """
        Initialize belief state
        """
        d, *_ = mean.shape
        bel = states.BOCHDGaussState(
            mean=jnp.zeros((self.K, d)),
            cov=jnp.zeros((self.K, d, d)),
            log_joint=jnp.ones((self.K,)) * -jnp.inf,
            runlength=jnp.zeros(self.K),
            changepoints=jnp.zeros(self.K)
        )

        bel_init = states.BOCHDGaussState(
            mean=mean,
            cov=cov,
            log_joint=log_weight,
            runlength=jnp.array(0),
            changepoints=jnp.array(0)
        )

        bel = jax.tree.map(lambda param_hist, param: param_hist.at[0].set(param), bel, bel_init)

        return bel


    def compute_log_posterior_predictive(self, y, X, bel):
        """
        Compute log-posterior predictive for a Gaussian with known variance
        """
        mean = bel.mean @ X
        scale = 1 / self.beta + X @  bel.cov @ X
        log_p_pred = distrax.Normal(mean, scale).log_prob(y)
        return log_p_pred


    def update_bel(self, y, X, bel):
        cov_previous = bel.cov
        mean_previous = bel.mean

        prec_previous = jnp.linalg.inv(cov_previous)

        prec_posterior = prec_previous + self.beta * jnp.outer(X, X)
        cov_posterior = jnp.linalg.inv(prec_posterior)
        mean_posterior = cov_posterior @ (prec_previous @ mean_previous + self.beta * X * y)

        bel = bel.replace(mean=mean_posterior, cov=cov_posterior)
        return bel

