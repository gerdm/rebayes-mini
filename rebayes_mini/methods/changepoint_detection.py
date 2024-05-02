import jax
import einops
import distrax
import jax.numpy as jnp
from tqdm import tqdm
from abc import ABC, abstractmethod
from functools import partial
from rebayes_mini import states
from rebayes_mini import callbacks


class FullMemoryBayesianOnlineChangepointDetection(ABC):
    def __init__(self, p_change):
        self.p_change = p_change

    @abstractmethod
    def init_bel(self, y_hist, X_hist, bel_init, size_filter):
        ...


    @abstractmethod
    def update_bel_single(self, y, X, bel_prev):
        ...


    @abstractmethod
    def log_predictive_density(self, y, X, bel):
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
        log_p_pred = self.log_predictive_density(y, X, bel_prior)

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
        log_p_pred = self.log_predictive_density(y, X, bel_posterior)

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


    def scan(self, y, X, bel_hist):
        """
        Bayesian online changepoint detection (BOCD)
        discrete filter
        """
        n_samples, d = X.shape

        size_filter = n_samples * (n_samples + 1) // 2
        marginal = jnp.zeros(n_samples)
        log_joint = jnp.zeros((size_filter,))
        log_cond = jnp.zeros((size_filter,))
        # bel_hist = self.init_bel(y, X, bel_prior, size_filter)

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


class Runlength(ABC):
    def __init__(self, p_change, K):
        self.p_change = p_change
        self.K = K


    @abstractmethod
    def init_bel(self, y, X, bel_init):
        ...


    @abstractmethod
    def log_predictive_density(self, y, X, bel):
        ...


    @abstractmethod
    def update_bel(self, y, X, bel):
        """
        Update belief state (posterior)
        """
        ...


    @partial(jax.vmap, in_axes=(None, None, None, 0))
    def update_log_joint_increase(self, y, X, bel):
        log_p_pred = self.log_predictive_density(y, X, bel)
        log_joint = log_p_pred + bel.log_joint + jnp.log(1 - self.p_change)
        return log_joint


    def update_log_joint_reset(self, y, X, bel, bel_prior):
        log_p_pred = self.log_predictive_density(y, X, bel_prior)
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
        bel = jax.tree.map(lambda prior, updates: jnp.concatenate([prior[None], updates]), bel_prior, bel)
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


class RunlengthSoftReset(Runlength):
    def __init__(self, p_change, K, shock, deflate_mean):
        super().__init__(p_change, K)
        self.shock = shock
        self.deflate_mean = deflate_mean * 1.0

    def update_log_joint(self, y, X, bel, bel_prior):
        log_joint_reset = self.update_log_joint_reset(y, X, bel, bel_prior)
        log_joint_increase = self.update_log_joint_increase(y, X, bel)
        # Expand log-joint
        log_joint = jnp.concatenate([log_joint_reset, log_joint_increase])
        log_joint = jnp.nan_to_num(log_joint, nan=-jnp.inf, neginf=-jnp.inf)
        # Compute log-posterior before reducing
        log_posterior = log_joint - jax.nn.logsumexp(log_joint)
        # reduce to K values --- index 0 is a changepoint
        log_joint, top_indices = jax.lax.top_k(log_joint, k=self.K)
        log_posterior = log_posterior[top_indices]
        return log_posterior, log_joint, top_indices
    

    def deflate_belief(self, bel, bel_prior):
        gamma = jnp.exp(bel.log_posterior)
        dim = bel.mean.shape[0]
        deflate_mean = gamma ** self.deflate_mean

        new_mean = bel.mean * deflate_mean
        new_cov = bel.cov * gamma ** 2 + (1 - gamma ** 2) * jnp.eye(dim) * self.shock
        bel = bel.replace(mean=new_mean, cov=new_cov)
        return bel


    def step(self, y, X, bel, bel_prior, callback_fn):
        """
        Update belief state and log-joint for a single observation
        """

        log_posterior, log_joint, top_indices = self.update_log_joint(y, X, bel, bel_prior)
        bel_posterior = jax.vmap(self.deflate_belief, in_axes=(0, None))(bel, bel_prior)
        bel_posterior = self.update_bel_indices(y, X, bel_posterior, bel_prior, top_indices)

        bel_posterior = self.update_runlengths(bel_posterior, top_indices)
        bel_posterior = bel_posterior.replace(log_joint=log_joint, log_posterior=log_posterior)

        out = callback_fn(bel_posterior, bel, y, X, top_indices)

        return bel_posterior, out


class RunlengthChangepointCount(ABC):
    def __init__(self, K, b, reset_mean=True):
        """
        Bayesian online changepoint and hazard detection
        RLCP
        """
        self.K = K
        self.b = b
        self.reset_mean = reset_mean * 1.0

    @abstractmethod
    def init_bel(self, y, X, bel_init):
        ...

    @abstractmethod
    def log_predictive_density(self, y, X, bel):
        ...

    @abstractmethod
    def update_bel(self, y, X, bel):
        """
        Update belief state (posterior)
        """
        ...

    def p_change(self, changepoints, t):
        alpha = changepoints
        beta = t - alpha + self.b
        proba = (alpha + 1) / (alpha + beta + 2)
        return proba

    @partial(jax.vmap, in_axes=(None, None, None, None, 0))
    def update_log_joint_increase(self, y, X, t, bel):
        log_p_pred = self.log_predictive_density(y, X, bel)
        log_joint = log_p_pred + bel.log_joint + jnp.log1p(-self.p_change(bel.changepoints, t))
        return log_joint

    @partial(jax.vmap, in_axes=(None, None, None, None, 0, None))
    def update_log_joint_reset(self, y, X, t, bel, bel_prior):
        log_p_pred = self.log_predictive_density(y, X, bel_prior)
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

    def update_log_joint(self, y, X, t, bel, bel_prior):
        log_joint_reset = self.update_log_joint_reset(y, X, t, bel, bel_prior)
        log_joint_increase = self.update_log_joint_increase(y, X, t, bel)
        # Expand log-joint
        log_joint = jnp.concatenate([log_joint_reset, log_joint_increase])
        log_joint = jnp.nan_to_num(log_joint, nan=-jnp.inf, neginf=-jnp.inf)
        # sekect top K values
        _, top_indices = jax.lax.top_k(log_joint, k=self.K)
        return log_joint, top_indices

    def update_bel_increase(self, y, X, bel, bel_prior):
        bel_update = self.update_bel(y, X, bel)
        bel_update = self.update_runlengths(bel_update)
        return bel_update

    def update_bel_reset(self, y, X, bel, bel_prior):
        changepoints = bel.changepoints + 1 # increase changepoints by one
        runlengths = bel.runlength * 0
        new_mean = bel_prior.mean * self.reset_mean + bel.mean * (1 - self.reset_mean)
        bel_update = bel_prior.replace(mean=new_mean, changepoints=changepoints, runlength=runlengths)
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
        bel_posterior = jax.tree.map(lambda breset, bupdate: jnp.concatenate([breset, bupdate]), bel_posterior_reset, bel_posterior_increase)
        bel_posterior = bel_posterior.replace(log_joint=log_joint)

        # from 2K to K belief states
        bel_posterior = jax.tree.map(lambda param: jnp.take(param, top_indices, axis=0), bel_posterior)
        # re-normalise log-joint
        log_joint_norm = bel_posterior.log_joint - jax.nn.logsumexp(bel_posterior.log_joint)
        bel_posterior = bel_posterior.replace(log_joint=log_joint_norm)

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


class RunlengthCovarianceReset(Runlength):
    def __init__(self, p_change, K, shock=0.0):
        """
        Covariance-reset runlength (CRRL)
        """
        super().__init__(p_change, K)
        self.shock = shock

    def step(self, y, X, bel, bel_prior, callback_fn):
        """
        Update belief state and log-joint for a single observation
        """
        ix_max = jnp.nanargmax(bel.log_joint)
        bel_prior = jax.tree.map(lambda x: x[ix_max], bel)
        dim = bel_prior.mean.shape[0]
        new_cov = jnp.eye(dim) / self.shock
        new_mean = bel_prior.mean
        bel_prior = bel_prior.replace(
            mean=new_mean,
            cov=new_cov,
            log_joint=bel_prior.log_joint,
            runlength=bel_prior.runlength
        )

        log_joint, top_indices = self.update_log_joint(y, X, bel, bel_prior)
        bel_posterior = self.update_bel_indices(y, X, bel, bel_prior, top_indices)

        bel_posterior = self.update_runlengths(bel_posterior, top_indices)
        bel_posterior = bel_posterior.replace(log_joint=log_joint)

        out = callback_fn(bel_posterior, bel, y, X, top_indices)

        return bel_posterior, out


class ChangepointLocation(ABC):
    """
    Changepoint location detection (CPL)
    """
    def __init__(self, p_change, K, shock, inflate_covariance, reset_mean):
        self.p_change = p_change
        self.K = K
        self.shock = shock
        self.inflate_covariance = inflate_covariance
        self.reset_mean = reset_mean * 1.0


    @abstractmethod
    def init_bel(self, mean, cov, log_weight):
        """
        Initialize belief state
        """
        ...


    @abstractmethod
    def log_predictive_density(self, y, X, bel):
        ...


    @abstractmethod
    def update_bel():
        ...

    def predict_cov_changepoint(self, bel):
        cov_previous = bel.cov
        dim = cov_previous.shape[0]
        cov_if_changepoint = cov_previous * self.inflate_covariance + jnp.eye(dim) * (1 - self.inflate_covariance)
        cov_if_changepoint = cov_if_changepoint / self.shock
        return cov_if_changepoint


    def predict_bel(self, bel, bel_prior, has_changepoint):
        cov_changepoint = self.predict_cov_changepoint(bel)
        cov_pred = cov_changepoint * has_changepoint + bel.cov * (1 - has_changepoint)
        cond_reset_mean = has_changepoint * self.reset_mean
        mean_pred = bel_prior.mean * cond_reset_mean + bel.mean * (1 - cond_reset_mean)

        bel = bel.replace(mean=mean_pred, cov=cov_pred)
        return bel


    def predict_and_update_bel(self, y, X, bel, bel_prior, has_changepoint):
        bel = self.predict_bel(bel, bel_prior, has_changepoint)
        bel = self.update_bel(y, X, bel)
        bel = bel.replace(segment=bel.segment + has_changepoint)
        return bel


    def split_and_update(self, y, X, bel, bel_prior):
        """
        Update belief state and log-joint for a single observation
        """
        bel_up = self.predict_and_update_bel(y, X, bel, bel_prior, has_changepoint=1.0)
        bel_down = self.predict_and_update_bel(y, X, bel, bel_prior, has_changepoint=0.0)

        log_pp_up = self.log_predictive_density(y, X, bel_up)
        log_pp_down = self.log_predictive_density(y, X, bel_down)

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


    def step(self, y, X, bel, bel_prior):
        # from K to 2K belief states
        vmap_split_and_update = jax.vmap(self.split_and_update, in_axes=(None, None, 0, None))
        bel = vmap_split_and_update(y, X, bel, bel_prior)
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
        bel_prior = jax.tree.map(lambda x: x[0], bel)
        def _step(bel, yX):
            y, X = yX
            bel_posterior = self.step(y, X, bel, bel_prior)
            out = callback_fn(bel_posterior, bel, y, X)

            return bel_posterior, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist


class EmpiricalBayesAdaptive(ABC):
    def __init__(
        self, n_inner, ebayes_lr, state_drift, deflate_mean, deflate_covariance
    ):
        self.n_inner = n_inner
        self.ebayes_lr = ebayes_lr # empirical bayes learning rate
        self.state_drift = state_drift
        self.deflate_mean = deflate_mean * 1.0
        self.deflate_covariance = deflate_covariance * 1.0

    @abstractmethod
    def init_bel(self):
        """
        Initialize belief state
        """
        ...

    @abstractmethod
    def log_predictive_density(self, y, X, bel):
        ...


    @abstractmethod
    def update_bel(self, y, X, bel):
        ...

    def predict_bel(self, eta, bel):
        gamma = jnp.exp(-eta / 2)
        dim = bel.mean.shape[0]

        deflation_mean = gamma ** self.deflate_mean
        deflation_covariance = (gamma ** 2) ** self.deflate_covariance

        mean = deflation_mean * bel.mean
        cov = deflation_covariance * bel.cov + (1 - gamma ** 2) * jnp.eye(dim) * self.state_drift
        bel = bel.replace(mean=mean, cov=cov)
        return bel


    def log_reg_predictive_density(self, eta, y, X, bel):
        bel = self.predict_bel(eta, bel)
        log_p_pred = self.log_predictive_density(y, X, bel)
        return log_p_pred


    def step(self, y, X, bel):
        grad_log_predict_density = jax.grad(self.log_reg_predictive_density, argnums=0)

        def _inner_pred(i, eta, bel):
            grad = grad_log_predict_density(eta, y, X, bel)
            eta = eta + self.ebayes_lr * grad
            eta = eta * (eta > 0) # hard threshold
            return eta

        _inner = partial(_inner_pred, bel=bel)
        eta = jax.lax.fori_loop(0, self.n_inner, _inner, bel.eta)
        bel = bel.replace(eta=eta)
        bel = self.predict_bel(bel.eta, bel)
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


class LinearModelFMBOCD(FullMemoryBayesianOnlineChangepointDetection):
    """
    Full-memory Bayesian Online Changepoint Detection for linear model
    with known measurement variance
    For a run with T data points and paramter dimension D,
    the algorithm has memory requirement of O(T * (T + 1) / 2 * D ^ 2)
    """
    def __init__(self, p_change, beta):
        super().__init__(p_change)
        self.beta = beta


    def init_bel(self, mean, cov, n_samples):
        d = mean.shape[0]
        size_filter = n_samples * (n_samples + 1) // 2
        hist_mean = jnp.zeros((size_filter, d))
        hist_cov = jnp.zeros((size_filter, d, d))
        bel_hist = states.GaussState(
            mean=hist_mean.at[0].set(mean),
            cov=hist_cov.at[0].set(cov * jnp.eye(d))
        )
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
    def log_predictive_density(self, y, X, bel):
        """
        Compute log-posterior predictive for a Gaussian with known variance
        """
        mean = bel.mean @ X
        scale = 1 / self.beta + X @  bel.cov @ X
        log_p_pred = distrax.Normal(mean, scale).log_prob(y)
        return log_p_pred


class ExpfamRLPR(Runlength):
    """
    Runlength prior reset (RL-PR)
    """
    def __init__(
            self, p_change, K, filter
    ):
        super().__init__(p_change, K)
        self.filter = filter
    
    def log_predictive_density(self, y, X, bel):
        return self.filter.log_predictive_density(y, X, bel)
    

    def update_bel(self, y, X, bel):
        bel_pred = self.filter._predict(bel)
        bel = self.filter._update(bel_pred, y, X)
        return bel
    

    def init_bel(self, mean, cov, log_joint_init):
        """
        Initialize belief state
        """
        state_filter = self.filter.init_bel(mean, cov)
        mean = state_filter.mean
        cov = state_filter.cov

        bel = states.BOCDGaussState(
            mean=einops.repeat(mean, "i -> k i", k=self.K),
            cov=einops.repeat(cov, "i j -> k i j", k=self.K),
            log_joint=(jnp.ones((self.K,)) * -jnp.inf).at[0].set(log_joint_init),
            runlength=jnp.zeros(self.K)
        )

        return bel


class ExpfamCPL(ChangepointLocation):
    def __init__(
        self, p_change, K, shock, inflate_covariance, reset_mean, filter
    ):
        super().__init__(p_change, K, shock, inflate_covariance, reset_mean)
        self.filter = filter

    def log_predictive_density(self, y, X, bel):
        return self.filter.log_predictive_density(y, X, bel)
    

    def update_bel(self, y, X, bel):
        bel = self.filter._predict(bel)
        bel = self.filter._update(bel, y, X)
        return bel

    def init_bel(self, mean, cov, log_weight):
        """
        Initialize belief state
        """
        state_filter = self.filter.init_bel(mean, cov)
        mean = state_filter.mean
        cov = state_filter.cov

        bel = states.BernoullChangeGaussState(
            mean=einops.repeat(mean, "i -> k i", k=self.K),
            cov=einops.repeat(cov, "i j -> k i j", k=self.K),
            log_weight=(jnp.ones((self.K,)) * -jnp.inf).at[0].set(log_weight),
            segment=jnp.zeros(self.K)
        )
        return bel


class ExpfamRLCC(RunlengthChangepointCount):
    def __init__(
        self, K, b, reset_mean, filter
    ):
        """
        Runlength and changepoint count with prior reset (RLCP-PR)
        """
        super().__init__(K, b, reset_mean)
        self.filter = filter
    

    def init_bel(self, mean, cov, log_weight):
        """
        Initialize belief state
        """
        state_filter = self.filter.init_bel(mean, cov)
        mean = state_filter.mean
        cov = state_filter.cov

        bel = states.BOCHDGaussState(
            mean=einops.repeat(mean, "i -> k i", k=self.K),
            cov=einops.repeat(cov, "i j -> k i j", k=self.K),
            log_joint=(jnp.ones((self.K,)) * -jnp.inf).at[0].set(log_weight),
            runlength=jnp.zeros(self.K),
            changepoints=jnp.zeros(self.K)
        )

        return bel

    
    def log_predictive_density(self, y, X, bel):
        return self.filter.log_predictive_density(y, X, bel)
    

    def update_bel(self, y, X, bel):
        bel_pred = self.filter._predict(bel)
        bel = self.filter._update(bel_pred, y, X)
        return bel


class ExpfamRLSR(RunlengthSoftReset):
    """
    Runlength with soft reset (RL-SR)
    We softly revert to the prior mean / covariance, as long
    as the hypothesis with highest density is not a changepoint (k=0)

    This method composes RL-SPR and RL-SCR
    """
    def __init__(
        self, p_change, K, shock, deflate_mean, filter,
    ):
        super().__init__(p_change, K, shock, deflate_mean)
        self.filter = filter

    def log_predictive_density(self, y, X, bel):
        return self.filter.log_predictive_density(y, X, bel)

    def update_bel(self, y, X, bel):
        bel_pred = self.filter._predict(bel)
        bel = self.filter._update(bel_pred, y, X)
        return bel

    def init_bel(self, mean, cov, log_joint_init):
        """
        Initialize belief state
        """
        state_filter = self.filter.init_bel(mean, cov)
        mean = state_filter.mean
        cov = state_filter.cov

        bel = states.ABOCDGaussState(
            mean=einops.repeat(mean, "i -> k i", k=self.K),
            cov=einops.repeat(cov, "i j -> k i j", k=self.K),
            log_joint=(jnp.ones((self.K,)) * -jnp.inf).at[0].set(log_joint_init),
            runlength=jnp.zeros(self.K),
            log_posterior=jnp.zeros(self.K),
        )

        return bel


class ExpfamRLCR(RunlengthCovarianceReset):
    """
    Runlength with covariance reset.
    """
    def __init__(self, p_change, K, shock, filter):
        super().__init__(p_change, K, shock)
        self.filter = filter

    def init_bel(self, mean, cov, log_joint_init):
        """
        Initialize belief state
        """
        state_filter = self.filter.init_bel(mean, cov)
        mean = state_filter.mean
        cov = state_filter.cov

        bel = states.BOCDGaussState(
            mean=einops.repeat(mean, "i -> k i", k=self.K),
            cov=einops.repeat(cov, "i j -> k i j", k=self.K),
            log_joint=(jnp.ones((self.K,)) * -jnp.inf).at[0].set(log_joint_init),
            runlength=jnp.zeros(self.K),
        )

        return bel

    def log_predictive_density(self, y, X, bel):
        return self.filter.log_predictive_density(y, X, bel)

    def update_bel(self, y, X, bel):
        bel_pred = self.filter._predict(bel)
        bel = self.filter._update(bel_pred, y, X)
        return bel
        

class ExpfamEBA(EmpiricalBayesAdaptive):
    def __init__(
        self, n_inner, ebayes_lr, state_drift,  deflate_mean, deflate_covariance, filter
    ):
        super().__init__(n_inner, ebayes_lr, state_drift, deflate_mean, deflate_covariance)
        self.filter = filter

    def init_bel(self, mean, cov, eta=0.0):
        """
        Initialize belief state
        """
        state_filter = self.filter.init_bel(mean, cov)
        mean = state_filter.mean
        cov = state_filter.cov

        bel = states.GammaFilterState(
            mean=mean,
            cov=cov,
            eta=eta
        )
        return bel

    def log_predictive_density(self, y, X, bel):
        return self.filter.log_predictive_density(y, X, bel)

    def update_bel(self, y, X, bel):
        bel_pred = self.filter._predict(bel)
        bel = self.filter._update(bel_pred, y, X)
        return bel


class RobustLinearModelFMBOCD(LinearModelFMBOCD):
    """"""
    def __init__(self, p_change, beta, c):
        super().__init__(p_change, beta)
        self.c = c
    

    def imq_kernel(self, residual):
        """
        Inverse multi-quadratic kernel
        """
        return 1 / jnp.sqrt(1 + residual ** 2 / self.c ** 2)

    def  update_bel_single(self, y, X, bel ):
        """
        Update belief state for a single observation
        """
        cov_previous = bel.cov
        mean_previous = bel.mean

        prec_previous = jnp.linalg.inv(cov_previous)

        Wt = self.imq_kernel(y - X @ mean_previous)
        prec_posterior = prec_previous + Wt ** 2 * self.beta * jnp.outer(X, X)
        cov_posterior = jnp.linalg.inv(prec_posterior)
        mean_posterior = cov_posterior @ (prec_previous @ mean_previous + Wt ** 2 * self.beta * X * y)

        bel = states.GaussState(mean=mean_posterior, cov=cov_posterior)
        return bel
    
    def log_predictive_density(self, y, X, bel):
        """
        Compute log-posterior predictive for a Gaussian with known variance
        """
        mean = bel.mean @ X
        residual = y - mean
        Wt = self.imq_kernel(residual)

        scale = 1 / (self.beta * Wt ** 2) + X @ bel.cov @ X
        log_p_pred = distrax.Normal(mean, scale).log_prob(y)
        return log_p_pred


class LoFiExpfamFBOCD(RunlengthSoftReset):
    """
    Low-memory Kalman-filter BOCD
    """
    def __init__(
        self, p_change, K, filter, shock
    ):
        super().__init__(p_change, K, shock)
        self.filter = filter

    def deflate_belief(self, bel, bel_prior):
        gamma = jnp.exp(bel.log_posterior)
        new_mean = bel.mean * gamma
        new_diagonal = bel_prior.diagonal * (1 - gamma) * self.shock + bel.diagonal * gamma
        low_rank = bel_prior.low_rank * (1 - gamma) + bel.low_rank * gamma
        bel = bel.replace(mean=new_mean, diagonal=new_diagonal, low_rank=low_rank)
        return bel

    def log_predictive_density(self, y, X, bel):
        return self.filter.log_predictive_density(y, X, bel)

    def update_bel(self, y, X, bel):
        # bel = self.filter._predict(bel)
        bel = self.filter._update(bel, y, X)
        return bel

    def init_bel(self, mean, cov=1.0):
        state_filter = self.filter.init_bel(mean, cov)
        mean = state_filter.mean
        diagonal = state_filter.diagonal
        low_rank = state_filter.low_rank

        bel = states.ABOCDLoFiState(
            mean=einops.repeat(mean, "i -> k i", k=self.K),
            diagonal=einops.repeat(diagonal, "i -> k i", k=self.K),
            low_rank=einops.repeat(low_rank, "i j -> k i j", k=self.K),
            log_joint=(jnp.ones((self.K,)) * -jnp.inf).at[0].set(0.0),
            runlength=jnp.zeros(self.K),
            log_posterior=jnp.zeros(self.K),
        )

        return bel
