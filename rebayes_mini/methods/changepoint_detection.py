import jax
import distrax
import jax.numpy as jnp
from tqdm import tqdm
from functools import partial

class LinearModelBayesianOnlineChangepointDetection:
    """
    LM-BOCD: Bayesian Online Changepoint Detection for linear regression
    with known measurement variance
    For a run with T data points and paramter dimension D,
    the algorithm has memory requirement of O(T * (T + 1) / 2 * D ^ 2)
    """
    def __init__(self, mean_prior, cov_prior, p_change, beta):
        # TODO (?) refactor into chex state
        self.mean_prior = mean_prior
        self.cov_prior = cov_prior
        self.p_change = p_change
        self.beta = beta


    @partial(jax.jit, static_argnums=(0,))
    def get_ix(self,t, ell):
        # Number of steps to get to first observation at time t
        ix_step = t * (t + 1) // 2
        # Increase runlength
        ix = ix_step + ell
        return ix

    @partial(jax.jit, static_argnums=(0,))
    def compute_log_posterior_predictive(self, mean_posterior, cov_posterior, y, X):
        """
        Compute log-posterior predictive for a Gaussian with known variance
        """
        mean = mean_posterior @ X
        scale = 1 / self.beta + X @ cov_posterior @ X
        log_p_pred = distrax.Normal(mean, scale).log_prob(y)
        return log_p_pred


    # @jax.jit
    def update_log_joint_reset(self, t, ell, y, X, mean_hist, cov_hist, log_joint_hist):
        log_p_pred = self.compute_log_posterior_predictive(self.mean_prior, self.cov_prior, y, X)
        
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
    def update_log_joint_increase(self, t, ell, y, X, mean_hist, cov_hist, log_joint_hist):
        ix_update = self.get_ix(t, ell)
        ix_prev = self.get_ix(t-1, ell-1)

        mean_posterior = mean_hist[ix_update]
        cov_posterior = cov_hist[ix_update]
        log_p_pred = self.compute_log_posterior_predictive(mean_posterior, cov_posterior, y, X)
        
        log_joint = log_p_pred + log_joint_hist[ix_prev] + jnp.log(1 - self.p_change)
        log_joint = log_joint.squeeze()
        return log_joint_hist.at[ix_update].set(log_joint)


    @partial(jax.jit, static_argnums=(0,))   
    def update_posterior_params_reset(self, t, ell, y, X, mean_hist, cov_hist):
        ix_update = self.get_ix(t+1, ell)
        
        mean_hist = mean_hist.at[ix_update].set(self.mean_prior)
        cov_hist = cov_hist.at[ix_update].set(self.cov_prior)
        return mean_hist, cov_hist


    @partial(jax.jit, static_argnums=(0,))   
    def update_posterior_params_increase(self, t, ell, y, X, mean_hist, cov_hist):
        ix_prev = self.get_ix(t, ell-1)
        ix_update = self.get_ix(t+1, ell)
        
        cov_previous = cov_hist[ix_prev]
        mean_previous = mean_hist[ix_prev]
        prec_previous = jnp.linalg.inv(cov_previous)
        
        prec_posterior = prec_previous + self.beta * jnp.outer(X, X)
        cov_posterior = jnp.linalg.inv(prec_posterior)
        mean_posterior = cov_posterior @ (prec_previous @ mean_previous + self.beta * X * y)

        mean_hist = mean_hist.at[ix_update].set(mean_posterior)
        cov_hist = cov_hist.at[ix_update].set(cov_posterior)
        return mean_hist, cov_hist


    @partial(jax.jit, static_argnums=(0,))   
    def update_posterior_params(self, t, ell, y, X, mean_hist, cov_hist):
        params = t, ell, y, X, mean_hist, cov_hist
        mean_hist, cov_hist = jax.lax.cond(
            ell == 0,
            self.update_posterior_params_reset,
            self.update_posterior_params_increase,
            *params
        )
            
        return mean_hist, cov_hist
    

    def update_log_posterior(self, t, runlength_log_posterior, marginal, log_joint_hist):
        ix_init = self.get_ix(t, 0)
        ix_end = self.get_ix(t, t+1)

        section = slice(ix_init, ix_end, 1)
        log_joint_sub =  log_joint_hist[section]
        
        marginal_t = jax.nn.logsumexp(log_joint_sub)
        marginal = marginal.at[t].set(marginal_t)
        
        update = log_joint_sub - marginal_t
        runlength_log_posterior = runlength_log_posterior.at[section].set(update)

        return marginal, runlength_log_posterior


    def scan(self, y, X):
        """
        Bayesian online changepoint detection (BOCD)
        discrete filter
        """
        n_samples, d = X.shape
        
        hist_mean = jnp.zeros((n_samples, n_samples, d))
        hist_cov = jnp.zeros((n_samples, n_samples, d, d))

        size_filter = n_samples * (n_samples + 1) // 2
        
        marginal = jnp.zeros(n_samples)
        log_joint = jnp.zeros((size_filter,))
        log_cond = jnp.zeros((size_filter,))
        hist_mean = jnp.zeros((size_filter, d))
        hist_cov = jnp.zeros((size_filter, d, d))
        
        hist_mean = hist_mean.at[0].set(self.mean_prior)
        hist_cov = hist_cov.at[0].set(self.cov_prior)
        
        for t in tqdm(range(n_samples)):
            xt = X[t].squeeze()
            yt = y[t].squeeze()
        
            # Compute log-joint
            for ell in range(t+1):
                if ell == 0:
                    log_joint = self.update_log_joint_reset(t, ell, yt, xt, hist_mean, hist_cov, log_joint)
                else:
                    log_joint = self.update_log_joint_increase(t, ell, yt, xt, hist_mean, hist_cov, log_joint)
                
            # compute runlength log-posterior
            marginal, log_cond = self.update_log_posterior(t, log_cond, marginal, log_joint)
            
            # Update posterior parameters
            for ell in range(t+1):
                hist_mean, hist_cov = self.update_posterior_params(t, ell, yt, xt, hist_mean, hist_cov)

        out = {
            "log_joint": log_joint,
            "log_runlength_posterior": log_cond,
            "marginal": marginal,
            "means": hist_mean,
            "covs": hist_cov
        }

        return out
