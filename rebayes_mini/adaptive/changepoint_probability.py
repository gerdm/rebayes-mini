import jax
import jax.numpy as jnp
from functools import partial
from abc import ABC, abstractmethod
from rebayes_mini import callbacks
from rebayes_mini.states import gaussian


class GaussianOUEmpiricalBayes:
    def __init__(
            self, updater, n_inner, ebayes_lr, state_drift, deflate_mean=True, deflate_covariance=True
    ):
        self.n_inner = n_inner
        self.ebayes_lr = ebayes_lr 
        self.state_drift = state_drift
        self.deflate_mean = deflate_mean * 1.0
        self.deflate_covariance = deflate_covariance * 1.0
        self.updater = updater


    def predict(self, bel, X):
        yhat = self.updater.predict_fn(bel.mean, X)
        return yhat


    def init_bel(self, mean, cov, eta=-10.0):
        """
        Initialize belief state
        """
        state_updater = self.updater.init_bel(mean, cov)
        mean = state_updater.mean
        cov = state_updater.cov
        bel = gaussian.GaussCPP.init_bel(mean, cov, eta)
        return bel
    
    def get_rate(self, bel):
        gamma = jnp.exp(-bel.eta / 2)
        return gamma
    
    def predict_bel(self, eta, bel):
        gamma = jnp.exp(-eta / 2)
        dim = bel.mean.shape[0]

        deflation_mean = gamma ** self.deflate_mean
        deflation_covariance = (gamma ** 2) ** self.deflate_covariance

        mean = deflation_mean * bel.mean
        cov = deflation_covariance * bel.cov + (1 - gamma ** 2) * jnp.eye(dim) * self.state_drift
        bel = bel.replace(mean=mean, cov=cov)
        return bel


    def log_predictive_density(self, y, X, bel):
        return self.updater.log_predictive_density(y, X, bel)


    def update_bel(self, bel, y, X):
        bel = self.updater.update(bel, y, X)
        return bel
    

    def log_reg_predictive_density(self, eta, y, X, bel):
        bel = self.predict_bel(eta, bel)
        log_p_pred = self.log_predictive_density(y, X, bel)
        return log_p_pred


    def step(self, bel, y, X):
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
        bel = self.update_bel(bel, y, X)
        return bel


    def scan(self, bel, y, X, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        def _step(bel, yX):
            y, X = yX
            bel_posterior = self.step(bel, y, X)
            out = callback_fn(bel_posterior, bel, y, X, self)

            return bel_posterior, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist