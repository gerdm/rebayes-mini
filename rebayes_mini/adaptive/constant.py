import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from rebayes_mini import callbacks
from rebayes_mini.states import gaussian

class Constant(ABC):
    @abstractmethod
    def init_bel(self, y, X, bel_init):
        ...


    @abstractmethod
    def log_predictive_density(self, y, X, bel):
        ...


    @abstractmethod
    def update_bel(self, bel, y, X):
        """
        Update belief state (posterior)
        """
        ...


    @abstractmethod
    def step(self, bel, y, X, callback_fn):
        """
        Update belief state and log-joint for a single observation
        """

    @abstractmethod
    def predict(self, bel, X):
        ...

    def scan(self, bel, y, X, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        def _step(bel, yX):
            y, X = yX
            bel, out = self.step(bel, y, X, callback_fn)
            return bel, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist

class GaussianStateSpaceModel(Constant):
    """
    Gaussian state-space model with constant transition dynamics Ft
    and constant dynamics's covariance Qt
    """
    def __init__(self, updater, transition_dynamics, dynamics_covariance):
        self.updater = updater
        self.transition_dynamics = transition_dynamics
        self.dynamics_covariance = dynamics_covariance

    def predict(self, bel, X):
        return self.updater.predict_fn(bel.mean, X)

    def log_predictive_density(self, y, X, bel):
        return self.updater.log_predictive_density(y, X, bel)


    def predict_bel(self, bel):
        mean_pred = self.transition_dynamics @ bel.mean
        cov_pred = self.transition_dynamics @ bel.cov @ self.transition_dynamics.T + self.dynamics_covariance
        bel_pred = bel.replace(mean=mean_pred, cov=cov_pred)
        return bel_pred


    def update_bel(self, bel, y, X):
        bel = self.updater.update(bel, y, X)
        return bel
    

    def init_bel(self, mean, cov):
        """
        Initialize belief state
        """
        state_updater = self.updater.init_bel(mean, cov)
        mean = state_updater.mean
        cov = state_updater.cov
        bel = gaussian.Gauss.init_bel( mean, cov)
        return bel
    

    def step(self, bel, y, x, callback_fn):
        bel_predict = self.predict_bel(bel)
        bel_update = self.update_bel(bel_predict, y, x)
        output = callback_fn(bel_update, bel_predict, y, x, self)
        return bel_update, output


class GaussianCovarianceInflation(GaussianStateSpaceModel):
    """
    Constant additive covariance inflation
    """
    def __init__(self, updater, dynamics_covariance):
        super().__init__(updater, 1.0, dynamics_covariance)
    
    def init_bel(self, mean, cov):
        """
        Initialize belief state
        """
        state_updater = self.updater.init_bel(mean, cov)
        mean = state_updater.mean
        cov = state_updater.cov

        dim_state = len(mean)
        self.transition_dynamics = jnp.eye(dim_state)
        if isinstance(self.dynamics_covariance, float):
            self.dynamics_covariance = self.dynamics_covariance * jnp.eye(dim_state)

        bel = gaussian.Gauss.init_bel( mean, cov)
        return bel
