import jax
from abc import ABC, abstractmethod
from rebayes_mini import callbacks

class BaseAuxiliary(ABC):
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