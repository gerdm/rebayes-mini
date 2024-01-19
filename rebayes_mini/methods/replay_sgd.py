import jax
import chex
import jax.numpy as jnp
from typing import Tuple
from functools import partial
from flax.training.train_state import TrainState
from rebayes_mini import callbacks


class FifoTrainState(TrainState):
    buffer_size: int
    num_obs: int
    buffer_X: chex.Array
    buffer_y: chex.Array
    counter: int

    @property
    def mean(self):
        return self.params

    def _update_buffer(self, step, buffer, item):
        ix_buffer = step % self.buffer_size
        buffer = buffer.at[ix_buffer].set(item)
        return buffer


    def apply_buffers(self, X, y):
        # TODO: rename to update_buffers
        n_count = self.num_obs
        buffer_X = self._update_buffer(n_count, self.buffer_X, X)
        buffer_y = self._update_buffer(n_count, self.buffer_y, y)
        counter = self._update_buffer(n_count, self.counter, 1.0)

        return self.replace(
            num_obs=n_count + 1,
            buffer_X=buffer_X,
            buffer_y=buffer_y,
            counter=counter,
        )


    @classmethod
    def create(cls, *, apply_fn, params, tx,
               buffer_size, dim_features, dim_output, **kwargs):
        opt_state = tx.init(params)
        if isinstance(dim_features, int): 
            buffer_X = jnp.empty((buffer_size, dim_features))
        else:
            buffer_X = jnp.empty((buffer_size, *dim_features))
        buffer_y = jnp.empty((buffer_size, dim_output))
        counter = jnp.zeros(buffer_size)

        return cls(
            step=0,
            num_obs=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            buffer_size=buffer_size,
            buffer_X=buffer_X,
            buffer_y=buffer_y,
            counter=counter,
            **kwargs
        )


class FifoSGD:
    """
    FIFO Replay-buffer SGD training procedure
    """
    def __init__(self, apply_fn, lossfn, tx, buffer_size, dim_features, dim_output, n_inner=1):
        self.apply_fn = apply_fn
        self.lossfn = lossfn
        self.tx = tx
        self.buffer_size = buffer_size
        self.dim_features = dim_features
        self.dim_output = dim_output
        self.n_inner = n_inner
        self.loss_grad = jax.value_and_grad(self.lossfn, 0)


    # TODO: implement buffer initialisation with X, y
    # TODO: define self.dim_features, dim_output in this step
    def init_bel(self, params, X=None, y=None):
        if self.apply_fn is None:
            raise ValueError("Must provide apply_fn")
        bel_init = FifoTrainState.create(
            apply_fn=self.apply_fn,
            params = params,
            tx = self.tx,
            buffer_size = self.buffer_size,
            dim_features = self.dim_features,
            dim_output = self.dim_output
        )
        return bel_init

    def predict_obs(self, bel, X):
        yhat = self.apply_fn(bel.params, X)
        return yhat

    def _train_step(
        self,
        bel: FifoTrainState,
    ) -> Tuple[float, FifoTrainState]:
        X, y = bel.buffer_X, bel.buffer_y
        loss, grads = self.loss_grad(bel.params, bel.counter, X, y, bel.apply_fn)
        bel = bel.apply_gradients(grads=grads)
        return loss, bel

    def update_state(self, bel, Xt, yt):
        bel = bel.apply_buffers(Xt, yt)

        def partial_step(_, bel):
            _, bel = self._train_step(bel)
            return bel
        bel = jax.lax.fori_loop(0, self.n_inner - 1, partial_step, bel)
        # Do not count inner steps as part of the outer step
        _, bel = self._train_step(bel)
        return bel
    
    def _step(self, bel, xs, callback_fn):
        X, y = xs
        bel_update = self.update_state(bel, X, y)
        output = callback_fn(bel_update, bel, y, X)
        return bel_update, output

    def scan(self, bel, y, x, callback_fn=None):
        D = (x, y)
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        _step = partial(self._step, callback_fn=callback_fn)
        bel, hist = jax.lax.scan(_step, bel, D)
        return bel, hist
