import jax
import chex
import jax.numpy as jnp
from functools import partial

@partial(jax.vmap, in_axes=(0, None, None))
@partial(jax.vmap, in_axes=(None, 0, None))
def matern_kernel(u, v, length_scale=1.0, nu=1/2):
    """
    https://andrewcharlesjones.github.io/journal/matern-kernels.html
    """
    # Compute the distance between u and v
    r = jnp.linalg.norm(u - v, ord=2)

    # Calculate the scaling factor
    scaled_distance = r / length_scale
    
    # Define the Matérn kernel based on the value of nu
    if nu == 1/2:
        kernel_value = jnp.exp(-scaled_distance)
    elif nu == 3/2:
        kernel_value = (1 + jnp.sqrt(3) * scaled_distance) * jnp.exp(-jnp.sqrt(3) * scaled_distance)
    elif nu == 5/2:
        kernel_value = (1 + jnp.sqrt(5) * scaled_distance + (5 * scaled_distance**2) / 3) * jnp.exp(-jnp.sqrt(5) * scaled_distance)
    else:
        raise ValueError(f"Unsupported nu value: {nu}")

    return kernel_value


@chex.dataclass
class FIFOGPState:
    buffer_size: int
    num_obs: int
    X: jax.Array
    y: jax.Array
    counter: jax.Array

    def _update_buffer(self, step, buffer, item):
        ix_buffer = step % self.buffer_size
        buffer = buffer.at[ix_buffer].set(item)
        return buffer

    def update_buffer(self, y, X):
        n_count = self.num_obs
        X = self._update_buffer(n_count, self.X, X)
        y = self._update_buffer(n_count, self.y, y)
        counter = self._update_buffer(n_count, self.counter, 1.0)

        return self.replace(
            num_obs=n_count + 1,
            X=X,
            y=y,
            counter=counter,
        )


class GaussianProcessRegression:
    """
    Rebayes-mini-compatible GP regressor
    """
    def __init__(self, sigma=1.0):
        self.kernel = matern_kernel
        self.sigma = sigma

    def init_bel(self, dim_in, buffer_size):
        X = jnp.zeros((buffer_size, dim_in))
        y = jnp.zeros(buffer_size)
        counter = jnp.zeros(buffer_size)
        bel = FIFOGPState(X=X, y=y, buffer_size=buffer_size, counter=counter, num_obs=0)
        return bel

    def update(self, bel, y, x):
        bel = bel.update_buffer(y, x)
        return bel

    def mean_fn(self, bel, x):
        var_train = self.kernel(bel.X, bel.X, self.sigma)
        cov_test_train = self.kernel(x, bel.X, self.sigma)
        # var_test = self.kernel(x, x, self.sigma)
        K = jnp.linalg.solve(var_train, cov_test_train.T).T # cov_test_train @ inv(var_train)
        return K @ bel.y
