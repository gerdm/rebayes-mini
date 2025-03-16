import jax
import chex
import jax.numpy as jnp
from functools import partial
from rebayes_mini.methods.base_filter import BaseFilter

@partial(jax.vmap, in_axes=(0, None))
@partial(jax.vmap, in_axes=(None, 0))
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


class GaussianProcessRegression(BaseFilter):
    """
    Rebayes-mini-compatible GP regressor
    """
    def __init__(self, obs_variance, kernel):
        self.kernel = kernel
        self.obs_variance = obs_variance

    def init_bel(self, dim_in, buffer_size):
        X = jnp.zeros((buffer_size, dim_in))
        y = jnp.zeros(buffer_size)
        counter = jnp.zeros(buffer_size)
        bel = FIFOGPState(X=X, y=y, buffer_size=buffer_size, counter=counter, num_obs=0)
        return bel

    def predict(self, bel):
        return bel

    def update(self, bel, y, x):
        bel = bel.update_buffer(y, x)
        return bel
    
    def _sample_multivariate_gauss(self, key, mean, cov, n_samples, min_ev=1e-6):
        dim = cov.shape[0]
        U, S, Vh = jnp.linalg.svd(cov)
        S = jnp.sqrt(S)
        rvs = jax.random.normal(key, shape=(dim, n_samples))
        L = U @ jnp.diag(S) @ Vh
        return L @ rvs + mean[:, None]


    def _build_kernel_matrices(self, bel, x):
        mask = jnp.where(bel.counter == 0)[0]

        var_train = self.kernel(bel.X, bel.X)
        var_train_diag = jnp.diag(var_train) + self.obs_variance
        var_train = var_train.at[jnp.diag_indices_from(var_train)].set(var_train_diag)
        var_train_masked = var_train.at[mask].set(0.0).at[:, mask].set(0.0)

        cov_test_train = self.kernel(x, bel.X)
        var_test = self.kernel(x, x)

        return cov_test_train, var_train_masked, var_test


    def sample_fn(self, key, bel):
        # TODO: double check that nothing funny is happening the K when buffer is not filled
        def fn(x):
            cov_test_train, var_train, var_test = self._build_kernel_matrices(bel, x)
            K = jnp.linalg.lstsq(var_train, cov_test_train.T)[0].T
            mu_pred = K @ bel.y # mean posterior predictive
            # cov_pred = var_test - K @ var_train @ K.T
            cov_pred = var_pred = var_test - jnp.einsum("ij,jk,lk->il", K, var_train, K,  precision="highest")
            sample = self._sample_multivariate_gauss(key, mu_pred, cov_pred, n_samples=1)
            return sample
        return fn


    def mean_fn(self, bel, x):
        cov_test_train, var_train, _ = self._build_kernel_matrices(bel, x)
        # Takes care of rows and columns set to zero
        K = jnp.linalg.lstsq(var_train, cov_test_train.T)[0].T
        return K @ bel.y
