import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

class ContaminatedSSM:
    def __init__(
        self, transition_matrix, projection_matrix, dynamics_covariance, observation_covariance,
        p_contamination, contamination_value,
        
    ):
        self.transition_matrix = transition_matrix
        self.projection_matrix = projection_matrix
        self.dynamics_covariance = dynamics_covariance
        self.observation_covariance = observation_covariance
        self.p_contamination = p_contamination
        self.contamination_value = contamination_value
        

    def step(self, z, key):
        key_latent, key_obs, key_contamination = jax.random.split(key, 3)
        dim_obs, dim_latent = self.projection_matrix.shape
        is_contaminated = jax.random.bernoulli(key_contamination, p=self.p_contamination)

        eps_obs = jax.random.multivariate_normal(
            key_obs, mean=jnp.zeros(dim_obs), cov=self.observation_covariance
        )

        eps_latent = jax.random.multivariate_normal(
            key_latent, mean=jnp.zeros(dim_latent), cov=self.dynamics_covariance
        )

        z_next = self.transition_matrix @ z + eps_latent
        x_next = self.projection_matrix @ z_next + eps_obs
        x_next = x_next * (1 - is_contaminated) + self.contamination_value * is_contaminated

        output = {
            "observed": x_next,
            "latent": z_next,
        }

        return z_next, output

    def sample(self, key, z0, n_steps):
        keys = jax.random.split(key, n_steps)
        _, output = jax.lax.scan(self.step, z0, keys)
        return output
    

class StudentT1D:
    def __init__(
        self, transition_matrix, projection_matrix, dynamics_scale, observation_scale,
        dof_latent, dof_observed,     
    ):
        self.transition = transition_matrix
        self.projection = projection_matrix
        self.dynamics_scale = dynamics_scale
        self.observation_scale = observation_scale
        self.dof_latent = dof_latent
        self.dof_observed = dof_observed
    
    def step(self, z, key):
        key_latent, key_obs = jax.random.split(key, 2)
        z_next =  jax.random.t(key_latent, df=self.dof_latent) * self.dynamics_scale + self.transition * z
        x_next = jax.random.t(key_obs, df=self.dof_observed) * self.observation_scale + self.projection * z_next

        output = {
            "observed": x_next,
            "latent": z_next,
        }

        return z_next, output
    
    def sample(self, key, z0, n_steps):
        keys = jax.random.split(key, n_steps)
        _, output = jax.lax.scan(self.step, z0, keys)
        return output
