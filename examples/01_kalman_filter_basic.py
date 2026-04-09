"""
Basic Kalman Filter Example

This example demonstrates how to use the Kalman filter for tracking
a simple 1D random walk with noisy observations.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rebayes_mini.methods.gauss_filter import KalmanFilter
from rebayes_mini.callbacks import get_updated_mean, get_updated_bel


def main():
    # Setup the problem
    key = jax.random.PRNGKey(42)
    
    # Parameters for a 1D random walk
    transition_matrix = jnp.array([[1.0]])  # x[t+1] = x[t] + noise
    dynamics_covariance = jnp.array([[0.1]])  # Process noise
    observation_covariance = jnp.array([[1.0]])  # Observation noise
    
    # Create Kalman filter
    kf = KalmanFilter(
        transition_matrix=transition_matrix,
        dynamics_covariance=dynamics_covariance,
        observation_covariance=observation_covariance
    )
    
    # Generate synthetic data (true random walk + noisy observations)
    n_steps = 100
    true_states = jnp.zeros(n_steps)
    observations = jnp.zeros(n_steps)
    
    key, subkey = jax.random.split(key)
    process_noise = jax.random.normal(subkey, (n_steps,)) * jnp.sqrt(0.1)
    
    key, subkey = jax.random.split(key)
    obs_noise = jax.random.normal(subkey, (n_steps,)) * jnp.sqrt(1.0)
    
    # Generate true trajectory
    true_state = 0.0
    for t in range(n_steps):
        true_state += process_noise[t]
        true_states = true_states.at[t].set(true_state)
        observations = observations.at[t].set(true_state + obs_noise[t])
    
    # Initialize filter
    initial_mean = jnp.array([0.0])
    bel = kf.init_bel(initial_mean, cov=1.0)
    
    # Run filter
    observation_matrix = jnp.array([[1.0]])  # Direct observation of state
    obs_matrices = jnp.tile(observation_matrix[None, ...], (n_steps, 1, 1))
    
    final_bel, means_history = kf.scan(
        bel, observations, obs_matrices, callback_fn=get_updated_mean
    )
    
    # Extract results
    estimated_means = jnp.array([m[0] for m in means_history])
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(true_states, 'g-', label='True State', linewidth=2)
    plt.plot(observations, 'r.', alpha=0.6, label='Noisy Observations', markersize=4)
    plt.plot(estimated_means, 'b-', label='Kalman Filter Estimate', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('1D Kalman Filter Tracking')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    errors = jnp.abs(estimated_means - true_states)
    plt.plot(errors, 'k-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error')
    plt.title('Tracking Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('kalman_filter_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    rmse = jnp.sqrt(jnp.mean((estimated_means - true_states)**2))
    print(f"Root Mean Square Error: {rmse:.4f}")
    print(f"Final estimated state: {estimated_means[-1]:.4f}")
    print(f"True final state: {true_states[-1]:.4f}")


if __name__ == "__main__":
    main()