"""
Robust Filtering with Contaminated Data

This example demonstrates how robust filters handle outliers and contaminated
observations better than standard Kalman filters.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rebayes_mini.methods.gauss_filter import KalmanFilter  
from rebayes_mini.methods.student_t_filter import StudentTFilter
from rebayes_mini.datasets.linear_ssm import ContaminatedSSM
from rebayes_mini.callbacks import get_updated_mean


def main():
    key = jax.random.PRNGKey(123)
    
    # Create contaminated state space model
    ssm = ContaminatedSSM(
        transition_matrix=jnp.array([[1.0]]),
        projection_matrix=jnp.array([[1.0]]),
        dynamics_covariance=jnp.array([[0.1]]),
        observation_covariance=jnp.array([[1.0]]),
        p_contamination=0.1,  # 10% contamination
        contamination_value=10.0  # Large outlier value
    )
    
    # Generate contaminated data
    initial_state = jnp.array([0.0])
    data = ssm.sample(key, initial_state, n_steps=100)
    
    true_states = data['latent']
    observations = data['observed']
    
    # Standard Kalman Filter
    kf = KalmanFilter(
        transition_matrix=jnp.array([[1.0]]),
        dynamics_covariance=jnp.array([[0.1]]),
        observation_covariance=jnp.array([[1.0]])
    )
    
    # Student-t Filter (robust to outliers)
    stf = StudentTFilter(
        transition_matrix=jnp.array([[1.0]]),
        dynamics_covariance=jnp.array([[0.1]]),
        observation_covariance=jnp.array([[1.0]]),
        dof=3.0  # Degrees of freedom for t-distribution
    )
    
    # Initialize both filters
    bel_kf = kf.init_bel(jnp.array([0.0]), cov=1.0)
    bel_stf = stf.init_bel(jnp.array([0.0]), cov=1.0)
    
    # Run both filters
    obs_matrix = jnp.array([[1.0]])
    obs_matrices = jnp.tile(obs_matrix[None, ...], (len(observations), 1, 1))
    
    _, kf_means = kf.scan(bel_kf, observations, obs_matrices, get_updated_mean)
    _, stf_means = stf.scan(bel_stf, observations, obs_matrices, get_updated_mean)
    
    kf_estimates = jnp.array([m[0] for m in kf_means])
    stf_estimates = jnp.array([m[0] for m in stf_means])
    
    # Identify outliers for visualization
    outliers = jnp.abs(observations - true_states) > 5.0
    
    # Plot comparison
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(true_states, 'g-', label='True State', linewidth=2)
    normal_obs = observations * (~outliers)
    outlier_obs = observations * outliers
    plt.plot(jnp.where(~outliers, observations, jnp.nan), 'b.', 
             label='Normal Observations', alpha=0.6, markersize=4)
    plt.plot(jnp.where(outliers, observations, jnp.nan), 'r*', 
             label='Outliers', markersize=8)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('True State and Contaminated Observations')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(true_states, 'g-', label='True State', linewidth=2)
    plt.plot(kf_estimates, 'b-', label='Kalman Filter', linewidth=2)
    plt.plot(stf_estimates, 'r-', label='Student-t Filter', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Estimated State')
    plt.title('Filter Estimates Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    kf_errors = jnp.abs(kf_estimates - true_states)
    stf_errors = jnp.abs(stf_estimates - true_states)
    plt.plot(kf_errors, 'b-', label='Kalman Filter Error', linewidth=2)
    plt.plot(stf_errors, 'r-', label='Student-t Filter Error', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error')
    plt.title('Tracking Errors')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('robust_filtering_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    kf_rmse = jnp.sqrt(jnp.mean((kf_estimates - true_states)**2))
    stf_rmse = jnp.sqrt(jnp.mean((stf_estimates - true_states)**2))
    
    print(f"Kalman Filter RMSE: {kf_rmse:.4f}")
    print(f"Student-t Filter RMSE: {stf_rmse:.4f}")
    print(f"Improvement: {((kf_rmse - stf_rmse) / kf_rmse * 100):.1f}%")
    print(f"Number of outliers: {jnp.sum(outliers)}")


if __name__ == "__main__":
    main()