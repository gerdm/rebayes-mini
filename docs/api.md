# API Reference

## Core Modules

### rebayes_mini.states

State data structures for Bayesian filters.

#### GaussState
```python
@chex.dataclass
class GaussState:
    mean: chex.Array  # Posterior mean vector
    cov: chex.Array   # Posterior covariance matrix
```

Standard Gaussian belief state representing a multivariate normal distribution.

#### PULSEGaussState
```python
@chex.dataclass  
class PULSEGaussState:
    mean_hidden: chex.Array  # Hidden layer posterior mean
    prec_hidden: chex.Array  # Hidden layer precision matrix
    mean_last: chex.Array    # Last layer posterior mean
    prec_last: chex.Array    # Last layer precision matrix
```

State for PULSE (Probabilistic Updates of Last Subspace Estimate) filtering.

#### BOCDGaussState
```python
@chex.dataclass
class BOCDGaussState:
    mean: chex.Array      # Posterior mean
    cov: chex.Array       # Posterior covariance  
    log_joint: chex.Array # Log joint probabilities
    runlength: chex.Array # Runlength distribution
```

State for Bayesian Online Changepoint Detection with Gaussian posteriors.

### rebayes_mini.callbacks

Callback functions for extracting filter outputs.

#### get_null(bel_update, bel_prev, y, x, *args, **kwargs) -> None
Default callback that returns None.

#### get_updated_mean(bel_update, bel_prev, y, x, *args, **kwargs) -> Array
Returns the posterior mean after update.

#### get_updated_bel(bel_update, bel_prev, y, x, *args, **kwargs) -> State  
Returns the complete updated belief state.

#### get_predicted_mean(bel_update, bel_prev, y, x, *args, **kwargs) -> Array
Returns the predicted mean before update.

## Filter Methods

### rebayes_mini.methods.gauss_filter

#### KalmanFilter
```python
class KalmanFilter:
    def __init__(self, transition_matrix, dynamics_covariance, observation_covariance):
        """
        Standard Kalman filter for linear Gaussian state space models.
        
        Args:
            transition_matrix: State transition matrix A
            dynamics_covariance: Process noise covariance Q  
            observation_covariance: Observation noise covariance R
        """
```

**Methods:**
- `init_bel(mean, cov=1.0) -> GaussState`: Initialize belief state
- `step(bel, y, obs_matrix, callback_fn) -> (GaussState, output)`: Single filter step
- `scan(bel, y, X, callback_fn=None) -> (GaussState, history)`: Batch filtering

### rebayes_mini.methods.base_filter

#### BaseFilter
```python
class BaseFilter(ABC):
    def __init__(self, mean_fn, cov_fn):
        """
        Abstract base class for all filters.
        
        Args:
            mean_fn: Mean function (maps parameters to prediction)
            cov_fn: Covariance function (maps prediction to noise level)
        """
```

**Abstract Methods:**
- `init_bel()`: Initialize belief state
- `predict(bel)`: Prediction step
- `update(bel, y, x)`: Update step  
- `sample_fn(key, bel)`: Sample function from posterior

#### ExtendedFilter
Extended Kalman filter implementation for nonlinear functions.

### rebayes_mini.methods.student_t_filter

#### StudentTFilter
```python
class StudentTFilter:
    def __init__(self, transition_matrix, dynamics_covariance, observation_covariance, dof):
        """
        Robust filter using Student-t distribution for observations.
        
        Args:
            transition_matrix: State transition matrix
            dynamics_covariance: Process noise covariance
            observation_covariance: Base observation noise covariance
            dof: Degrees of freedom for t-distribution
        """
```

Better handling of outliers compared to Gaussian filters.

### rebayes_mini.methods.low_rank_filter

Memory-efficient filters using low-rank approximations for high-dimensional problems.

### rebayes_mini.methods.gaussian_process

Online Gaussian Process regression and classification methods.

### rebayes_mini.methods.ensemble_kalman_filter

Ensemble-based filtering methods for nonlinear problems.

## Datasets

### rebayes_mini.datasets.linear_ssm

#### ContaminatedSSM
```python
class ContaminatedSSM:
    def __init__(self, transition_matrix, projection_matrix, dynamics_covariance, 
                 observation_covariance, p_contamination, contamination_value):
        """
        Linear state space model with contaminated observations.
        
        Args:
            transition_matrix: State dynamics matrix
            projection_matrix: Observation matrix
            dynamics_covariance: Process noise covariance
            observation_covariance: Clean observation noise covariance  
            p_contamination: Probability of contamination
            contamination_value: Value of contaminated observations
        """
```

**Methods:**
- `sample(key, z0, n_steps) -> dict`: Generate synthetic trajectory

## Usage Patterns

### Basic Filtering
```python
import jax.numpy as jnp
from rebayes_mini.methods.gauss_filter import KalmanFilter
from rebayes_mini.callbacks import get_updated_mean

# Create filter
kf = KalmanFilter(
    transition_matrix=jnp.array([[1.0]]),
    dynamics_covariance=jnp.array([[0.1]]), 
    observation_covariance=jnp.array([[1.0]])
)

# Initialize
bel = kf.init_bel(jnp.array([0.0]))

# Process data
observations = jnp.array([1.0, 2.0, 1.5])
obs_matrices = jnp.tile(jnp.array([[1.0]])[None, ...], (len(observations), 1, 1))

final_bel, means_history = kf.scan(bel, observations, obs_matrices, get_updated_mean)
```

### Custom Callbacks
```python
def custom_callback(bel_update, bel_prev, y, x):
    return {
        'mean': bel_update.mean,
        'variance': jnp.diag(bel_update.cov),
        'prediction_error': y - bel_prev.mean
    }

_, history = filter.scan(bel, y, X, custom_callback)
```