"""
Bayesian Filtering Methods

This module contains implementations of various recursive Bayesian estimation
algorithms. All filters follow a common interface defined in base_filter.py
and are designed for online learning and sequential data processing.

Filter Categories:

Linear Filters:
    - gauss_filter.py: Kalman Filter variants (standard, extended, square-root)
    
Robust Filters:
    - robust_filter.py: Robust filters for handling outliers and heavy-tailed noise
    - student_t_filter.py: Student-t filter for robust estimation
    
Low-Rank Approximations:
    - low_rank_filter.py: Memory-efficient filters for high-dimensional problems
    - low_rank_filter_revised.py: Improved low-rank implementations
    - low_rank_last_layer.py: Low-rank approximation for neural network last layers
    
Subspace Methods:
    - subspace_filter.py: Subspace-based filtering methods
    - subspace_last_layer.py: Subspace methods for neural network outputs
    
Ensemble Methods:
    - ensemble_kalman_filter.py: Ensemble Kalman Filter (EnKF)
    
Gaussian Processes:
    - gaussian_process.py: Online Gaussian Process regression and classification
    
Variational Methods:
    - recursive_vi_gauss.py: Recursive Variational Inference with Gaussian approximation
    
Optimization-based:
    - replay_sgd.py: SGD with experience replay for online learning
    
Adaptive Methods:
    - adaptive.py: Various adaptive filtering techniques including changepoint detection

Base Classes:
    - base_filter.py: Abstract base classes and common functionality

All filters support:
    - JAX JIT compilation for fast execution
    - Automatic differentiation through JAX
    - Immutable state updates
    - Functional programming paradigms
    - Modular callback system for output collection

Example:
    >>> from rebayes_mini.methods.gauss_filter import KalmanFilter
    >>> from rebayes_mini.methods.robust_filter import StudentTFilter
    >>> from rebayes_mini.methods.gaussian_process import SparseGPFilter
"""

__all__ = [
    "base_filter",
    "gauss_filter", 
    "robust_filter",
    "student_t_filter",
    "low_rank_filter",
    "low_rank_filter_revised", 
    "low_rank_last_layer",
    "subspace_filter",
    "subspace_last_layer",
    "ensemble_kalman_filter",
    "gaussian_process",
    "recursive_vi_gauss",
    "replay_sgd",
    "adaptive",
]