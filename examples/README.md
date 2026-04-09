# Examples Directory

This directory contains practical examples demonstrating the usage of various filters in rebayes-mini.

## Available Examples

### 01_kalman_filter_basic.py
- Basic Kalman filter usage for 1D state tracking
- Synthetic data generation 
- Visualization of filtering results
- Error analysis

### 02_robust_filtering.py
- Comparison between standard Kalman filter and robust Student-t filter
- Handling of contaminated/outlier observations
- Performance comparison on noisy data

### 03_online_gp_regression.py (TODO)
- Online Gaussian Process regression example
- Streaming data processing
- Uncertainty quantification

### 04_neural_network_last_layer.py (TODO)
- Bayesian treatment of neural network last layer
- Online learning with neural networks
- Comparison with standard gradient descent

### 05_changepoint_detection.py (TODO)
- Bayesian Online Changepoint Detection (BOCD)
- Detecting shifts in data streams
- Adaptive filtering

## Running Examples

Each example is self-contained and can be run independently:

```bash
cd examples
python 01_kalman_filter_basic.py
python 02_robust_filtering.py
```

## Requirements

All examples require:
- JAX and JAX-compatible libraries
- Matplotlib for visualization
- NumPy for numerical operations

Install with:
```bash
pip install jax matplotlib numpy
```

## Output

Examples generate both console output (statistics) and saved plots showing:
- Filter performance
- Comparison with ground truth
- Error analysis
- Visualization of key concepts