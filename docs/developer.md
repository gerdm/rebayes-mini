# Developer Guide

## Contributing to Rebayes Mini

Thank you for your interest in contributing to rebayes-mini! This guide will help you understand the codebase structure and development practices.

## Project Structure

```
rebayes_mini/
├── __init__.py           # Main package interface
├── states.py            # Data structures for filter states
├── callbacks.py         # Output extraction functions
├── methods/             # Filter implementations
│   ├── __init__.py      # Methods package interface
│   ├── base_filter.py   # Abstract base classes
│   ├── gauss_filter.py  # Kalman filter variants
│   ├── robust_filter.py # Robust filtering methods
│   ├── low_rank_*.py    # Low-rank approximations
│   ├── gaussian_process.py # GP-based methods
│   └── adaptive.py      # Adaptive/changepoint methods
└── datasets/            # Synthetic datasets
    ├── __init__.py      # Datasets package interface
    └── linear_ssm.py    # Linear state space models
```

## Design Principles

### 1. Functional Programming
- All functions should be pure (no side effects)
- Use immutable data structures (chex.dataclass)
- State updates return new state objects

### 2. JAX Compatibility
- All arrays should be JAX arrays (jnp.array)
- Functions should be JAX-transformable (jit, grad, vmap)
- Use jax.random for random number generation

### 3. Consistent Interface
All filters should implement the `BaseFilter` interface:

```python
class MyFilter(BaseFilter):
    def init_bel(self, *args) -> State:
        """Initialize belief state"""
        
    def predict(self, bel: State) -> State:
        """Prediction step"""
        
    def update(self, bel: State, y: Array, x: Array) -> State:
        """Update step"""
        
    def sample_fn(self, key: PRNGKey, bel: State) -> Callable:
        """Sample function from posterior"""
```

### 4. Type Safety
- Use chex.Array for array types
- Use chex.dataclass for state structures
- Include type hints for all function signatures

## Adding a New Filter

### Step 1: Create the Filter Class
```python
# rebayes_mini/methods/my_new_filter.py
import jax
import jax.numpy as jnp
import chex
from .base_filter import BaseFilter
from ..states import GaussState

class MyNewFilter(BaseFilter):
    def __init__(self, param1, param2):
        """
        My new filtering method.
        
        Args:
            param1: Description of parameter 1
            param2: Description of parameter 2
        """
        self.param1 = param1
        self.param2 = param2
    
    def init_bel(self, initial_mean, initial_cov=1.0):
        """Initialize belief state."""
        return GaussState(
            mean=initial_mean,
            cov=jnp.eye(len(initial_mean)) * initial_cov
        )
    
    def predict(self, bel):
        """Prediction step."""
        # Implement prediction logic
        return bel
    
    def update(self, bel, y, x):
        """Update step.""" 
        # Implement update logic
        return bel
    
    def sample_fn(self, key, bel):
        """Sample function from posterior."""
        # Implement sampling logic
        pass
```

### Step 2: Add Tests
```python
# tests/test_my_new_filter.py
import jax
import jax.numpy as jnp
import pytest
from rebayes_mini.methods.my_new_filter import MyNewFilter

def test_init():
    """Test filter initialization."""
    filter = MyNewFilter(param1=1.0, param2=2.0)
    assert filter.param1 == 1.0
    assert filter.param2 == 2.0

def test_init_bel():
    """Test belief initialization."""
    filter = MyNewFilter(param1=1.0, param2=2.0)
    bel = filter.init_bel(jnp.array([0.0, 1.0]))
    assert bel.mean.shape == (2,)
    assert bel.cov.shape == (2, 2)

def test_step():
    """Test single filtering step."""
    filter = MyNewFilter(param1=1.0, param2=2.0)
    bel = filter.init_bel(jnp.array([0.0]))
    y = 1.0
    x = jnp.array([1.0])
    
    bel_new, _ = filter.step(bel, y, x, lambda *args: None)
    # Add assertions about expected behavior
```

### Step 3: Update Documentation
1. Add the new filter to `methods/__init__.py`
2. Update the API documentation in `docs/api.md`
3. Add usage example to `examples/` directory

## State Structures

### Creating New States
If your filter needs a new state structure:

```python
# In states.py
@chex.dataclass
class MyFilterState:
    """State for my custom filter."""
    mean: chex.Array           # Posterior mean
    cov: chex.Array            # Posterior covariance  
    auxiliary_param: float     # Custom parameter
    history: chex.Array        # Historical information
```

### State Design Guidelines
- Use descriptive field names
- Include docstring explaining the state
- Keep states minimal (only necessary information)
- Use appropriate array shapes and types

## Testing Guidelines

### Test Structure
```python
def test_function_name():
    """Brief description of what is being tested."""
    # Arrange - set up test data
    
    # Act - call the function being tested
    
    # Assert - check the results
```

### Key Tests to Include
1. **Initialization tests**: Verify correct setup
2. **Shape tests**: Check array dimensions
3. **Mathematical properties**: Verify algorithmic correctness
4. **Edge cases**: Test boundary conditions
5. **JAX compatibility**: Ensure jit/grad work

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_my_new_filter.py

# Run with coverage
python -m pytest --cov=rebayes_mini tests/
```

## Code Style

### Formatting
- Use 4 spaces for indentation
- Maximum line length: 88 characters
- Use descriptive variable names
- Follow PEP 8 conventions

### Documentation
- Include docstrings for all public functions
- Use Google-style docstrings
- Provide examples in docstrings when helpful

```python
def my_function(param1: float, param2: jnp.Array) -> jnp.Array:
    """
    Brief description of the function.
    
    Longer description if needed, explaining the algorithm,
    mathematical background, or implementation details.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Returns:
        Description of return value
        
    Example:
        >>> result = my_function(1.0, jnp.array([1, 2, 3]))
        >>> print(result.shape)
        (3,)
    """
```

## Performance Considerations

### JAX Best Practices
1. **Use pure functions**: Avoid side effects for jit compatibility
2. **Avoid Python loops**: Use jax.lax.scan for iteration
3. **Batch operations**: Vectorize computations when possible
4. **Minimize allocations**: Reuse arrays when safe

### Memory Efficiency
1. **Low-rank approximations**: For high-dimensional problems
2. **In-place updates**: When mathematically valid
3. **Sparse representations**: For sparse problems

### Benchmarking
Include performance benchmarks for new filters:

```python
def benchmark_my_filter():
    """Benchmark filter performance."""
    import time
    
    # Set up problem
    filter = MyNewFilter(param1=1.0, param2=2.0)
    
    # Time the critical operations
    start = time.time()
    # ... run filter ...
    elapsed = time.time() - start
    
    print(f"Filter took {elapsed:.4f} seconds")
```

## Release Process

1. **Update version** in `__init__.py` and `pyproject.toml`
2. **Update CHANGELOG** with new features and fixes
3. **Run full test suite** to ensure everything works
4. **Update documentation** with any API changes
5. **Create release** on GitHub with tag

## Getting Help

- **Issues**: Open GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub discussions for questions
- **Email**: Contact maintainers for sensitive issues

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions. We aim to create a welcoming environment for all contributors regardless of background or experience level.