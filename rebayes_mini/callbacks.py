"""
Callback Functions for Bayesian Filters

This module provides callback functions that extract specific information
from the filtering process. Callbacks are used to collect intermediate
results, statistics, or transformed outputs during filter execution.

Callback functions are called at each filtering step with the following signature:
    callback_fn(bel_update, bel_prev, y, x, *args, **kwargs)

Where:
    - bel_update: Updated belief state after processing observation
    - bel_prev: Previous belief state (before update)
    - y: Current observation
    - x: Current input/feature vector
    - *args, **kwargs: Additional arguments passed from the filter

Available Callbacks:
    get_null: Returns None (default callback, no output collected)
    get_updated_mean: Extracts the posterior mean after update
    get_updated_bel: Returns the complete updated belief state
    get_predicted_bel: Returns the predicted belief state (before update)
    get_predicted_mean: Extracts the predicted mean (before update)

Example:
    >>> from rebayes_mini.callbacks import get_updated_mean
    >>> filter.scan(bel, y, X, callback_fn=get_updated_mean)
    # Returns (final_bel, history_of_means)
"""


def get_null(bel_update, bel_prev, y, x, *args, **kwargs):
    return None


def get_updated_mean(bel_update, bel_prev, y, x, *args, **kwargs):
    return bel_update.mean


def get_updated_bel(bel_update, bel_prev, y, x):
    return bel_update


def get_predicted_bel(bel_update, bel_prev, y, x, *args, **kwargs):
    return bel_prev


def get_predicted_mean(bel_update, bel_prev, y, x, *args, **kwargs):
    return bel_prev.mean
