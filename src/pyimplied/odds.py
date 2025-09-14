"""Optimized conversion from probabilities to bookmaker odds."""

import numpy as np
from numba import njit
from typing import Union, List

from .types import Method, MethodType, Odds, Probabilities
from .utils import solve_root_brent, validate_probabilities_fast


@njit(fastmath=True)
def _basic_odds(probs: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Basic method: add margin proportionally."""
    if margin <= 0:
        return 1.0 / probs

    # Add margin proportionally to probabilities
    target_sum = 1.0 + margin
    scaled_probs = probs * target_sum
    return 1.0 / scaled_probs


@njit(fastmath=True)
def _wpo_odds(probs: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Weights proportional to odds method."""
    if margin <= 0:
        return 1.0 / probs

    # Calculate raw odds
    raw_odds = 1.0 / probs

    # Weights proportional to odds
    weights = raw_odds / np.sum(raw_odds)

    # Add margin based on weights
    margin_to_add = margin * weights
    new_probs = probs + margin_to_add

    return 1.0 / new_probs


@njit(fastmath=True)
def _bb_odds(probs: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Balanced book method."""
    if margin <= 0:
        return 1.0 / probs

    # Scale probabilities to achieve target margin
    target_sum = 1.0 + margin
    scaled_probs = probs * target_sum
    return 1.0 / scaled_probs


@njit(fastmath=True)
def _additive_odds(probs: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Additive method: add margin equally."""
    if margin <= 0:
        return 1.0 / probs

    # Add margin equally to each probability
    margin_per_outcome = margin / len(probs)
    new_probs = probs + margin_per_outcome

    return 1.0 / new_probs


def _shin_odds(
    probs: np.ndarray,
    margin: float = 0.0,
    gross_margin: float = 0.0
) -> np.ndarray:
    """Shin's method for converting probabilities to odds."""
    if margin <= 0 and gross_margin <= 0:
        return 1.0 / probs

    probs_64 = probs.astype(np.float64)

    # Prepare parameters for solver
    params = np.concatenate([probs_64, np.array([margin, gross_margin])])

    # Solve for z parameter (method 0 for Shin)
    z = solve_root_brent(params, 0, 0.0, 100.0)

    if np.isnan(z):
        return _basic_odds(probs, margin)

    # Apply Shin transformation
    sqrt_probs = np.sqrt(probs_64)
    new_probs = sqrt_probs / (z + sqrt_probs)

    if gross_margin > 0:
        new_probs = new_probs * (1 + gross_margin) / np.sum(new_probs)

    return 1.0 / new_probs


def _or_odds(probs: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Odds ratio method."""
    if margin <= 0:
        return 1.0 / probs

    probs_64 = probs.astype(np.float64)

    # Prepare parameters for solver
    params = np.concatenate([probs_64, np.array([margin])])

    # Solve for c parameter (method 1 for OR)
    c = solve_root_brent(params, 1, 0.001, 1000.0)

    if np.isnan(c):
        return _basic_odds(probs, margin)

    # Apply odds ratio transformation
    new_probs = (c * probs_64) / (1 - probs_64 + c * probs_64)

    return 1.0 / new_probs


def _power_odds(probs: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Power method."""
    if margin <= 0:
        return 1.0 / probs

    probs_64 = probs.astype(np.float64)

    # Prepare parameters for solver
    params = np.concatenate([probs_64, np.array([margin])])

    # Solve for n parameter (method 2 for power)
    n = solve_root_brent(params, 2, 0.1, 10.0)

    if np.isnan(n):
        return _basic_odds(probs, margin)

    # Apply power transformation
    new_probs = np.power(probs_64, n)
    new_probs = new_probs / np.sum(new_probs)

    return 1.0 / new_probs


def implied_odds(
    probabilities: Union[List[float], np.ndarray],
    method: MethodType = Method.BASIC,
    margin: float = 0.0,
    gross_margin: float = 0.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Convert probabilities to bookmaker odds.

    Args:
        probabilities: Array of probabilities (must sum to <= 1.0)
        method: Conversion method
        margin: Target margin to add
        gross_margin: Gross margin parameter for Shin's method
        normalize: Whether to normalize probabilities first

    Returns:
        Array of decimal odds

    Raises:
        ValueError: If probabilities are invalid
    """
    probs_array = np.asarray(probabilities, dtype=np.float64)

    if not validate_probabilities_fast(probs_array):
        raise ValueError("All probabilities must be between 0 and 1")

    if len(probs_array) < 2:
        raise ValueError("At least 2 probabilities required")

    prob_sum = np.sum(probs_array)
    if prob_sum > 1.0001:  # Allow small numerical error
        raise ValueError("Probabilities cannot sum to more than 1.0")

    # Normalize if requested and needed
    if normalize and prob_sum > 0:
        probs_array = probs_array / prob_sum

    # Convert method string to enum if needed
    if isinstance(method, str):
        try:
            method = Method(method)
        except ValueError:
            raise ValueError(f"Unknown method: {method}")

    # Apply conversion method
    if method == Method.BASIC:
        odds = _basic_odds(probs_array, margin)
    elif method == Method.WPO:
        odds = _wpo_odds(probs_array, margin)
    elif method == Method.BB:
        odds = _bb_odds(probs_array, margin)
    elif method == Method.ADDITIVE:
        odds = _additive_odds(probs_array, margin)
    elif method == Method.SHIN:
        odds = _shin_odds(probs_array, margin, gross_margin)
    elif method == Method.OR:
        odds = _or_odds(probs_array, margin)
    elif method == Method.POWER:
        odds = _power_odds(probs_array, margin)
    else:
        raise ValueError(f"Method {method} not supported for odds conversion")

    return odds