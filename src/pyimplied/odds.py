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


@njit(fastmath=True)
def _power_odds(probs: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """
    Power method for adding margin to probabilities.

    Based on Clarke et al. (2017): "Adjusting Bookmaker's Odds to Allow for Overround"
    American Journal of Sports Science, Vol. 5, No. 6, pp. 45-49.
    DOI: 10.11648/j.ajss.20170506.12

    The power method applies the transformation:
        π_i = p_i^τ  (where τ is the power exponent)

    Then scales to achieve target margin:
        r_i = (1+m) * w_i  where w_i = π_i / Σπ_j

    For τ=1 (neutral/proportional distribution):
        The method reduces to: r_i = (1+m) * p_i
        This is equivalent to the multiplicative/basic method.

    For τ > 1: Concentrates more margin on favorites
    For τ < 1: Distributes more margin to longshots

    Advantages over other methods:
    - Never produces probabilities outside [0,1] range
    - Can be applied directly to prices following same power law
    - Accounts for favorite-longshot bias when τ ≠ 1
    - Conceptually simpler than iterative methods like Shin

    Args:
        probs: Array of fair probabilities that sum to 1.0
        margin: Target margin to add (e.g., 0.05 for 5%)

    Returns:
        Array of decimal odds with added margin

    Note:
        Current implementation uses τ=1 (neutral). Future versions
        may expose τ as a parameter for bias adjustment.
    """
    if margin <= 0:
        return 1.0 / probs

    # For τ=1 (neutral), the power method is identical to basic method
    # r_i = (1+m) * p_i^1 = (1+m) * p_i
    scaled_probs = probs * (1.0 + margin)
    return 1.0 / scaled_probs


def implied_odds(
    probabilities: Union[List[float], np.ndarray],
    method: MethodType = Method.BASIC,
    margin: float = 0.0,
    gross_margin: float = 0.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Convert probabilities to bookmaker odds with added margin.

    Transforms fair probabilities into bookmaker odds by adding a specified margin
    (overround/vig) using various mathematical methods.

    Methods Available:
        BASIC: Proportional scaling - multiplies all probabilities by (1+margin)
        WPO: Margin Weights Proportional to the Odds
        BB: Balanced book method (identical to BASIC)
        ADDITIVE: Adds margin equally to each probability
        SHIN: Shin's method using square root transformation (handles bias)
        OR: Odds ratio method with logarithmic transformation
        POWER: Power method (Clarke et al. 2017) - currently τ=1 (equivalent to BASIC)

    The Power method is based on Clarke et al. (2017) "Adjusting Bookmaker's Odds
    to Allow for Overround" and offers theoretical advantages:
    - Never produces invalid probabilities outside [0,1]
    - Conceptually simpler than iterative methods
    - Can handle favorite-longshot bias when τ ≠ 1
    - Direct application to both probabilities and odds

    Args:
        probabilities: Array of probabilities (must sum to <= 1.0)
        method: Conversion method (see Method enum)
        margin: Target margin to add (e.g., 0.05 for 5% overround)
        gross_margin: Gross margin parameter for Shin's method only
        normalize: Whether to normalize probabilities to sum to 1 first

    Returns:
        Array of decimal odds with the specified margin added

    Raises:
        ValueError: If probabilities are invalid (negative, >1, or sum >1)

    Example:
        >>> from pyimplied import implied_odds, Method
        >>> fair_probs = [0.45, 0.35, 0.20]
        >>> odds = implied_odds(fair_probs, method=Method.POWER, margin=0.05)
        >>> print(odds)  # [2.10, 2.72, 4.76] with 5% overround

    References:
        Clarke, S., Kovalchik, S., & Ingram, M. (2017). Adjusting Bookmaker's
        Odds to Allow for Overround. American Journal of Sports Science, 5(6), 45-49.
        DOI: 10.11648/j.ajss.20170506.12
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