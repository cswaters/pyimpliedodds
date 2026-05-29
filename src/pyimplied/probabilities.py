"""Optimized conversion from bookmaker odds to probabilities."""

import numpy as np
from numba import njit
from typing import Union, List, Optional

from .types import Method, MethodType, Odds, Probabilities
from .utils import (
    solve_root_brent,
    validate_odds_fast,
    js_divergence_fast,
    probit_transform,
    probit_inverse_shift,
    _jsd_inverse,
)


@njit(fastmath=True)
def _basic_probabilities(odds: np.ndarray) -> np.ndarray:
    """Basic method: normalize raw probabilities."""
    raw_probs = 1.0 / odds
    return raw_probs / np.sum(raw_probs)


@njit(fastmath=True)
def _wpo_probabilities(odds: np.ndarray) -> np.ndarray:
    """Weights proportional to odds method."""
    raw_probs = 1.0 / odds
    total_prob = np.sum(raw_probs)
    margin = total_prob - 1.0

    if margin <= 0:
        return raw_probs

    weights = odds / np.sum(odds)
    margin_to_remove = margin * weights
    return raw_probs - margin_to_remove


@njit(fastmath=True)
def _bb_probabilities(odds: np.ndarray) -> np.ndarray:
    """Balanced book method."""
    raw_probs = 1.0 / odds
    total_prob = np.sum(raw_probs)

    if total_prob <= 1.0:
        return raw_probs

    # Remove margin proportionally
    return raw_probs / total_prob


@njit(fastmath=True)
def _additive_probabilities(odds: np.ndarray) -> np.ndarray:
    """Additive method."""
    raw_probs = 1.0 / odds
    total_prob = np.sum(raw_probs)
    margin = total_prob - 1.0

    if margin <= 0:
        return raw_probs

    # Remove margin equally from all outcomes
    margin_per_outcome = margin / len(odds)
    return raw_probs - margin_per_outcome


def _shin_probabilities(
    odds: np.ndarray,
    margin: float = 0.0,
    gross_margin: float = 0.0
) -> np.ndarray:
    """Shin (1993) method for removing bookmaker margin.

    π_i = (√(z² + 4(1-z)·b_i²/Σb) − z) / (2(1-z))
    where b_i = 1/odds_i and z ∈ [0, 1) is the insider-trading parameter.
    """
    raw_probs = 1.0 / odds
    raw_probs = raw_probs.astype(np.float64)
    total_raw = np.sum(raw_probs)

    if total_raw <= 1.0 + margin:
        return raw_probs / total_raw

    # z ∈ [0, 1) is the full valid range for Shin's insider-trading parameter;
    # lopsided / high-margin books legitimately need z > 0.4. The solver guards
    # z ≥ 1 internally, so bracket up to 1 - 1e-9 (was 0.4, which silently fell
    # back to _basic and broke the round trip with the forward map).
    params = np.concatenate([raw_probs, np.array([margin, gross_margin])])
    z = solve_root_brent(params, 0, 1e-9, 1.0 - 1e-9)

    if np.isnan(z):
        return _basic_probabilities(odds)

    two_one_minus_z = 2.0 * (1.0 - z)
    new_probs = np.empty_like(raw_probs)
    for i in range(len(raw_probs)):
        discriminant = z * z + 4.0 * (1.0 - z) * raw_probs[i] * raw_probs[i] / total_raw
        new_probs[i] = (np.sqrt(discriminant) - z) / two_one_minus_z

    if gross_margin > 0:
        new_probs = new_probs * (1.0 + gross_margin) / np.sum(new_probs)

    return new_probs


def _or_probabilities(odds: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Odds ratio method."""
    raw_probs = 1.0 / odds
    raw_probs = raw_probs.astype(np.float64)

    if np.sum(raw_probs) <= 1.0 + margin:
        return raw_probs / np.sum(raw_probs)

    # Prepare parameters for solver
    params = np.concatenate([raw_probs, np.array([margin])])

    # Solve for c parameter
    c = solve_root_brent(params, 1, 0.001, 1000.0)

    if np.isnan(c):
        return _basic_probabilities(odds)

    # Calculate final probabilities
    new_probs = (c * raw_probs) / (1 - raw_probs + c * raw_probs)
    return new_probs


def _power_probabilities(odds: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Power method."""
    raw_probs = 1.0 / odds
    raw_probs = raw_probs.astype(np.float64)

    if np.sum(raw_probs) <= 1.0 + margin:
        return raw_probs / np.sum(raw_probs)

    # Prepare parameters for solver
    params = np.concatenate([raw_probs, np.array([margin])])

    # Solve for n parameter
    n = solve_root_brent(params, 2, 0.1, 10.0)

    if np.isnan(n):
        return _basic_probabilities(odds)

    # Calculate final probabilities
    new_probs = np.power(raw_probs, n)
    return new_probs / np.sum(new_probs)


def _probit_probabilities(odds: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Probit method: subtract a constant on the inverse-normal-CDF scale.

    Raw probabilities are mapped to z-scores via Φ⁻¹; a single shift c is
    solved so that Σ Φ(z_i - c) = 1 + margin, then the shifted z-scores are
    mapped back through Φ.
    """
    raw_probs = 1.0 / odds
    raw_probs = raw_probs.astype(np.float64)

    if np.sum(raw_probs) <= 1.0 + margin:
        return raw_probs / np.sum(raw_probs)

    z_scores = probit_transform(raw_probs)
    params = np.concatenate([z_scores, np.array([margin])])

    c = solve_root_brent(params, 4, -10.0, 20.0)

    if np.isnan(c):
        return _basic_probabilities(odds)

    return probit_inverse_shift(z_scores, c)


def _jsd_probabilities(odds: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Jensen-Shannon distance method (per Jonas Lindström's `implied` R package).

    Each devigged π_i lies below its raw b_i = 1/odds_i, such that the
    binomial Jensen-Shannon distance between (π_i, 1 − π_i) and
    (b_i, 1 − b_i) equals a single scalar d* shared across outcomes.
    d* is chosen so Σ π_i = 1 + margin.
    """
    raw_probs = 1.0 / odds
    raw_probs = raw_probs.astype(np.float64)

    if np.sum(raw_probs) <= 1.0 + margin:
        return raw_probs / np.sum(raw_probs)

    params = np.concatenate([raw_probs, np.array([margin])])
    d = solve_root_brent(params, 3, 1e-9, 0.8)

    if np.isnan(d):
        return _basic_probabilities(odds)

    new_probs = np.empty_like(raw_probs)
    for i in range(len(raw_probs)):
        new_probs[i] = _jsd_inverse(d, raw_probs[i])

    if not np.all(np.isfinite(new_probs)):
        return _basic_probabilities(odds)

    return new_probs


def implied_probabilities(
    odds: Union[List[float], np.ndarray],
    method: MethodType = Method.BASIC,
    margin: float = 0.0,
    gross_margin: float = 0.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Convert bookmaker odds to implied probabilities.

    Args:
        odds: Decimal odds (e.g., [2.0, 3.5, 4.0])
        method: Conversion method
        margin: Target margin for some methods
        gross_margin: Gross margin parameter for Shin's method
        normalize: Whether to normalize probabilities to sum to 1

    Returns:
        Array of implied probabilities

    Raises:
        ValueError: If odds are invalid
    """
    odds_array = np.asarray(odds, dtype=np.float64)

    if not validate_odds_fast(odds_array):
        raise ValueError("All odds must be positive and finite")

    if len(odds_array) < 2:
        raise ValueError("At least 2 odds required")

    # Convert method string to enum if needed
    if isinstance(method, str):
        try:
            method = Method(method)
        except ValueError:
            raise ValueError(f"Unknown method: {method}")

    # Apply conversion method
    if method == Method.BASIC:
        probs = _basic_probabilities(odds_array)
    elif method == Method.WPO:
        probs = _wpo_probabilities(odds_array)
    elif method == Method.BB:
        probs = _bb_probabilities(odds_array)
    elif method == Method.ADDITIVE:
        probs = _additive_probabilities(odds_array)
    elif method == Method.SHIN:
        probs = _shin_probabilities(odds_array, margin, gross_margin)
    elif method == Method.OR:
        probs = _or_probabilities(odds_array, margin)
    elif method == Method.POWER:
        probs = _power_probabilities(odds_array, margin)
    elif method == Method.JSD:
        probs = _jsd_probabilities(odds_array, margin)
    elif method == Method.PROBIT:
        probs = _probit_probabilities(odds_array, margin)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize if requested
    if normalize and np.sum(probs) > 0:
        probs = probs / np.sum(probs)

    return probs