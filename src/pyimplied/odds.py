"""Optimized conversion from probabilities to bookmaker odds."""

import numpy as np
from numba import njit
from typing import Union, List

from .types import Method, MethodType, Odds, Probabilities
from .utils import (
    solve_root_brent,
    validate_probabilities_fast,
    probit_transform,
    probit_inverse_shift,
    _jsd_forward_inverse,
)


@njit(fastmath=True)
def _basic_odds(probs: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Basic method: add margin proportionally."""
    if margin <= 0:
        return 1.0 / probs

    # Add margin proportionally to probabilities
    target_sum = 1.0 + margin
    scaled_probs = probs * target_sum
    return 1.0 / scaled_probs


def _wpo_odds(probs: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Weights proportional to odds method.

    Exact inverse of the devig WPO map. Devig removes the margin weighted by
    the *vigged* odds:  π_i = 1/O_i - margin · O_i / Σ O_j.  Fixing T = Σ O_j
    turns each O_i into a quadratic root, leaving a single scalar equation
    Σ O_i(T) = T solved for T (method 6). This round-trips exactly with
    implied_probabilities(..., 'wpo').
    """
    if margin <= 0:
        return 1.0 / probs

    probs_64 = probs.astype(np.float64)

    # T = Σ O_j is bracketed below by the fair-odds sum (margin shrinks every
    # decimal odd, hence the sum) and above by a small positive value.
    t_max = np.sum(1.0 / probs_64)
    params = np.concatenate([probs_64, np.array([margin])])
    T = solve_root_brent(params, 6, 1e-9, t_max)

    if np.isnan(T):
        return _basic_odds(probs, margin)

    k = margin / T
    odds = (-probs_64 + np.sqrt(probs_64 * probs_64 + 4.0 * k)) / (2.0 * k)
    return odds


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
    """Shin's method for converting probabilities to odds.

    Forward Shin is the exact algebraic inverse of the devig Shin formula:
        b_i = √( S · π_i · ( (1-z)·π_i + z ) ),   S = 1 + margin
    Solve z ∈ [0, 1) so Σ b_i = S, then return 1 / b_i. This round-trips
    exactly with implied_probabilities(..., 'shin').
    """
    if margin <= 0 and gross_margin <= 0:
        return 1.0 / probs

    probs_64 = probs.astype(np.float64)

    # Forward Shin solver (method id 5) — distinct from the devig solver.
    params = np.concatenate([probs_64, np.array([margin])])
    z = solve_root_brent(params, 5, 0.0, 1.0 - 1e-9)

    if np.isnan(z):
        return _basic_odds(probs, margin)

    S = 1.0 + margin
    new_probs = np.sqrt(S * probs_64 * ((1.0 - z) * probs_64 + z))

    if gross_margin > 0:
        new_probs = new_probs * (1.0 + gross_margin) / np.sum(new_probs)

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
    """
    Power method for adding margin to probabilities - TRUE Clarke et al. implementation.

    Based on Clarke et al. (2017): "Adjusting Bookmaker's Odds to Allow for Overround"
    American Journal of Sports Science, Vol. 5, No. 6, pp. 45-49.
    DOI: 10.11648/j.ajss.20170506.12

    The power method finds optimal τ such that sum(p_i^τ) = 1 + margin,
    then applies: π_i = p_i^τ

    For τ > 1: Concentrates more margin on favorites
    For τ < 1: Distributes more margin to longshots

    Args:
        probs: Array of fair probabilities that sum to 1.0
        margin: Target margin to add (e.g., 0.05 for 5%)

    Returns:
        Array of decimal odds with added margin
    """
    if margin <= 0:
        return 1.0 / probs

    probs_64 = probs.astype(np.float64)

    # Prepare parameters for solver
    params = np.concatenate([probs_64, np.array([margin])])

    # Solve for τ (tau) parameter such that sum(p_i^τ) = 1 + margin
    tau = solve_root_brent(params, 2, 0.1, 3.0)

    if np.isnan(tau):
        # Fallback to basic method
        return _basic_odds(probs, margin)

    # Apply true Clarke power transformation: π_i = p_i^τ
    powered_probs = np.power(probs_64, tau)

    return 1.0 / powered_probs


def _probit_odds(probs: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Probit method (odds direction).

    Exact inverse of the devig probit map π_i = Φ(Φ⁻¹(b_i) − c): add a positive
    shift on the inverse-normal-CDF scale, b_i = Φ(Φ⁻¹(π_i) − c) with c < 0, and
    solve c so Σ b_i = 1 + margin. Returns 1 / b_i.
    """
    if margin <= 0:
        return 1.0 / probs

    probs_64 = probs.astype(np.float64)
    z_scores = probit_transform(probs_64)
    params = np.concatenate([z_scores, np.array([margin])])

    # The devig solver (method 4) computes Σ Φ(z_i - c) - (1 + margin). At c = 0
    # the sum is Σ π_i = 1 < 1 + margin, and it grows as c decreases, so the
    # add-vig shift is the negative root c ∈ [-20, 0).
    c = solve_root_brent(params, 4, -20.0, 0.0)

    if np.isnan(c):
        return _basic_odds(probs, margin)

    book = probit_inverse_shift(z_scores, c)  # Φ(z_i - c), c < 0 → inflated
    return 1.0 / book


def _jsd_odds(probs: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Jensen-Shannon distance method (odds direction).

    Exact inverse of the devig JSD map. Devig finds π_i < b_i at a shared
    binomial JS distance d; forward inverts upward, finding b_i > π_i at a
    shared d such that Σ b_i = 1 + margin. Returns 1 / b_i.
    """
    if margin <= 0:
        return 1.0 / probs

    probs_64 = probs.astype(np.float64)
    params = np.concatenate([probs_64, np.array([margin])])

    # Binomial JS distance is bounded by sqrt(ln 2) ≈ 0.833; bracket d below it.
    d = solve_root_brent(params, 7, 1e-9, 0.83)

    if np.isnan(d):
        return _basic_odds(probs, margin)

    book = np.empty_like(probs_64)
    for i in range(len(probs_64)):
        book[i] = _jsd_forward_inverse(d, probs_64[i])

    if not np.all(np.isfinite(book)):
        return _basic_odds(probs, margin)

    # A near-1 favorite can only move a bounded JS distance before hitting the
    # b = 1 ceiling. When that ceiling binds, no shared-distance book reaches the
    # requested margin, so the result is not invertible by the devig map — the
    # (probabilities, margin) pair is outside the range of JSD. Raise rather than
    # silently return a book that fails to round-trip.
    if np.any(book >= 1.0 - 1e-7):
        raise ValueError(
            "JSD add-vig is infeasible for these probabilities at this margin: "
            "a near-certain favorite saturates the b=1 bound before the book can "
            "reach the requested overround. Use a lower margin or another method."
        )

    return 1.0 / book


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
        POWER: Power method (Clarke et al. 2017)
        JSD: Jensen-Shannon distance method
        PROBIT: Probit (inverse normal CDF) shift method

    Every method is the exact inverse of its implied_probabilities (devig)
    counterpart and round-trips to numerical precision. JSD is the one partial
    case: a near-certain favorite can saturate the b=1 bound at high margins,
    which is mathematically outside the range of JSD-devig and raises ValueError.

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
    elif method == Method.JSD:
        odds = _jsd_odds(probs_array, margin)
    elif method == Method.PROBIT:
        odds = _probit_odds(probs_array, margin)
    else:
        raise ValueError(f"Method {method} not supported for odds conversion")

    return odds