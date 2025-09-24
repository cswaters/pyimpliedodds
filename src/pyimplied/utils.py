"""Optimized mathematical utilities for probability and odds calculations."""

import numpy as np
from numba import njit
from typing import Optional


@njit(fastmath=True)
def solve_root_brent(
    func_params: np.ndarray,
    method: int,
    a: float,
    b: float,
    xtol: float = 1e-12,
    maxiter: int = 100
) -> float:
    """
    Optimized Brent's method root finding for specific functions.

    Args:
        func_params: Parameters for the function
        method: Method identifier (0=shin, 1=or, 2=power, 3=jsd)
        a: Lower bound
        b: Upper bound
        xtol: Tolerance
        maxiter: Maximum iterations

    Returns:
        Root value
    """
    fa = _eval_func(a, func_params, method)
    fb = _eval_func(b, func_params, method)

    if fa * fb > 0:
        return np.nan

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    mflag = True
    d = 0.0

    for _ in range(maxiter):
        if abs(b - a) < xtol:
            return b

        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                (c * fa * fb) / ((fc - fa) * (fc - fb))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        # Check conditions for bisection
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < xtol
        cond5 = not mflag and abs(c - d) < xtol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = _eval_func(s, func_params, method)
        d = c
        c = b
        fc = fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return b


@njit(fastmath=True)
def _eval_func(x: float, params: np.ndarray, method: int) -> float:
    """Evaluate specific function based on method."""
    if method == 0:  # Shin
        probs, margin, gross_margin = params[:-2], params[-2], params[-1]
        return _shin_solve_func(x, probs, margin, gross_margin)
    elif method == 1:  # Odds ratio
        probs, margin = params[:-1], params[-1]
        return _or_solve_func(x, probs, margin)
    elif method == 2:  # Power
        probs, margin = params[:-1], params[-1]
        return _power_solve_func(x, probs, margin)
    elif method == 3:  # JSD
        probs, margin = params[:-1], params[-1]
        return _jsd_solve_func(x, probs, margin)
    return 0.0


@njit(fastmath=True)
def _shin_solve_func(z: float, probs: np.ndarray, margin: float, gross_margin: float) -> float:
    """Shin method solver function."""
    sqrt_probs = np.sqrt(probs)
    numerator = sqrt_probs
    denominator = z + sqrt_probs
    new_probs = numerator / denominator

    if gross_margin > 0:
        new_probs = new_probs * (1 + gross_margin) / np.sum(new_probs)

    return np.sum(new_probs) - (1 + margin)


@njit(fastmath=True)
def _or_solve_func(c: float, probs: np.ndarray, margin: float) -> float:
    """Odds ratio method solver function."""
    new_probs = (c * probs) / (1 - probs + c * probs)
    return np.sum(new_probs) - (1 + margin)


@njit(fastmath=True)
def _power_solve_func(k: float, probs: np.ndarray, margin: float) -> float:
    """Power method solver function - find k such that sum(p_i^k) = 1 + margin."""
    powered_probs = np.power(probs, k)
    prob_sum = np.sum(powered_probs)
    return prob_sum - (1.0 + margin)


@njit(fastmath=True)
def _jsd_solve_func(lam: float, probs: np.ndarray, margin: float) -> float:
    """Jensen-Shannon distance method solver function."""
    uniform = np.ones_like(probs) / len(probs)
    mixed = lam * probs + (1 - lam) * uniform
    mixed_sum = np.sum(mixed)
    return mixed_sum - (1 + margin)


@njit(fastmath=True)
def kl_divergence_fast(p: np.ndarray, q: np.ndarray) -> float:
    """Fast KL divergence calculation."""
    result = 0.0
    eps = 1e-15

    for i in range(len(p)):
        p_val = max(p[i], eps)
        q_val = max(q[i], eps)
        result += p_val * np.log(p_val / q_val)

    return result


@njit(fastmath=True)
def js_divergence_fast(p: np.ndarray, q: np.ndarray) -> float:
    """Fast Jensen-Shannon divergence calculation."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence_fast(p, m) + 0.5 * kl_divergence_fast(q, m)


@njit(fastmath=True)
def validate_odds_fast(odds: np.ndarray) -> bool:
    """Fast odds validation."""
    for i in range(len(odds)):
        if odds[i] <= 0 or np.isinf(odds[i]) or np.isnan(odds[i]):
            return False
    return True


@njit(fastmath=True)
def validate_probabilities_fast(probs: np.ndarray) -> bool:
    """Fast probability validation."""
    for i in range(len(probs)):
        if probs[i] <= 0 or probs[i] >= 1 or np.isinf(probs[i]) or np.isnan(probs[i]):
            return False
    return True