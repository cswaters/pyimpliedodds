"""Optimized mathematical utilities for probability and odds calculations."""

import math

import numpy as np
from numba import njit
from typing import Optional

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


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
    elif method == 4:  # Probit
        z_scores, margin = params[:-1], params[-1]
        return _probit_solve_func(x, z_scores, margin)
    elif method == 5:  # Shin forward (odds direction)
        probs, margin = params[:-1], params[-1]
        return _shin_forward_solve_func(x, probs, margin)
    elif method == 6:  # WPO forward (odds direction)
        probs, margin = params[:-1], params[-1]
        return _wpo_forward_solve_func(x, probs, margin)
    elif method == 7:  # JSD forward (odds direction)
        probs, margin = params[:-1], params[-1]
        return _jsd_forward_solve_func(x, probs, margin)
    return 0.0


@njit(fastmath=True)
def _wpo_forward_solve_func(T: float, probs: np.ndarray, margin: float) -> float:
    # WPO forward = exact inverse of the devig WPO map. Devig solves
    #   π_i = 1/O_i - margin · O_i / Σ O_j.
    # Fixing T = Σ O_j makes each O_i a quadratic root:
    #   O_i(T) = (-π_i + √(π_i² + 4·margin/T)) / (2·margin/T).
    # The scalar self-consistency root is Σ O_i(T) = T.
    if T <= 0.0:
        return 1e30
    k = margin / T
    two_k = 2.0 * k
    total = 0.0
    for i in range(len(probs)):
        total += (-probs[i] + math.sqrt(probs[i] * probs[i] + 4.0 * k)) / two_k
    return total - T


@njit(fastmath=True)
def _shin_forward_solve_func(z: float, probs: np.ndarray, margin: float) -> float:
    # Forward Shin (odds direction) = exact inverse of the devig formula in
    # _shin_solve_func: b_i = √(S · π_i · ((1-z)·π_i + z)), S = 1 + margin.
    # Find z ∈ [0, 1) so Σ b_i = S.
    if z >= 1.0 or z < 0.0:
        return 1e30
    S = 1.0 + margin
    total = 0.0
    for i in range(len(probs)):
        total += math.sqrt(S * probs[i] * ((1.0 - z) * probs[i] + z))
    return total - S


@njit(fastmath=True)
def _shin_solve_func(z: float, probs: np.ndarray, margin: float, gross_margin: float) -> float:
    # Shin (1993): π_i = (√(z² + 4(1-z)·b_i²/Σb) − z) / (2(1-z)).
    # Match the R `implied` package — b_i² / Σb_j normalization under the sqrt.
    if z >= 1.0 or z < 0.0:
        return 1e30

    bb = 0.0
    for i in range(len(probs)):
        bb += probs[i]

    total = 0.0
    two_one_minus_z = 2.0 * (1.0 - z)

    for i in range(len(probs)):
        discriminant = z * z + 4.0 * (1.0 - z) * probs[i] * probs[i] / bb
        if discriminant < 0:
            return 1e30
        pi = (math.sqrt(discriminant) - z) / two_one_minus_z
        total += pi

    if gross_margin > 0:
        total = total * (1.0 + gross_margin)

    return total - (1.0 + margin)


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
def _jsd_solve_func(d: float, probs: np.ndarray, margin: float) -> float:
    # Outer JSD root: pick distance d so Σ π_i(d, b_i) = 1 + margin,
    # where each π_i is the inverse of binom_jsd(·, b_i) on [eps, b_i].
    total = 0.0
    for i in range(len(probs)):
        pi_i = _jsd_inverse(d, probs[i])
        if pi_i != pi_i:  # NaN propagation
            return 1e30
        total += pi_i
    return total - (1.0 + margin)


@njit(fastmath=True)
def _jsd_forward_solve_func(d: float, probs: np.ndarray, margin: float) -> float:
    # Outer JSD root for the odds direction: pick distance d so Σ b_i = 1 + margin,
    # where each b_i is the inverse of binom_jsd(·, π_i) on [π_i, 1] (book above fair).
    # Exact inverse of _jsd_solve_func / _jsd_inverse, which search below b_i.
    total = 0.0
    for i in range(len(probs)):
        b_i = _jsd_forward_inverse(d, probs[i])
        if b_i != b_i:  # NaN propagation
            return 1e30
        total += b_i
    return total - (1.0 + margin)


@njit(fastmath=True)
def _jsd_forward_inverse(d: float, pi: float) -> float:
    # Inner Brent: find b ∈ [pi, 1-eps] with _binom_jsd(pi, b) = d. The book
    # probability lies above the fair probability, so we search above pi —
    # the mirror of _jsd_inverse, which searches below.
    eps = 1e-10
    if d <= 0.0:
        return pi

    lo = pi
    hi = 1.0 - eps
    if hi <= lo:
        return pi

    f_lo = -d  # _binom_jsd(pi, pi) = 0
    f_hi = _binom_jsd(pi, hi) - d
    if f_hi < 0.0:
        # d exceeds the maximum JSD reachable on [pi, 1]; saturate at hi so the
        # outer solver sees a clean "this d is too big" signal.
        return hi
    if f_lo * f_hi > 0.0:
        return np.nan

    a = lo
    fa = f_lo
    c = hi
    fc = f_hi
    for _ in range(80):
        if abs(c - a) < 1e-12:
            return 0.5 * (a + c)
        mid = 0.5 * (a + c)
        fm = _binom_jsd(pi, mid) - d
        if fa * fm < 0.0:
            c = mid
            fc = fm
        else:
            a = mid
            fa = fm
    return 0.5 * (a + c)


@njit(fastmath=True)
def _binom_jsd(p: float, b: float) -> float:
    # JS distance between Bernoulli(p) and Bernoulli(b).
    # sqrt(0.5·KL((p,1-p)||m) + 0.5·KL((b,1-b)||m)),  m = 0.5·(p+b).
    eps = 1e-15
    if p < eps:
        p = eps
    elif p > 1.0 - eps:
        p = 1.0 - eps
    if b < eps:
        b = eps
    elif b > 1.0 - eps:
        b = 1.0 - eps

    m1 = 0.5 * (p + b)
    m2 = 1.0 - m1
    kl_p = p * math.log(p / m1) + (1.0 - p) * math.log((1.0 - p) / m2)
    kl_b = b * math.log(b / m1) + (1.0 - b) * math.log((1.0 - b) / m2)
    val = 0.5 * kl_p + 0.5 * kl_b
    if val < 0.0:
        val = 0.0
    return math.sqrt(val)


@njit(fastmath=True)
def _jsd_inverse(d: float, b: float) -> float:
    # Inner Brent: find π ∈ [eps, b] with _binom_jsd(π, b) = d. Assumes
    # the true probability lies below the vigged book probability, so we
    # search below b — mirroring the R `implied` package.
    eps = 1e-10
    if d <= 0.0:
        return b

    lo = eps
    hi = b
    if hi <= lo:
        return b

    f_lo = _binom_jsd(lo, b) - d
    f_hi = -d  # _binom_jsd(b, b) = 0
    if f_lo < 0.0:
        # d exceeds the maximum JSD reachable on [eps, b]; saturate at eps so
        # the outer solver sees a clean "this d is too big" signal.
        return lo
    if f_lo * f_hi > 0.0:
        return np.nan

    a = lo
    fa = f_lo
    c = hi
    fc = f_hi
    for _ in range(80):
        if abs(c - a) < 1e-12:
            return 0.5 * (a + c)
        mid = 0.5 * (a + c)
        fm = _binom_jsd(mid, b) - d
        if fa * fm < 0.0:
            c = mid
            fc = fm
        else:
            a = mid
            fa = fm
    return 0.5 * (a + c)


@njit(fastmath=True)
def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation of the standard-normal inverse CDF.
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf

    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
        den = ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
        return num / den
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        num = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q
        den = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
        return num / den
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
        den = ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
        return -num / den


@njit(fastmath=True)
def probit_transform(probs: np.ndarray) -> np.ndarray:
    out = np.empty_like(probs)
    for i in range(len(probs)):
        out[i] = _norm_ppf(probs[i])
    return out


@njit(fastmath=True)
def _probit_solve_func(c: float, z_scores: np.ndarray, margin: float) -> float:
    # Shift each probit z-score by c and sum the resulting normal CDF values;
    # root in c makes the sum equal 1 + margin.
    total = 0.0
    for i in range(len(z_scores)):
        total += 0.5 * (1.0 + math.erf((z_scores[i] - c) * _INV_SQRT2))
    return total - (1.0 + margin)


@njit(fastmath=True)
def probit_inverse_shift(z_scores: np.ndarray, c: float) -> np.ndarray:
    out = np.empty_like(z_scores)
    for i in range(len(z_scores)):
        out[i] = 0.5 * (1.0 + math.erf((z_scores[i] - c) * _INV_SQRT2))
    return out


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


@njit
def validate_odds_fast(odds: np.ndarray) -> bool:
    # fastmath intentionally disabled: np.isinf/np.isnan are no-ops under it.
    for i in range(len(odds)):
        if odds[i] <= 0 or np.isinf(odds[i]) or np.isnan(odds[i]):
            return False
    return True


@njit
def validate_probabilities_fast(probs: np.ndarray) -> bool:
    for i in range(len(probs)):
        if probs[i] <= 0 or probs[i] >= 1 or np.isinf(probs[i]) or np.isnan(probs[i]):
            return False
    return True