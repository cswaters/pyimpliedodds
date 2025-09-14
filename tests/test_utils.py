"""Tests for utility functions."""

import numpy as np
import pytest

from pyimplied.utils import (
    solve_root_brent,
    kl_divergence_fast,
    js_divergence_fast,
    validate_odds_fast,
    validate_probabilities_fast
)


class TestRootSolver:
    """Test root solving functionality."""

    def test_shin_solver(self):
        """Test Shin method solver."""
        probs = np.array([0.4, 0.3, 0.3])
        margin = 0.05
        gross_margin = 0.0

        params = np.concatenate([probs, np.array([margin, gross_margin])])
        root = solve_root_brent(params, 0, 0.0, 10.0)

        assert np.isfinite(root)
        assert root > 0

    def test_or_solver(self):
        """Test odds ratio method solver."""
        probs = np.array([0.4, 0.3, 0.3])
        margin = 0.05

        params = np.concatenate([probs, np.array([margin])])
        root = solve_root_brent(params, 1, 0.001, 100.0)

        assert np.isfinite(root)
        assert root > 0

    def test_power_solver(self):
        """Test power method solver."""
        probs = np.array([0.4, 0.3, 0.3])
        margin = 0.05

        params = np.concatenate([probs, np.array([margin])])
        root = solve_root_brent(params, 2, 0.1, 10.0)

        assert np.isfinite(root)
        assert root > 0

    def test_jsd_solver(self):
        """Test JSD method solver."""
        probs = np.array([0.4, 0.3, 0.3])
        margin = 0.05

        params = np.concatenate([probs, np.array([margin])])
        root = solve_root_brent(params, 3, 0.0, 1.0)

        assert np.isfinite(root)
        assert 0 <= root <= 1

    def test_no_root_case(self):
        """Test case where no root exists."""
        # Create parameters that don't have a root in the given interval
        probs = np.array([0.4, 0.3, 0.3])
        margin = -0.5  # Negative margin might not have a solution

        params = np.concatenate([probs, np.array([margin])])
        root = solve_root_brent(params, 1, 0.001, 100.0)

        # Should return NaN when no root found
        assert np.isnan(root)


class TestDivergenceFunctions:
    """Test KL and JS divergence functions."""

    def test_kl_divergence_identical(self):
        """Test KL divergence with identical distributions."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.5, 0.3, 0.2])

        kl_div = kl_divergence_fast(p, q)
        assert abs(kl_div) < 1e-10

    def test_kl_divergence_different(self):
        """Test KL divergence with different distributions."""
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.4, 0.4, 0.2])

        kl_div = kl_divergence_fast(p, q)
        assert kl_div > 0

    def test_js_divergence_identical(self):
        """Test JS divergence with identical distributions."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.5, 0.3, 0.2])

        js_div = js_divergence_fast(p, q)
        assert abs(js_div) < 1e-10

    def test_js_divergence_different(self):
        """Test JS divergence with different distributions."""
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.4, 0.4, 0.2])

        js_div = js_divergence_fast(p, q)
        assert js_div > 0

    def test_js_divergence_symmetric(self):
        """Test that JS divergence is symmetric."""
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.4, 0.4, 0.2])

        js_pq = js_divergence_fast(p, q)
        js_qp = js_divergence_fast(q, p)

        np.testing.assert_almost_equal(js_pq, js_qp, decimal=10)

    def test_divergence_with_zeros(self):
        """Test divergence functions handle near-zero values."""
        p = np.array([0.99, 0.005, 0.005])
        q = np.array([0.33, 0.33, 0.34])

        # Should not raise errors
        kl_div = kl_divergence_fast(p, q)
        js_div = js_divergence_fast(p, q)

        assert np.isfinite(kl_div)
        assert np.isfinite(js_div)


class TestValidationFunctions:
    """Test validation functions."""

    def test_validate_odds_valid(self):
        """Test validation with valid odds."""
        odds = np.array([1.5, 2.0, 3.5, 10.0])
        assert validate_odds_fast(odds) is True

    def test_validate_odds_invalid_zero(self):
        """Test validation with zero odds."""
        odds = np.array([0.0, 2.0, 3.0])
        assert validate_odds_fast(odds) is False

    def test_validate_odds_invalid_negative(self):
        """Test validation with negative odds."""
        odds = np.array([-1.0, 2.0, 3.0])
        assert validate_odds_fast(odds) is False

    def test_validate_odds_invalid_infinite(self):
        """Test validation with infinite odds."""
        odds = np.array([np.inf, 2.0, 3.0])
        assert validate_odds_fast(odds) is False

    def test_validate_odds_invalid_nan(self):
        """Test validation with NaN odds."""
        odds = np.array([np.nan, 2.0, 3.0])
        assert validate_odds_fast(odds) is False

    def test_validate_probabilities_valid(self):
        """Test validation with valid probabilities."""
        probs = np.array([0.1, 0.3, 0.5, 0.9])
        assert validate_probabilities_fast(probs) is True

    def test_validate_probabilities_invalid_zero(self):
        """Test validation with zero probability."""
        probs = np.array([0.0, 0.3, 0.5])
        assert validate_probabilities_fast(probs) is False

    def test_validate_probabilities_invalid_one(self):
        """Test validation with probability = 1."""
        probs = np.array([1.0, 0.3, 0.5])
        assert validate_probabilities_fast(probs) is False

    def test_validate_probabilities_invalid_negative(self):
        """Test validation with negative probability."""
        probs = np.array([-0.1, 0.3, 0.5])
        assert validate_probabilities_fast(probs) is False

    def test_validate_probabilities_invalid_over_one(self):
        """Test validation with probability > 1."""
        probs = np.array([1.1, 0.3, 0.5])
        assert validate_probabilities_fast(probs) is False

    def test_validate_probabilities_invalid_infinite(self):
        """Test validation with infinite probability."""
        probs = np.array([np.inf, 0.3, 0.5])
        assert validate_probabilities_fast(probs) is False

    def test_validate_probabilities_invalid_nan(self):
        """Test validation with NaN probability."""
        probs = np.array([np.nan, 0.3, 0.5])
        assert validate_probabilities_fast(probs) is False


class TestPerformance:
    """Performance tests for utility functions."""

    def test_solver_performance(self):
        """Test that solver completes in reasonable time."""
        import time

        probs = np.array([0.4, 0.3, 0.2, 0.1])
        margin = 0.05
        params = np.concatenate([probs, np.array([margin])])

        start_time = time.time()
        root = solve_root_brent(params, 1, 0.001, 100.0)
        end_time = time.time()

        assert end_time - start_time < 0.01  # Should complete in < 10ms
        assert np.isfinite(root)

    def test_divergence_performance(self):
        """Test divergence calculation performance."""
        import time

        p = np.random.dirichlet([1] * 100)  # 100-dimensional probability vector
        q = np.random.dirichlet([1] * 100)

        start_time = time.time()
        for _ in range(1000):
            js_divergence_fast(p, q)
        end_time = time.time()

        # Should complete 1000 calculations in reasonable time
        assert end_time - start_time < 1.0