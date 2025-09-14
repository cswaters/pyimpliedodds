"""Tests for probability conversion functions."""

import numpy as np
import pytest

from pyimplied import implied_probabilities, Method


class TestBasicProbabilities:
    """Test basic probability conversions."""

    def test_basic_method(self):
        odds = [2.0, 3.0, 6.0]
        probs = implied_probabilities(odds, method=Method.BASIC)

        # Basic method normalizes raw probabilities
        raw = np.array([0.5, 1/3, 1/6])
        expected = raw / np.sum(raw)

        np.testing.assert_array_almost_equal(probs, expected, decimal=10)

    def test_wpo_method(self):
        odds = [2.0, 3.0, 6.0]
        probs = implied_probabilities(odds, method=Method.WPO)

        # Should sum to 1.0 and be different from basic
        assert abs(np.sum(probs) - 1.0) < 1e-10
        assert len(probs) == 3
        assert all(p > 0 for p in probs)

    def test_bb_method(self):
        odds = [2.0, 3.0, 6.0]
        probs = implied_probabilities(odds, method=Method.BB)

        # BB method should normalize by total probability
        raw = np.array([0.5, 1/3, 1/6])
        expected = raw / np.sum(raw)

        np.testing.assert_array_almost_equal(probs, expected, decimal=10)

    def test_additive_method(self):
        odds = [2.0, 3.0, 6.0]
        probs = implied_probabilities(odds, method=Method.ADDITIVE)

        assert abs(np.sum(probs) - 1.0) < 1e-10
        assert len(probs) == 3
        assert all(p > 0 for p in probs)

    def test_shin_method(self):
        odds = [2.0, 3.0, 6.0]
        probs = implied_probabilities(odds, method=Method.SHIN)

        assert abs(np.sum(probs) - 1.0) < 1e-10
        assert len(probs) == 3
        assert all(p > 0 for p in probs)

    def test_or_method(self):
        odds = [2.0, 3.0, 6.0]
        probs = implied_probabilities(odds, method=Method.OR)

        assert abs(np.sum(probs) - 1.0) < 1e-10
        assert len(probs) == 3
        assert all(p > 0 for p in probs)

    def test_power_method(self):
        odds = [2.0, 3.0, 6.0]
        probs = implied_probabilities(odds, method=Method.POWER)

        assert abs(np.sum(probs) - 1.0) < 1e-10
        assert len(probs) == 3
        assert all(p > 0 for p in probs)

    def test_jsd_method(self):
        odds = [2.0, 3.0, 6.0]
        probs = implied_probabilities(odds, method=Method.JSD)

        assert abs(np.sum(probs) - 1.0) < 1e-10
        assert len(probs) == 3
        assert all(p > 0 for p in probs)


class TestInputValidation:
    """Test input validation."""

    def test_invalid_odds(self):
        with pytest.raises(ValueError, match="positive and finite"):
            implied_probabilities([0.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="positive and finite"):
            implied_probabilities([-1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="positive and finite"):
            implied_probabilities([np.inf, 2.0, 3.0])

    def test_insufficient_odds(self):
        with pytest.raises(ValueError, match="At least 2 odds"):
            implied_probabilities([2.0])

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            implied_probabilities([2.0, 3.0], method="unknown")


class TestEdgeCases:
    """Test edge cases."""

    def test_fair_odds(self):
        """Test odds that already sum to 1.0."""
        odds = [2.0, 2.0]  # probabilities sum to 1.0
        probs = implied_probabilities(odds, method=Method.BASIC)

        expected = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(probs, expected, decimal=10)

    def test_high_margin_odds(self):
        """Test odds with high margins."""
        odds = [1.5, 1.5, 1.5]  # Very high margin
        probs = implied_probabilities(odds, method=Method.SHIN)

        assert abs(np.sum(probs) - 1.0) < 1e-10
        assert all(p > 0 for p in probs)

    def test_large_number_of_outcomes(self):
        """Test with many outcomes."""
        odds = [2.0] * 10
        probs = implied_probabilities(odds, method=Method.BASIC)

        assert abs(np.sum(probs) - 1.0) < 1e-10
        assert len(probs) == 10
        assert all(abs(p - 0.1) < 1e-10 for p in probs)


class TestStringMethods:
    """Test string method names."""

    def test_string_methods(self):
        odds = [2.0, 3.0, 6.0]

        # Test all string method names
        methods = ["basic", "wpo", "bb", "additive", "shin", "or", "power", "jsd"]

        for method in methods:
            probs = implied_probabilities(odds, method=method)
            assert abs(np.sum(probs) - 1.0) < 1e-10
            assert len(probs) == 3


class TestNumpyArrays:
    """Test with numpy arrays."""

    def test_numpy_input(self):
        odds = np.array([2.0, 3.0, 6.0])
        probs = implied_probabilities(odds, method=Method.BASIC)

        assert isinstance(probs, np.ndarray)
        assert abs(np.sum(probs) - 1.0) < 1e-10


class TestMarginParameter:
    """Test margin parameter functionality."""

    def test_shin_with_margin(self):
        odds = [2.0, 3.0, 6.0]
        probs = implied_probabilities(odds, method=Method.SHIN, margin=0.05)

        # Should still sum to 1.0 after normalization
        assert abs(np.sum(probs) - 1.0) < 1e-10

    def test_or_with_margin(self):
        odds = [2.0, 3.0, 6.0]
        probs = implied_probabilities(odds, method=Method.OR, margin=0.05)

        assert abs(np.sum(probs) - 1.0) < 1e-10