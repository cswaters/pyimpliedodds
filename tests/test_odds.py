"""Tests for odds conversion functions."""

import numpy as np
import pytest

from pyimplied import implied_odds, Method


class TestBasicOdds:
    """Test basic odds conversions."""

    def test_basic_method_no_margin(self):
        probs = [0.5, 0.3, 0.2]
        odds = implied_odds(probs, method=Method.BASIC, margin=0.0)

        expected = np.array([2.0, 1/0.3, 5.0])
        np.testing.assert_array_almost_equal(odds, expected, decimal=10)

    def test_basic_method_with_margin(self):
        probs = [0.5, 0.5]
        odds = implied_odds(probs, method=Method.BASIC, margin=0.1)

        # With 10% margin, each probability becomes 0.55
        # So odds should be 1/0.55 = 1.818...
        expected_odd = 1.0 / 0.55
        expected = np.array([expected_odd, expected_odd])

        np.testing.assert_array_almost_equal(odds, expected, decimal=10)

    def test_wpo_method(self):
        probs = [0.5, 0.3, 0.2]
        odds = implied_odds(probs, method=Method.WPO, margin=0.05)

        # Should be positive and finite
        assert all(o > 0 and np.isfinite(o) for o in odds)
        assert len(odds) == 3

        # Check that implied probabilities have higher margin
        implied_probs = 1.0 / odds
        assert np.sum(implied_probs) > 1.0

    def test_bb_method(self):
        probs = [0.5, 0.3, 0.2]
        odds = implied_odds(probs, method=Method.BB, margin=0.05)

        # BB method scales all probabilities by (1 + margin)
        expected_probs = np.array(probs) * 1.05
        expected_odds = 1.0 / expected_probs

        np.testing.assert_array_almost_equal(odds, expected_odds, decimal=10)

    def test_additive_method(self):
        probs = [0.5, 0.5]
        odds = implied_odds(probs, method=Method.ADDITIVE, margin=0.1)

        # Additive adds margin/n to each probability
        # 0.5 + 0.05 = 0.55
        expected_odd = 1.0 / 0.55
        expected = np.array([expected_odd, expected_odd])

        np.testing.assert_array_almost_equal(odds, expected, decimal=10)

    def test_shin_method(self):
        probs = [0.4, 0.3, 0.3]
        odds = implied_odds(probs, method=Method.SHIN, margin=0.05)

        assert all(o > 0 and np.isfinite(o) for o in odds)
        assert len(odds) == 3

    def test_or_method(self):
        probs = [0.4, 0.3, 0.3]
        odds = implied_odds(probs, method=Method.OR, margin=0.05)

        assert all(o > 0 and np.isfinite(o) for o in odds)
        assert len(odds) == 3

    def test_power_method(self):
        probs = [0.4, 0.3, 0.3]
        odds = implied_odds(probs, method=Method.POWER, margin=0.05)

        assert all(o > 0 and np.isfinite(o) for o in odds)
        assert len(odds) == 3


class TestInputValidation:
    """Test input validation."""

    def test_invalid_probabilities(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            implied_odds([0.0, 0.5, 0.5])

        with pytest.raises(ValueError, match="between 0 and 1"):
            implied_odds([1.0, 0.5, 0.5])

        with pytest.raises(ValueError, match="between 0 and 1"):
            implied_odds([-0.1, 0.5, 0.5])

    def test_probabilities_sum_too_high(self):
        with pytest.raises(ValueError, match="sum to more than 1.0"):
            implied_odds([0.6, 0.6])

    def test_insufficient_probabilities(self):
        with pytest.raises(ValueError, match="At least 2 probabilities"):
            implied_odds([0.5])

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="not supported"):
            implied_odds([0.5, 0.5], method=Method.JSD)


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_margin(self):
        probs = [0.5, 0.5]
        odds = implied_odds(probs, method=Method.BASIC, margin=0.0)

        expected = np.array([2.0, 2.0])
        np.testing.assert_array_almost_equal(odds, expected, decimal=10)

    def test_normalization(self):
        """Test probability normalization."""
        probs = [0.4, 0.4]  # Sum to 0.8
        odds = implied_odds(probs, method=Method.BASIC, normalize=True)

        # After normalization: [0.5, 0.5]
        expected = np.array([2.0, 2.0])
        np.testing.assert_array_almost_equal(odds, expected, decimal=10)

    def test_no_normalization(self):
        """Test without normalization."""
        probs = [0.4, 0.4]  # Sum to 0.8
        odds = implied_odds(probs, method=Method.BASIC, normalize=False)

        expected = np.array([2.5, 2.5])  # 1/0.4 = 2.5
        np.testing.assert_array_almost_equal(odds, expected, decimal=10)

    def test_large_margin(self):
        """Test with large margin."""
        probs = [0.5, 0.5]
        odds = implied_odds(probs, method=Method.BASIC, margin=0.5)

        # Each prob becomes 0.75, so odds = 1/0.75 = 1.333...
        expected_odd = 1.0 / 0.75
        expected = np.array([expected_odd, expected_odd])

        np.testing.assert_array_almost_equal(odds, expected, decimal=10)


class TestStringMethods:
    """Test string method names."""

    def test_string_methods(self):
        probs = [0.4, 0.3, 0.3]

        # Test supported string method names
        methods = ["basic", "wpo", "bb", "additive", "shin", "or", "power"]

        for method in methods:
            odds = implied_odds(probs, method=method, margin=0.05)
            assert all(o > 0 and np.isfinite(o) for o in odds)
            assert len(odds) == 3


class TestNumpyArrays:
    """Test with numpy arrays."""

    def test_numpy_input(self):
        probs = np.array([0.5, 0.3, 0.2])
        odds = implied_odds(probs, method=Method.BASIC)

        assert isinstance(odds, np.ndarray)
        expected = np.array([2.0, 1/0.3, 5.0])
        np.testing.assert_array_almost_equal(odds, expected, decimal=10)


class TestRoundTripConsistency:
    """Test that conversions are consistent."""

    def test_basic_round_trip(self):
        from pyimplied import implied_probabilities

        original_odds = [2.0, 3.0, 6.0]

        # Odds -> Probabilities -> Odds
        probs = implied_probabilities(original_odds, method=Method.BASIC)
        back_to_odds = implied_odds(probs, method=Method.BASIC, margin=0.0)

        # Should be close to original (allowing for margin removal)
        raw_probs = 1.0 / np.array(original_odds)
        expected_odds = 1.0 / (raw_probs / np.sum(raw_probs))

        np.testing.assert_array_almost_equal(back_to_odds, expected_odds, decimal=8)