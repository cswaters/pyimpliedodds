"""Benchmark tests for performance evaluation."""

import numpy as np
import pytest

from pyimplied import implied_probabilities, implied_odds, Method


@pytest.mark.benchmark
class TestProbabilityBenchmarks:
    """Benchmark probability conversion functions."""

    @pytest.mark.parametrize("method", [
        Method.BASIC,
        Method.WPO,
        Method.BB,
        Method.ADDITIVE,
        Method.SHIN,
        Method.OR,
        Method.POWER,
        Method.JSD
    ])
    def test_probability_conversion_speed(self, benchmark, method):
        """Benchmark probability conversion methods."""
        odds = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        def convert_probabilities():
            return implied_probabilities(odds, method=method)

        result = benchmark(convert_probabilities)
        assert len(result) == 5
        assert abs(np.sum(result) - 1.0) < 1e-10

    def test_large_array_probabilities(self, benchmark):
        """Benchmark with large number of outcomes."""
        odds = np.random.uniform(1.1, 10.0, 1000)

        def convert_large():
            return implied_probabilities(odds, method=Method.BASIC)

        result = benchmark(convert_large)
        assert len(result) == 1000
        assert abs(np.sum(result) - 1.0) < 1e-10

    def test_many_small_conversions(self, benchmark):
        """Benchmark many small conversions."""
        def many_conversions():
            results = []
            for _ in range(1000):
                odds = [2.0, 3.0, 5.0]
                result = implied_probabilities(odds, method=Method.BASIC)
                results.append(result)
            return results

        results = benchmark(many_conversions)
        assert len(results) == 1000
        assert all(len(r) == 3 for r in results)


@pytest.mark.benchmark
class TestOddsBenchmarks:
    """Benchmark odds conversion functions."""

    @pytest.mark.parametrize("method", [
        Method.BASIC,
        Method.WPO,
        Method.BB,
        Method.ADDITIVE,
        Method.SHIN,
        Method.OR,
        Method.POWER
    ])
    def test_odds_conversion_speed(self, benchmark, method):
        """Benchmark odds conversion methods."""
        probs = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

        def convert_odds():
            return implied_odds(probs, method=method, margin=0.05)

        result = benchmark(convert_odds)
        assert len(result) == 5
        assert all(o > 0 for o in result)

    def test_large_array_odds(self, benchmark):
        """Benchmark with large number of outcomes."""
        probs = np.random.dirichlet([1] * 1000)

        def convert_large():
            return implied_odds(probs, method=Method.BASIC, margin=0.05)

        result = benchmark(convert_large)
        assert len(result) == 1000
        assert all(o > 0 for o in result)

    def test_many_small_odds_conversions(self, benchmark):
        """Benchmark many small odds conversions."""
        def many_conversions():
            results = []
            for _ in range(1000):
                probs = [0.4, 0.35, 0.25]
                result = implied_odds(probs, method=Method.BASIC, margin=0.05)
                results.append(result)
            return results

        results = benchmark(many_conversions)
        assert len(results) == 1000
        assert all(len(r) == 3 for r in results)


@pytest.mark.benchmark
class TestRoundTripBenchmarks:
    """Benchmark round-trip conversions."""

    def test_round_trip_basic(self, benchmark):
        """Benchmark round-trip conversion with basic method."""
        original_odds = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        def round_trip():
            probs = implied_probabilities(original_odds, method=Method.BASIC)
            return implied_odds(probs, method=Method.BASIC)

        result = benchmark(round_trip)
        assert len(result) == 5

    def test_round_trip_complex(self, benchmark):
        """Benchmark round-trip conversion with complex method."""
        original_odds = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        def round_trip():
            probs = implied_probabilities(original_odds, method=Method.SHIN)
            return implied_odds(probs, method=Method.SHIN, margin=0.05)

        result = benchmark(round_trip)
        assert len(result) == 5


@pytest.mark.benchmark
class TestUtilityBenchmarks:
    """Benchmark utility functions."""

    def test_validation_speed(self, benchmark):
        """Benchmark validation functions."""
        from pyimplied.utils import validate_odds_fast

        odds = np.random.uniform(1.1, 10.0, 10000)

        def validate():
            return validate_odds_fast(odds)

        result = benchmark(validate)
        assert result is True

    def test_divergence_calculation_speed(self, benchmark):
        """Benchmark divergence calculations."""
        from pyimplied.utils import js_divergence_fast

        p = np.random.dirichlet([1] * 1000)
        q = np.random.dirichlet([1] * 1000)

        def calculate_divergence():
            return js_divergence_fast(p, q)

        result = benchmark(calculate_divergence)
        assert np.isfinite(result)

    def test_root_solver_speed(self, benchmark):
        """Benchmark root solving."""
        from pyimplied.utils import solve_root_brent

        probs = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        margin = 0.05
        params = np.concatenate([probs, np.array([margin])])

        def solve():
            return solve_root_brent(params, 1, 0.001, 100.0)

        result = benchmark(solve)
        assert np.isfinite(result)