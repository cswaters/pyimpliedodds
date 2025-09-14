"""Basic usage examples for PyImplied."""

import numpy as np
from pyimplied import implied_probabilities, implied_odds, Method


def basic_examples():
    """Demonstrate basic functionality."""
    print("=== Basic Odds to Probabilities ===")

    # Example 1: Basic conversion
    odds = [2.0, 3.0, 6.0]
    print(f"Odds: {odds}")

    probs = implied_probabilities(odds, method=Method.BASIC)
    print(f"Probabilities (Basic): {probs}")
    print(f"Sum: {sum(probs):.6f}")
    print()

    # Example 2: Compare different methods
    print("=== Method Comparison ===")
    methods = [Method.BASIC, Method.SHIN, Method.OR, Method.POWER]

    for method in methods:
        probs = implied_probabilities(odds, method=method)
        print(f"{method.value:8}: {[f'{p:.4f}' for p in probs]}")
    print()


def margin_examples():
    """Demonstrate margin handling."""
    print("=== Margin Examples ===")

    # Fair probabilities
    fair_probs = [0.45, 0.35, 0.20]
    print(f"Fair probabilities: {fair_probs}")
    print(f"Sum: {sum(fair_probs)}")

    # Add margins using different methods
    margins = [0.05, 0.10, 0.15]

    for margin in margins:
        print(f"\nWith {margin*100}% margin:")
        for method in [Method.BASIC, Method.SHIN, Method.ADDITIVE]:
            odds = implied_odds(fair_probs, method=method, margin=margin)
            overround = sum(1/o for o in odds)
            print(f"  {method.value:8}: overround = {overround:.4f}")


def performance_example():
    """Demonstrate performance with large arrays."""
    print("=== Performance Example ===")

    import time

    # Large array of odds
    n_odds = 100000
    odds = np.random.uniform(1.1, 10.0, n_odds)

    print(f"Converting {n_odds:,} odds...")

    # Time the conversion
    start = time.time()
    probs = implied_probabilities(odds, method=Method.BASIC)
    end = time.time()

    print(f"Completed in {end - start:.4f} seconds")
    print(f"Rate: {n_odds / (end - start):,.0f} conversions/second")
    print(f"Probability sum: {np.sum(probs):.6f}")


def round_trip_example():
    """Demonstrate round-trip conversion accuracy."""
    print("=== Round-Trip Accuracy ===")

    original_odds = [2.1, 3.4, 4.8]
    print(f"Original odds: {original_odds}")

    # Convert to probabilities and back
    probs = implied_probabilities(original_odds, method=Method.BASIC)
    back_to_odds = implied_odds(probs, method=Method.BASIC, margin=0.0)

    print(f"After round-trip: {[f'{o:.6f}' for o in back_to_odds]}")

    # Calculate the expected result (normalized odds)
    raw_probs = 1.0 / np.array(original_odds)
    normalized_probs = raw_probs / np.sum(raw_probs)
    expected_odds = 1.0 / normalized_probs

    print(f"Expected: {[f'{o:.6f}' for o in expected_odds]}")

    # Check accuracy
    max_diff = np.max(np.abs(back_to_odds - expected_odds))
    print(f"Maximum difference: {max_diff:.10f}")


if __name__ == "__main__":
    basic_examples()
    margin_examples()
    performance_example()
    round_trip_example()