"""Example: Adding margin to devigged probabilities using the POWER method."""

import numpy as np
from pyimplied import implied_odds, Method


def add_power_margin_example():
    """Add 4.58% margin to devigged probabilities using POWER method."""

    # Devigged probabilities (fair probabilities that sum to 1.0)
    teams = ["Buffalo Bills", "Miami Dolphins", "New England Patriots", "New York Jets"]
    fair_probs = [0.9196, 0.0107, 0.0591, 0.0107]

    print("=== Adding 4.58% Margin Using POWER Method ===")
    print(f"Original devigged probabilities:")
    for team, prob in zip(teams, fair_probs):
        print(f"  {team}: {prob:.4f} ({prob*100:.2f}%)")
    print(f"Sum: {sum(fair_probs):.6f}")
    print()

    # Target margin (4.58%)
    target_margin = 0.0458

    # Debug: test what k value we need
    print("Debug: Testing manual power calculation...")
    test_k = 0.8  # Test value
    manual_powered = [p**test_k for p in fair_probs]
    manual_sum = sum(manual_powered)
    print(f"With k={test_k}: sum = {manual_sum:.6f}, target = {1+target_margin:.6f}")

    # Add margin using POWER method
    odds_with_margin = implied_odds(
        probabilities=fair_probs,
        method=Method.POWER,
        margin=target_margin
    )

    # Calculate the actual overround to verify
    actual_overround = sum(1.0 / odd for odd in odds_with_margin)
    actual_margin = actual_overround - 1.0

    def decimal_to_american(decimal_odds):
        """Convert decimal odds to American odds format."""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))

    print("Results with 4.58% margin (POWER method):")
    print("Team                    | Decimal | American | Implied Prob")
    print("-" * 65)
    for team, odd in zip(teams, odds_with_margin):
        implied_prob = 1.0 / odd
        american_odds = decimal_to_american(odd)
        print(f"{team:22} | {odd:7.3f} | {american_odds:8} | {implied_prob:.4f} ({implied_prob*100:.2f}%)")

    print(f"\nOverround: {actual_overround:.6f}")
    print(f"Actual margin: {actual_margin:.4f} ({actual_margin*100:.2f}%)")
    print(f"Target margin: {target_margin:.4f} ({target_margin*100:.2f}%)")
    print(f"Difference: {abs(actual_margin - target_margin):.6f}")

    # Show comparison with other methods
    print(f"\n=== Comparison with Other Methods ===")
    methods_to_test = [Method.BASIC, Method.ADDITIVE, Method.WPO, Method.BB]

    for method in methods_to_test:
        odds = implied_odds(fair_probs, method=method, margin=target_margin)
        overround = sum(1.0 / odd for odd in odds)
        margin = overround - 1.0
        print(f"{method.value:8}: overround = {overround:.6f}, margin = {margin:.4f} ({margin*100:.2f}%)")


if __name__ == "__main__":
    add_power_margin_example()