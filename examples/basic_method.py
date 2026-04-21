"""BASIC method: normalize raw implied probabilities.

    p_i = (1/odds_i) / Σ_j (1/odds_j)

The simplest devig: divide every book probability by the overround. Every
outcome keeps the same *share* of the market's implied probability.
"""

from pyimplied import implied_probabilities, Method


def main() -> None:
    odds = [1.91, 3.60, 4.20]  # 3-way market with ~5% vig
    raw = [1 / o for o in odds]
    overround = sum(raw)

    probs = implied_probabilities(odds, method=Method.BASIC)

    print("BASIC method")
    print(f"  odds       : {odds}")
    print(f"  raw probs  : {[f'{p:.4f}' for p in raw]}")
    print(f"  overround  : {overround:.4f}")
    print(f"  devigged   : {[f'{p:.4f}' for p in probs]}")
    print(f"  sum        : {probs.sum():.6f}")


if __name__ == "__main__":
    main()
