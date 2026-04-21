"""SHIN method: Shin (1993) insider-trading model.

Treats the overround as compensation for informed traders. Solves for
the insider parameter z ∈ [0, 1) such that

    π_i = (√(z² + 4(1 − z) · b_i² / Σb_j) − z) / (2(1 − z))

sums to 1 + margin, where b_i = 1/odds_i.
"""

from pyimplied import implied_probabilities, Method


def main() -> None:
    odds = [1.40, 4.50, 9.00]  # heavy favorite + longshot, ~4.8% vig

    probs_basic = implied_probabilities(odds, method=Method.BASIC)
    probs_shin = implied_probabilities(odds, method=Method.SHIN)

    print("SHIN method")
    print(f"  odds  : {odds}")
    print(f"  BASIC : {[f'{p:.4f}' for p in probs_basic]}")
    print(f"  SHIN  : {[f'{p:.4f}' for p in probs_shin]}")
    print(f"  sum   : {probs_shin.sum():.6f}")

    print()
    print("  Per-outcome shift vs BASIC (Shin removes more vig from longshots):")
    for i, (o, b, s) in enumerate(zip(odds, probs_basic, probs_shin)):
        diff = (s - b) * 100
        print(f"    leg {i} @ {o:>5}: BASIC={b:.4f}  SHIN={s:.4f}  Δ={diff:+.3f}pp")


if __name__ == "__main__":
    main()
