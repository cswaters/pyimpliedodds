"""POWER method: Clarke et al. (2017) power transformation.

Solves for an exponent n such that

    π_i = p_i^n  with  Σ π_i = 1 + margin

where p_i is the raw book probability. Always yields valid probabilities
in [0, 1] and handles favorite-longshot bias when n ≠ 1.
"""

from pyimplied import implied_probabilities, Method


def main() -> None:
    odds = [1.55, 3.90, 7.00]

    probs = implied_probabilities(odds, method=Method.POWER)
    basic = implied_probabilities(odds, method=Method.BASIC)

    print("POWER method")
    print(f"  odds  : {odds}")
    print(f"  BASIC : {[f'{p:.4f}' for p in basic]}")
    print(f"  POWER : {[f'{p:.4f}' for p in probs]}")
    print(f"  sum   : {probs.sum():.6f}")

    print()
    print("  POWER vs BASIC (sign of Δ indicates where vig is reallocated):")
    for i, (o, b, p) in enumerate(zip(odds, basic, probs)):
        diff = (p - b) * 100
        print(f"    leg {i} @ {o:>5}: BASIC={b:.4f}  POWER={p:.4f}  Δ={diff:+.3f}pp")


if __name__ == "__main__":
    main()
