"""ADDITIVE method: subtract the per-outcome share of the overround.

    p_i = raw_i − margin / N

Removes vig as a flat absolute deduction from every outcome. This can
produce negative probabilities for heavy longshots in high-margin books,
so use with care on skewed markets.
"""

from pyimplied import implied_probabilities, Method


def main() -> None:
    odds = [1.80, 3.50, 5.00]
    raw = [1 / o for o in odds]
    overround = sum(raw)

    probs = implied_probabilities(odds, method=Method.ADDITIVE)

    print("ADDITIVE method")
    print(f"  odds       : {odds}")
    print(f"  overround  : {overround:.4f}  (margin = {overround - 1:.4f})")
    print(f"  devigged   : {[f'{p:.4f}' for p in probs]}")
    print(f"  sum        : {probs.sum():.6f}")

    print()
    print("  Skewed market — watch the longshot's share compared to BASIC:")
    basic = implied_probabilities(odds, method=Method.BASIC)
    for i, (o, b, a) in enumerate(zip(odds, basic, probs)):
        print(f"    leg {i} @ {o:>4}: BASIC={b:.4f}  ADDITIVE={a:.4f}")


if __name__ == "__main__":
    main()
