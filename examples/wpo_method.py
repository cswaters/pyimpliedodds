"""WPO method: margin weights proportional to the odds.

The margin that needs to be removed is allocated across outcomes in
proportion to each outcome's decimal odds — so longshots (higher odds)
absorb a larger share of the overround than favorites.
"""

from pyimplied import implied_probabilities, Method


def main() -> None:
    odds = [1.50, 3.80, 7.50]
    probs = implied_probabilities(odds, method=Method.WPO)

    print("WPO method")
    print(f"  odds     : {odds}")
    print(f"  devigged : {[f'{p:.4f}' for p in probs]}")
    print(f"  sum      : {probs.sum():.6f}")

    print()
    print("  Compare the *change* vs BASIC — WPO shifts more vig onto longshots:")
    basic = implied_probabilities(odds, method=Method.BASIC)
    for i, (o, b, w) in enumerate(zip(odds, basic, probs)):
        diff = (w - b) * 100
        print(f"    leg {i} @ {o:>5}: BASIC={b:.4f}  WPO={w:.4f}  Δ={diff:+.3f}pp")


if __name__ == "__main__":
    main()
