"""PROBIT method: shift on the inverse-normal-CDF scale.

Removes vig in nonlinear space rather than on raw probabilities:

    z_i = Φ⁻¹(p_i)          (raw probabilities → probit z-scores)
    find c s.t. Σ Φ(z_i − c) = 1 + margin
    π_i = Φ(z_i − c)

Reference: "How Wide Is the Goalie? Quantifying Vig in Nonlinear Space",
Plus EV Analytics, June 2025.
https://plusevanalytics.wordpress.com/2025/06/09/how-wide-is-the-goalie-quantifying-vig-in-nonlinear-space/
"""

from pyimplied import implied_probabilities, Method


def american_to_decimal(american: int) -> float:
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / -american)


def main() -> None:
    # Worked example 1: -110 / -110 → 50 / 50
    odds = [american_to_decimal(-110), american_to_decimal(-110)]
    probs = implied_probabilities(odds, method=Method.PROBIT)
    print("PROBIT — -110 / -110")
    print(f"  devigged : {[f'{p:.4f}' for p in probs]}")
    print(f"  sum      : {probs.sum():.6f}")

    print()

    # Worked example 2: -50000 / +4000 → ~99.2% / ~0.8%
    odds = [american_to_decimal(-50000), american_to_decimal(4000)]
    probs = implied_probabilities(odds, method=Method.PROBIT)
    print("PROBIT — -50000 / +4000  (heavy-favorite stress test)")
    print(f"  devigged : {[f'{p:.4f}' for p in probs]}")
    print(f"  sum      : {probs.sum():.6f}")

    print()

    # Three-way comparison against every other method
    odds = [1.55, 3.90, 7.00]
    print(f"PROBIT vs other methods on {odds}:")
    for m in (Method.BASIC, Method.SHIN, Method.OR, Method.POWER, Method.JSD, Method.PROBIT):
        p = implied_probabilities(odds, method=m)
        print(f"  {m.value:8s}: {[f'{v:.4f}' for v in p]}")


if __name__ == "__main__":
    main()
