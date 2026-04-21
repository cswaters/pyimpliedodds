"""JSD method: Jensen-Shannon-distance-based blend.

Blends the raw book probabilities with a uniform distribution via

    π_i = λ · p_i + (1 − λ) / N

solving for λ ∈ [0, 1] that makes Σ π_i = 1 + margin. The resulting
π_i are then renormalized so Σ π_i = 1.
"""

from pyimplied import implied_probabilities, Method


def main() -> None:
    # A near-fair three-way market (short-circuit path)
    odds = [2.00, 3.00, 6.00]
    probs = implied_probabilities(odds, method=Method.JSD)
    print("JSD method — fair market")
    print(f"  odds     : {odds}")
    print(f"  devigged : {[f'{p:.4f}' for p in probs]}")
    print(f"  sum      : {probs.sum():.6f}")

    print()
    # A vigged market side-by-side against other methods
    odds = [1.91, 3.60, 4.20]
    print(f"JSD method — vigged market {odds}")
    for m in (Method.BASIC, Method.OR, Method.POWER, Method.JSD):
        p = implied_probabilities(odds, method=m)
        print(f"  {m.value:6s}: {[f'{v:.4f}' for v in p]}")


if __name__ == "__main__":
    main()
