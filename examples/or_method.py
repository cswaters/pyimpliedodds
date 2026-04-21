"""OR method: odds-ratio transformation.

Solves for a scalar c > 0 such that

    π_i = c · p_i / (1 − p_i + c · p_i)

sums to 1 + margin, where p_i is the raw book probability. Preserves
the pairwise odds ratios between outcomes while scaling the implied
probabilities down to a fair distribution.
"""

from pyimplied import implied_probabilities, Method


def main() -> None:
    odds = [1.75, 3.20, 5.50]

    probs = implied_probabilities(odds, method=Method.OR)
    basic = implied_probabilities(odds, method=Method.BASIC)

    print("OR (odds ratio) method")
    print(f"  odds  : {odds}")
    print(f"  BASIC : {[f'{p:.4f}' for p in basic]}")
    print(f"  OR    : {[f'{p:.4f}' for p in probs]}")
    print(f"  sum   : {probs.sum():.6f}")

    print()
    print("  Odds ratios between outcomes are preserved:")
    for i in range(len(odds) - 1):
        raw_i, raw_j = 1 / odds[i], 1 / odds[i + 1]
        book_ratio = (raw_i / (1 - raw_i)) / (raw_j / (1 - raw_j))
        fair_ratio = (probs[i] / (1 - probs[i])) / (probs[i + 1] / (1 - probs[i + 1]))
        print(f"    leg{i}:leg{i+1}  book={book_ratio:.4f}  fair={fair_ratio:.4f}")


if __name__ == "__main__":
    main()
