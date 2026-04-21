"""BB method: balanced book.

Equivalent to BASIC — each raw probability is divided by the overround.
Kept as a separate method for API completeness; included here to show that
the two produce identical results.
"""

import numpy as np

from pyimplied import implied_probabilities, Method


def main() -> None:
    odds = [2.10, 3.30, 4.50]

    bb = implied_probabilities(odds, method=Method.BB)
    basic = implied_probabilities(odds, method=Method.BASIC)

    print("BB (balanced book) method")
    print(f"  odds  : {odds}")
    print(f"  BB    : {[f'{p:.6f}' for p in bb]}")
    print(f"  BASIC : {[f'{p:.6f}' for p in basic]}")
    print(f"  equal : {np.allclose(bb, basic)}")


if __name__ == "__main__":
    main()
