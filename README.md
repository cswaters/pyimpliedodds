# PyImplied

A high-performance Python library for converting between bookmaker odds and probabilities. PyImplied implements multiple methods to remove bookmaker margins and handle biases, optimized with Numba for maximum speed.

## Features

- **Fast**: Optimized with Numba JIT compilation for high-performance numerical operations
- **Multiple Methods**: 8 different conversion methods including Shin's method, odds ratio, power method, and more
- **Comprehensive**: Handles both odds-to-probabilities and probabilities-to-odds conversions
- **Well-Tested**: Extensive test suite with benchmarks
- **Type Safe**: Full type hints for better development experience

## Installation

```bash
pip install pyimplied
```

## Quick Start

```python
import numpy as np
from pyimplied import implied_probabilities, implied_odds, Method

# Convert odds to probabilities
odds = [2.0, 3.5, 4.0]
probs = implied_probabilities(odds, method=Method.SHIN)
print(probs)  # [0.486, 0.320, 0.194]

# Convert probabilities to odds with margin
probs = [0.4, 0.35, 0.25]
odds = implied_odds(probs, method=Method.BASIC, margin=0.05)
print(odds)  # [2.381, 2.721, 3.810]
```

## Available Methods

### For Probability Conversion

- **BASIC**: Simple normalization of raw probabilities
- **WPO**: Margin Weights Proportional to the Odds
- **BB**: Balanced book method
- **ADDITIVE**: Equal margin removal from all outcomes
- **SHIN**: Shin's method using square root transformation
- **OR**: Odds ratio method
- **POWER**: Power method with exponent optimization
- **JSD**: Jensen-Shannon distance method

### For Odds Conversion

- **BASIC**: Proportional margin addition
- **WPO**: Margin Weights Proportional to the Odds
- **BB**: Balanced book method
- **ADDITIVE**: Equal margin addition to all probabilities
- **SHIN**: Shin's method (inverse transformation)
- **OR**: Odds ratio method (inverse transformation)
- **POWER**: Power method (inverse transformation)

## Examples

### Basic Usage

```python
from pyimplied import implied_probabilities, Method

# Example with overround (bookmaker margin)
odds = [1.91, 2.30, 4.50]  # Probabilities sum to ~1.065
probs = implied_probabilities(odds, method=Method.BASIC)
print(f"Sum: {sum(probs):.3f}")  # Sum: 1.000
```

### Advanced Methods

```python
# Shin's method - handles favorite-longshot bias
odds = [1.50, 2.20, 8.00]
probs_shin = implied_probabilities(odds, method=Method.SHIN)

# Odds ratio method with custom margin
probs_or = implied_probabilities(odds, method=Method.OR, margin=0.02)

# Jensen-Shannon distance method
probs_jsd = implied_probabilities(odds, method=Method.JSD)
```

### Converting Back to Odds

```python
from pyimplied import implied_odds

# Start with fair probabilities
fair_probs = [0.45, 0.35, 0.20]

# Add 5% margin using different methods
odds_basic = implied_odds(fair_probs, method=Method.BASIC, margin=0.05)
odds_shin = implied_odds(fair_probs, method=Method.SHIN, margin=0.05)

print("Basic method:", odds_basic)
print("Shin method:", odds_shin)
```

### Working with NumPy Arrays

```python
import numpy as np

# Works seamlessly with NumPy arrays
odds_array = np.array([2.10, 3.40, 4.80, 8.50])
probs_array = implied_probabilities(odds_array, method=Method.POWER)

# Vectorized operations maintain performance
large_odds = np.random.uniform(1.1, 10.0, 10000)
large_probs = implied_probabilities(large_odds, method=Method.BASIC)
```

## Performance

PyImplied is optimized for high-performance applications:

```python
import time
import numpy as np
from pyimplied import implied_probabilities, Method

# Benchmark with large array
odds = np.random.uniform(1.1, 10.0, 100000)

start = time.time()
probs = implied_probabilities(odds, method=Method.BASIC)
end = time.time()

print(f"Converted {len(odds)} odds in {end-start:.4f} seconds")
```

## API Reference

### implied_probabilities

```python
implied_probabilities(
    odds: Union[List[float], np.ndarray],
    method: Union[Method, str] = Method.BASIC,
    margin: float = 0.0,
    gross_margin: float = 0.0,
    normalize: bool = True
) -> np.ndarray
```

Convert bookmaker odds to implied probabilities.

**Parameters:**
- `odds`: Decimal odds (e.g., [2.0, 3.5, 4.0])
- `method`: Conversion method (see Methods section)
- `margin`: Target margin for some methods
- `gross_margin`: Gross margin parameter for Shin's method
- `normalize`: Whether to normalize probabilities to sum to 1

**Returns:** NumPy array of implied probabilities

### implied_odds

```python
implied_odds(
    probabilities: Union[List[float], np.ndarray],
    method: Union[Method, str] = Method.BASIC,
    margin: float = 0.0,
    gross_margin: float = 0.0,
    normalize: bool = True
) -> np.ndarray
```

Convert probabilities to bookmaker odds.

**Parameters:**
- `probabilities`: Array of probabilities (must sum to ≤ 1.0)
- `method`: Conversion method
- `margin`: Target margin to add
- `gross_margin`: Gross margin parameter for Shin's method
- `normalize`: Whether to normalize probabilities first

**Returns:** NumPy array of decimal odds

## Method Details

### Shin's Method

Shin's method (1993) addresses the favorite-longshot bias commonly observed in betting markets. It uses a square root transformation that tends to increase the probabilities of longshots relative to favorites.

### Odds Ratio Method

The odds ratio method transforms probabilities using a scaling factor that maintains the relative odds between outcomes while adjusting the overall margin.

### Power Method

The power method applies an exponent to the probabilities, with the exponent chosen to achieve the desired probability sum.

### Jensen-Shannon Distance Method

This method finds the optimal mixture between the bookmaker's probabilities and a uniform distribution that minimizes the Jensen-Shannon divergence.

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- Numba ≥ 0.56.0

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see the GitHub repository for guidelines.

## References

- Shin, H. S. (1993). Measuring the incidence of insider trading in a market for state-contingent claims. Economic Journal, 103, 142-153.
- Jullien, B., & Salanié, B. (2000). Estimating preferences under risk: The case of racetrack bettors. Journal of Political Economy, 108(4), 601-635.
