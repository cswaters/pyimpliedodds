"""
PyImplied: Convert between bookmaker odds and probabilities

A Python library for converting bookmaker odds to probabilities and vice versa,
with support for multiple methods to remove bookmaker margins and handle biases.
"""

from .probabilities import implied_probabilities
from .odds import implied_odds
from .types import Method

__version__ = "0.1.0"
__all__ = ["implied_probabilities", "implied_odds", "Method"]