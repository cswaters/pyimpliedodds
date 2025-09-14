"""Type definitions for PyImplied."""

from enum import Enum
from typing import List, Literal, Optional, Union

import numpy as np


class Method(str, Enum):
    """Available methods for probability/odds conversion."""

    BASIC = "basic"
    SHIN = "shin"
    BB = "bb"  # Balanced book
    WPO = "wpo"  # Margin weights proportional to odds
    OR = "or"  # Odds ratio
    POWER = "power"
    ADDITIVE = "additive"
    JSD = "jsd"  # Jensen-Shannon distance


# Type aliases
Odds = Union[List[float], np.ndarray]
Probabilities = Union[List[float], np.ndarray]
MethodType = Union[Method, str]