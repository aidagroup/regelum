"""Classic control environments with discrete state and action spaces.

Includes:
- GridWorld: 2D grid navigation with obstacles
- Chain: 1D chain with left/right actions
- DiscretePendulum: Discretized pendulum dynamics
"""

from .grid_world import GridWorld
from .chain import Chain
from .discrete_pendulum import DiscretePendulum

__all__ = [
    "GridWorld",
    "Chain",
    "DiscretePendulum",
]
