"""Measuring and selecting diverse sets of plans, over the unified-planning
framework."""

from unified_planning.shortcuts import get_environment

# The framework prints a credits banner on first use of the shortcuts module,
# which turns any script that scores plan sets into noise. Silenced once, here,
# rather than in each subpackage's __init__.
get_environment().credits_stream = None

VERSION: tuple[int | str, ...] = (1, 0, 0)
__version__ = ".".join(str(part) for part in VERSION)
