from enum import Enum


class Action(Enum):
    """
    Action Enum class.

    This class makes it such that every action taken can be described
     using text instead of the numbers.
    """
    NOTHING = 0
    LEFT    = 1
    MAIN    = 2
    RIGHT   = 3
