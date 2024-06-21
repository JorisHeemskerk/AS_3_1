from action import Action
from state import State

from dataclasses import dataclass


@dataclass
class Transition:
    """
    Transition Data class

    this class makes it such that a transition can be saved in a single
     object.
    """
    state: State
    action: Action
    reward: float
    next_state: State
    terminated: bool
