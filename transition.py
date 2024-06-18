from action import Action
from state import State

from dataclasses import dataclass


@dataclass
class Transition:
    state: State
    action: Action
    reward: float
    next_state: State
    terminated: bool
