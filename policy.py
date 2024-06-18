from action import Action
from state import State

from torch import nn


class Policy:
    """
    Policy
    
    Base policy class with random behavior.
    This policy works as follows:
    - select select a random action and return it.
    """

    def __init__(self, network: nn.Module)-> None:
        """
        Initializer for Policy.
        """
        self._network = network


    def select_action(self, state: State)-> Action:
        """
        Select action based on current policy.
        
        This policy works as follows:
        - select select a random action and return it.

        @param state: Current State to perform action in.

        @return Action with Action to perform.
        """
        return Action.UP
    
    def decay()-> None:
        """
        TODO: degrade epsilon (optional)
        """
        pass
