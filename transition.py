from action import Action
from state import State

import dataclasses


@dataclasses.dataclass
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

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)

    # def serialize(self):
    #     return {"state" : [str(x) for x in self.state],
    #             "next_state": [str(x) for x in self.next_state],
    #             "action": str(self.action),
    #             "reward": str(self.reward),
    #             "terminal": str(self.terminal)}