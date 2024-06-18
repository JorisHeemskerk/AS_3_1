from policy import Policy

class Agent:
    """
    Agent class.
    """

    def __init__(self, policy: Policy)-> None:
        self.policy = policy
