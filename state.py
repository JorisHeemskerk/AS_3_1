from dataclasses import dataclass


@dataclass
class State:
    """
    State Data class

    this class makes it such that a state can be saved in a single
     object.
    """
    x: float
    y: float
    linear_velocity_x: float
    linear_velocity_y: float
    angle: float
    angular_velocity: float
    left_leg_touch_ground: float
    right_leg_touch_ground: float
