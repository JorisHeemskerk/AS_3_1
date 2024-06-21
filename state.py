from dataclasses import dataclass


@dataclass
class State:
    x: float
    y: float
    linear_velocity_x: float
    linear_velocity_y: float
    angle: float
    angular_velocity: float
    left_leg_touch_ground: float
    right_leg_touch_ground: float
