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

    def __iter__(self):
        """Make the instance iterable, allowing it to be used as a tuple."""
        return iter((self.x, self.y, self.linear_velocity_x, self.linear_velocity_y,
                     self.angle, self.angular_velocity, self.left_leg_touch_ground, self.right_leg_touch_ground))
