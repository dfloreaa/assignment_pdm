from urdfenvs.robots.jackal import JackalRobot
import numpy as np

class MovingObstacle(JackalRobot):
    def __init__(self, mode: str, x_pos: float, y_pos: float, angle: float, vel: float, duration: int):
        super().__init__(mode)
        self.x = x_pos
        self.y = y_pos
        self.angle = angle
        self.vel = vel
        self.ang_vel = 0
        self.width = 0.5
        self.duration = duration

        self._limit_pos_j[0, 0:3] = np.array([-100.0, -100.0, -2 * np.pi])
        self._limit_pos_j[1, 0:3] = np.array([100.0, 100.0, 2 * np.pi])