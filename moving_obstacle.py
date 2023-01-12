from urdfenvs.robots.jackal import JackalRobot
import numpy as np

class MovingObstacle(JackalRobot):
    def __init__(self, mode: str, x_pos: float, y_pos: float, angle: float, width: float, height: float):
        super().__init__(mode)
        self.x = x_pos
        self.y = y_pos
        self.angle = angle
        self.vel = 0
        self.ang_vel = 0
        self.width = width
        self.height = height

        self._limit_pos_j[0, 0:3] = np.array([-100.0, -100.0, -2 * np.pi])
        self._limit_pos_j[1, 0:3] = np.array([100.0, 100.0, 2 * np.pi])