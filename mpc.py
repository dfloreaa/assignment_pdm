import numpy as np
import cvxpy as opt

import car_model as cm

class MPC():
    def __init__(self, car_model, N, M, Q, R, horizon = 10 ,dt = 0.2):
        self.car = cm.CarModel()
        self.state_len = N
        self.action_len = M

        self.state_cost = Q
        self.action_cost = R

        self.horizon = horizon      # Time Horizon for the prediction
        self.dt = dt                # Discretization Step
    
    def optimize_linearized_model(self, A, B, C, initial_state, target):
        assert len(initial_state) == self.state_len

        max_acc = self.car.MAX_ACC
        max_steer = self.car.MAX_STEER

        max_acc_dt = self.car.MAX_ACC_DT
        max_steer_dt = self.car.MAX_STEER_DT

        Q = self.state_cost
        R = self.action_cost

        # Create variables
        x = opt.Variable((self.state_len, self.horizon + 1), name = "states")
        u = opt.Variable((self.action_len, self.horizon), name = "actions")

        # Loop through the entire time_horizon and append costs
        cost_function = []

        for t in range(self.horizon):

            _cost = opt.quad_form(target[:, t + 1] - x[:, t + 1], Q) + opt.quad_form(u[:, t], R)

            _constraints = [
                x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C,
                u[0, t] >= -max_acc,
                u[0, t] <= max_acc,
                u[1, t] >= -max_steer,
                u[1, t] <= max_steer,
            ]

            # Actuation rate of change
            if t < (self.horizon - 1):
                _cost += opt.quad_form(u[:, t + 1] - u[:, t], R * 1)
                _constraints += [opt.abs(u[0, t + 1] - u[0, t]) / self.dt <= max_acc_dt]
                _constraints += [opt.abs(u[1, t + 1] - u[1, t]) / self.dt <= max_steer_dt]

            if t == 0:
                _constraints += [x[:, 0] == initial_state]

            cost_function.append(
                opt.Problem(opt.Minimize(_cost), constraints = _constraints)
            )

        # Add final cost
        problem = sum(cost_function)

        # Minimize Problem
        problem.solve(solver=opt.OSQP)
        return x, u

    def get_closest_waypoint(self, path):
        # Computes the index of the waypoint closest to vehicle

        dx = self.state[0] - path[0, :]
        dy = self.state[1] - path[1, :]
        dist = np.hypot(dx, dy)
        cw_idx = np.argmin(dist)

        try:
            v = [path[0, cw_idx + 1] - path[0, cw_idx], path[1, cw_idx + 1] - path[1, cw_idx]]
            v /= np.linalg.norm(v)

            d = [path[0, cw_idx] - self.state[0], path[1, cw_idx] - self.state[1]]

            if np.dot(d, v) > 0:
                target_idx = cw_idx
            else:
                target_idx = cw_idx + 1

        except IndexError as e:
            target_idx = cw_idx

        return target_idx

    def simplify_angle(angle):
        # Simllify an angle to [-pi, pi]
        
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle