import numpy as np
import cvxpy as opt

import car_model as cm

class MPC():
    def __init__(self, car_model, N, M, Q, R, horizon = 10 ,dt = 0.2):
        self.car = car_model
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
        # print(target.shape)
        # print(self.state_len, self.horizon + 1)
        x = opt.Variable((self.state_len, self.horizon + 1), name = "states")
        u = opt.Variable((self.action_len, self.horizon), name = "actions")

        # Loop through the entire time_horizon and append costs
        cost_function = []

        for t in range(self.horizon):

            _cost = opt.quad_form(target[:, t + 1] - x[:, t + 1], Q) + opt.quad_form(u[:, t], R)

            _constraints = [
                x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C.flatten(),
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

    def get_closest_waypoint(self, state, path):
        # Computes the index of the waypoint closest to vehicle

        dx = state[0] - path[0, :]
        dy = state[1] - path[1, :]
        dist = np.hypot(dx, dy)
        cw_idx = np.argmin(dist)

        try:
            v = [path[0, cw_idx + 1] - path[0, cw_idx], path[1, cw_idx + 1] - path[1, cw_idx]]
            v /= np.linalg.norm(v)

            d = [path[0, cw_idx] - state[0], path[1, cw_idx] - state[1]]

            if np.dot(d, v) > 0:
                target_idx = cw_idx
            else:
                target_idx = cw_idx + 1

        except IndexError as e:
            target_idx = cw_idx

        return target_idx

    def simplify_angle(self, angle):
        # Simllify an angle to [-pi, pi]
        
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle

    def get_ref_trajectory(self, state, path, target_v):
        """
        For each step in the time horizon
        modified reference in robot frame
        """
        xref = np.zeros((self.state_len, self.horizon + 1))
        dref = np.zeros((1, self.horizon + 1))

        # sp = np.ones((1,T +1))*target_v #speed profile

        ncourse = path.shape[1]

        ind = self.get_closest_waypoint(state, path)
        dx = path[0, ind] - state[0]
        dy = path[1, ind] - state[1]

        xref[0, 0] = dx * np.cos(-state[2]) - dy * np.sin(-state[2])  # X
        xref[1, 0] = dy * np.cos(-state[2]) + dx * np.sin(-state[2])  # Y
        xref[2, 0] = target_v  # V
        xref[3, 0] = self.simplify_angle(path[2, ind] - state[2])  # Theta
        dref[0, 0] = 0.0  # steer operational point should be 0

        dl = 0.05  # Waypoints spacing [m]
        travel = 0.0

        for i in range(1, self.horizon + 1):
            travel += abs(target_v) * self.dt  # current V or target V?
            dind = int(round(travel / dl))

            if (ind + dind) < ncourse:
                dx = path[0, ind + dind] - state[0]
                dy = path[1, ind + dind] - state[1]

                xref[0, i] = dx * np.cos(-state[2]) - dy * np.sin(-state[2])
                xref[1, i] = dy * np.cos(-state[2]) + dx * np.sin(-state[2])
                xref[2, i] = target_v  # sp[ind + dind]
                xref[3, i] = self.simplify_angle(path[2, ind + dind] - state[2])
                dref[0, i] = 0.0
            else:
                dx = path[0, ncourse - 1] - state[0]
                dy = path[1, ncourse - 1] - state[1]

                xref[0, i] = dx * np.cos(-state[2]) - dy * np.sin(-state[2])
                xref[1, i] = dy * np.cos(-state[2]) + dx * np.sin(-state[2])
                xref[2, i] = 0.0  # stop? #sp[ncourse - 1]
                xref[3, i] = self.simplify_angle(path[2, ncourse - 1] - state[2])
                dref[0, i] = 0.0

        return xref, dref