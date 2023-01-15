import numpy as np
import cvxpy as opt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

MAX_SPEED = 1.5  # m/s
MAX_ACC = 1.0  # m/ss
MAX_D_ACC = 1.0  # m/sss
MAX_STEER = np.radians(40)  # rad
MAX_D_STEER = np.radians(30)  # rad/s

class MPC():
    def __init__(self, N, M, Q, R, horizon = 10 ,dt = 0.2):
        self.N = N
        self.M = M

        self.Q = Q
        self.R = R

        self.horizon = horizon      # Time Horizon for the prediction
        self.dt = dt                # Discretization Step

    def optimize_linearized_model(self, A, B, C, state, target, moving_obstacles, time_horizon=10, verbose=False):

        # Create variables
        x = opt.Variable((self.N, time_horizon + 1), name="states")
        u = opt.Variable((self.M, time_horizon), name="actions")

        # Loop through the entire time_horizon and append costs
        cost_function = []

        for t in range(time_horizon):
            cost = opt.quad_form(target[:, t + 1] - x[:, t + 1], self.Q) + opt.quad_form(u[:, t], self.R)

            constraints = [
                x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C,
                u[0, t] >= - MAX_ACC,
                u[0, t] <= MAX_ACC,
                u[1, t] >= - MAX_STEER,
                u[1, t] <= MAX_STEER,
            ]

            # # Collision avoidance against objectives
            # for mov_obs in moving_obstacles:

            #     constraints += [opt.abs(x[0, t + 1] - mov_obs.x) <= 1]

            # Actuation rate of change
            if t < (time_horizon - 1):
                cost += opt.quad_form(u[:, t + 1] - u[:, t], self.R * 1)
                constraints += [opt.abs(u[0, t + 1] - u[0, t]) / self.dt <= MAX_D_ACC]
                constraints += [opt.abs(u[1, t + 1] - u[1, t]) / self.dt <= MAX_D_STEER]

            if t == 0:
                # _constraints += [opt.norm(target[:, time_horizon] - x[:, time_horizon], 1) <= 0.01,
                #                x[:, 0] == initial_state]
                constraints += [x[:, 0] == state]

            cost_function.append(
                opt.Problem(opt.Minimize(cost), constraints = constraints)
            )

        # Add final cost
        problem = sum(cost_function)

        # Minimize Problem
        problem.solve(verbose = verbose, solver = opt.OSQP)
        return x, u

    def predict(self, x0, u, L):
        """ """

        x_ = np.zeros((self.N, self.horizon + 1))

        x_[:, 0] = x0

        # solve ODE
        for t in range(1, self.horizon + 1):

            tspan = [0, self.dt]
            x_next = odeint(kinematics_model, x0, tspan, args=(u[:, t - 1], L))

            x0 = x_next[1]
            x_[:, t] = x_next[1]

            return x_
    
    #########################################################

def kinematics_model(x, t, u, L):
    """
    Returns the set of ODE of the vehicle model.
    """

    L = L  # vehicle wheelbase
    dxdt = x[2] * np.cos(x[3])
    dydt = x[2] * np.sin(x[3])
    dvdt = u[0]
    dthetadt = x[2] * np.tan(u[1]) / L

    dqdt = [dxdt, dydt, dvdt, dthetadt]

    return dqdt


def get_nn_idx(state, path):
    """
    Computes the index of the waypoint closest to vehicle
    """

    dx = state[0] - path[0, :]
    dy = state[1] - path[1, :]
    dist = np.hypot(dx, dy)
    nn_idx = np.argmin(dist)

    try:
        v = [
            path[0, nn_idx + 1] - path[0, nn_idx],
            path[1, nn_idx + 1] - path[1, nn_idx],
        ]
        v /= np.linalg.norm(v)

        d = [path[0, nn_idx] - state[0], path[1, nn_idx] - state[1]]

        if np.dot(d, v) > 0:
            target_idx = nn_idx
        else:
            target_idx = nn_idx + 1

    except IndexError as e:
        target_idx = nn_idx

    distance = np.sqrt(np.square(dx) + np.square(dy))[target_idx]

    return target_idx

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

def get_ref_trajectory(state, path, target_v, mpc):
    """
    For each step in the time horizon
    modified reference in robot frame
    """
    xref = np.zeros((mpc.N, mpc.horizon + 1))
    dref = np.zeros((1, mpc.horizon + 1))

    # sp = np.ones((1,T +1))*target_v #speed profile

    ncourse = path.shape[1]

    ind = get_nn_idx(state, path)
    dx = path[0, ind] - state[0]
    dy = path[1, ind] - state[1]

    distance = np.sqrt(dx**2 + dy**2)

    xref[0, 0] = dx * np.cos(-state[3]) - dy * np.sin(-state[3])  # X
    xref[1, 0] = dy * np.cos(-state[3]) + dx * np.sin(-state[3])  # Y
    xref[2, 0] = target_v  # V
    xref[3, 0] = normalize_angle(path[2, ind] - state[3])  # Theta
    dref[0, 0] = 0.0  # steer operational point should be 0

    dl = 0.05  # Waypoints spacing [m]
    travel = 0.0

    for i in range(1, mpc.horizon + 1):
        travel += abs(target_v) * mpc.dt  # current V or target V?
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            dx = path[0, ind + dind] - state[0]
            dy = path[1, ind + dind] - state[1]

            xref[0, i] = dx * np.cos(-state[3]) - dy * np.sin(-state[3])
            xref[1, i] = dy * np.cos(-state[3]) + dx * np.sin(-state[3])
            xref[2, i] = target_v  # sp[ind + dind]
            xref[3, i] = normalize_angle(path[2, ind + dind] - state[3])
            dref[0, i] = 0.0
        else:
            dx = path[0, ncourse - 1] - state[0]
            dy = path[1, ncourse - 1] - state[1]

            xref[0, i] = dx * np.cos(-state[3]) - dy * np.sin(-state[3])
            xref[1, i] = dy * np.cos(-state[3]) + dx * np.sin(-state[3])
            xref[2, i] = 0.0  # stop? #sp[ncourse - 1]
            xref[3, i] = normalize_angle(path[2, ncourse - 1] - state[3])
            dref[0, i] = 0.0

    return xref, dref, distance

def get_linear_model_matrices(x_bar, u_bar, mpc, L):
    """
    Computes the LTI approximated state space model x' = Ax + Bu + C
    """

    x = x_bar[0]
    y = x_bar[1]
    v = x_bar[2]
    theta = x_bar[3]

    a = u_bar[0]
    delta = u_bar[1]

    ct = np.cos(theta)
    st = np.sin(theta)
    cd = np.cos(delta)
    td = np.tan(delta)

    A = np.zeros((mpc.N, mpc.N))
    A[0, 2] = ct
    A[0, 3] = -v * st
    A[1, 2] = st
    A[1, 3] = v * ct
    A[3, 2] = v * td / L
    A_lin = np.eye(mpc.N) + mpc.dt * A

    B = np.zeros((mpc.N, mpc.M))
    B[2, 0] = 1
    B[3, 1] = v / (L * cd**2)
    B_lin = mpc.dt * B

    f_xu = np.array([v * ct, v * st, a, v * td / L]).reshape(mpc.N, 1)
    C_lin = mpc.dt * (
        f_xu - np.dot(A, x_bar.reshape(mpc.N, 1)) - np.dot(B, u_bar.reshape(mpc.M, 1))
    )  # .flatten()

    # return np.round(A_lin,6), np.round(B_lin,6), np.round(C_lin,6)
    return A_lin, B_lin, C_lin

def compute_path_from_wp(start_xp, start_yp, step=0.1):
    """
    Computes a reference path given a set of waypoints
    """

    final_xp = []
    final_yp = []
    delta = step  # [m]

    for idx in range(len(start_xp) - 1):
        section_len = np.sum(
            np.sqrt(
                np.power(np.diff(start_xp[idx : idx + 2]), 2)
                + np.power(np.diff(start_yp[idx : idx + 2]), 2)
            )
        )

        interp_range = np.linspace(0, 1, np.floor(section_len / delta).astype(int))

        fx = interp1d(np.linspace(0, 1, 2), start_xp[idx : idx + 2], kind=1)
        fy = interp1d(np.linspace(0, 1, 2), start_yp[idx : idx + 2], kind=1)

        # watch out to duplicate points!
        final_xp = np.append(final_xp, fx(interp_range)[1:])
        final_yp = np.append(final_yp, fy(interp_range)[1:])

    dx = np.append(0, np.diff(final_xp))
    dy = np.append(0, np.diff(final_yp))
    theta = np.arctan2(dy, dx)

    return np.vstack((final_xp, final_yp, theta))