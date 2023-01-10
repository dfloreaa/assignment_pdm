import numpy as np
from scipy.integrate import odeint

class CarModel():
    def __init__(self, initial_state, initial_u, wheel_dist, horizon = 10):
        self.state = initial_state  # np.array([x, y, theta, vel])
        self.u = initial_u          # np.array([vel, theta_dot])

        self.L = wheel_dist     # m
        self.horizon = horizon

        self.MAX_ACC = 1.0                  # m/s^2
        self.MAX_STEER = np.radians(30)     # rad

        self.MAX_ACC_DT = 1.0               # m/s^3
        self.MAX_STEER_DT = np.radians(30)  # rad/s

    def kinematic_model(self, state, t, u):
        """
        Returns the set of ODE of the bicycle model.
        """

        dvdt = u[0]
        x_dt = state[2] * np.cos(dvdt)
        y_dt = state[2] * np.sin(dvdt)
        theta_dt = state[2] * np.tan(u[1]) / self.L

        state_dt = [x_dt, y_dt, theta_dt, dvdt]

        return state_dt

    def get_linear_model_matrices(self, state, action, dt):
        """
        Computes the LTI approximated state space model x' = Ax + Bu + C
        """

        N = len(state)
        M = len(action)

        x = state[0]
        y = state[1]
        theta = state[2]
        v = state[3]

        a = action[0]
        theta_dot = action[1]

        A = np.eye(N)
        A[0, 2] = -v * np.sin(theta) * dt
        A[0, 3] = np.cos(theta) * dt
        A[1, 2] = v * np.cos(theta) * dt
        A[1, 3] = np.sin(theta) * dt
        A[3, 3] = (v * np.tan(theta_dot) / self.L) * dt

        B = np.zeros((N, M))
        B[2, 1] = v / (self.L * np.cos(theta_dot)**2)
        B[3, 0] = 1
        B *= dt

        f_xu = np.array([v * np.cos(theta), v * np.sin(theta), v * np.tan(theta_dot) / self.L, a]).reshape(N, 1)
        C = dt * (f_xu - np.dot(A, state.reshape(N, 1)) - np.dot(B, action.reshape(M, 1)))  # .flatten()

        return A, B, C
        
    def predict_model(self, init_state, u):
        x_ = np.zeros((len(init_state), self.horizon + 1))
        x_[:, 0] = init_state

        # Solve ODE
        for t in range(1, self.horizon + 1):

            time_range = [0, self.dt]
            x_next = odeint(self.kinematic_model, init_state, time_range, args = (u[:, t - 1],))

            init_state = x_next[1]
            x_[:, t] = x_next[1]

        return x_