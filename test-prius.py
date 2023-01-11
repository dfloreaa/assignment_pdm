import gym
from urdfenvs.robots.prius import Prius
import numpy as np
import car_model as cm
import mpc2 as mpc
from scipy.integrate import odeint
import pybullet as p
from matplotlib import pyplot as plt

DELTA_TIME = 0.2
MAX_SPEED = 1.5  # m/s
MAX_ACC = 1.0  # m/ss
MAX_D_ACC = 1.0  # m/sss
MAX_STEER = np.radians(30)  # rad
MAX_D_STEER = np.radians(30)  # rad/s

L = 0.494

def run_env(n_steps=200, render=False, goal=True, obstacles=True):
    N = 4
    M = 2

    init_state = np.array([0, -0.25, np.radians(-0), 0])
    action = np.array([1, 2])
    speed = 0
    ang_vel = 0

    x_sim = np.zeros((N, n_steps))
    u_sim = np.zeros((M, n_steps - 1))

    print("Starting episode")
    history = []

    path = np.load("assignment_pdm/track.npy")

    # for x_, y_ in zip(path[0, :], path[1, :]):
        # p.addUserDebugLine([x_, y_, 0], [x_, y_, 0.33], [0, 0, 1])

    x_sim = np.zeros((N, n_steps + 1))
    u_sim = np.zeros((M, n_steps))

    # Starting Condition
    x0 = np.zeros(N)
    x0[0] = 0  # x
    x0[1] = -0.25  # y
    x0[2] = 0.0  # v
    x0[3] = np.radians(-0)  # yaw
    x_sim[:, 0] = x0  # simulation_starting conditions

    # starting guess
    action = np.zeros(M)
    action[0] = MAX_ACC / 2  # a
    action[1] = 0.0  # delta
    u_sim[:, 0] = action

    # Cost Matrices
    Q = np.diag([20, 20, 10, 20])  # state error cost
    Qf = np.diag([30, 30, 30, 30])  # state final error cost
    R = np.diag([10, 10])  # input cost
    R_ = np.diag([10, 10])  # input rate of change cost

    controller = mpc.MPC(N, M, Q, R, horizon = 10, dt = DELTA_TIME)
    REF_VEL = 1.0

    for sim_step in range(n_steps):

        # dynamics starting state w.r.t. robot are always null except vel
        start_state = np.array([0, 0, x_sim[2, sim_step], 0])

        # OPTIONAL: Add time delay to starting State- y

        current_action = np.array([u_sim[0, sim_step], u_sim[1, sim_step]])

        # State Matrices
        A, B, C = mpc.get_linear_model_matrices(start_state, current_action, controller, L)

        # Get Reference_traj -> inputs are in worldframe
        target, _ = mpc.get_ref_trajectory(x_sim[:, sim_step], path, REF_VEL, controller)

        x_mpc, u_mpc = controller.optimize_linearized_model(
            A, B, C, start_state, target, time_horizon=10, verbose=False
        )

        # retrieved optimized U and assign to u_bar to linearize in next step
        u_bar = np.vstack(
            (np.array(u_mpc.value[0, :]).flatten(), (np.array(u_mpc.value[1, :]).flatten()))
        )

        u_sim[:, sim_step] = u_bar[:, 0]

        speed += u_bar[:, 0][0] * DELTA_TIME
        ang_vel = u_bar[:, 0][1]

        tspan = [0, DELTA_TIME]
        x_sim[:, sim_step + 1] = controller.predict(x_sim[:, sim_step], u_bar, L)[:, 1]

        # print([speed, ang_vel])
        # print([speed, ang_vel])

    # plot trajectory
    grid = plt.GridSpec(4, 5)

    plt.figure(figsize=(15, 10))

    plt.subplot(grid[0:4, 0:4])
    plt.plot(path[0, :], path[1, :], "b+")
    plt.plot(x_sim[0, :], x_sim[1, :])
    plt.axis("equal")
    plt.ylabel("y")
    plt.xlabel("x")

    plt.subplot(grid[0, 4])
    plt.plot(u_sim[0, :])
    plt.ylabel("a(t) [m/ss]")

    plt.subplot(grid[1, 4])
    plt.plot(x_sim[2, :])
    plt.ylabel("v(t) [m/s]")

    plt.subplot(grid[2, 4])
    plt.plot(np.degrees(u_sim[1, :]))
    plt.ylabel("delta(t) [rad]")

    plt.subplot(grid[3, 4])
    plt.plot(x_sim[3, :])
    plt.ylabel("theta(t) [rad]")

    plt.tight_layout()
    plt.show()

    while True:
        continue
    return history


if __name__ == "__main__":
    run_env(render=True)