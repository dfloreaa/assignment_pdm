import gym
from urdfenvs.robots.jackal import JackalRobot
import numpy as np
import car_model as cm
import mpc

DELTA_TIME = 0.01

def run_env(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        JackalRobot(mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt = DELTA_TIME, robots=robots, render=render
    )

    N = 4
    M = 2

    init_state = np.array([0, 0, -1.0])
    action = np.array([0, 2])

    x_sim = np.zeros((N, n_steps))
    u_sim = np.zeros((M, n_steps - 1))

    ob = env.reset(pos = init_state)
    print(f"Initial observation : {ob}")

    print("Starting episode")
    history = []

    for sim_step in range(n_steps):

        start_state = np.array([0, 0, 0, x_sim[3, sim_step]])

        current_action = np.array([u_sim[0, sim_step], u_sim[1, sim_step]])

        # State Matrices
        # A, B, C = cm.get_linear_model_matrices(start_state, current_action)

        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_env(render=True)