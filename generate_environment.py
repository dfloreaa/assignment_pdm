import gym
from urdfenvs.robots.prius import Prius
import numpy as np

def run_env(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        Prius(mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.array([0.8, 0.2])
    pos0 = np.array([1.0, 0.2, -1.0])
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")

    """"Dimensions for the walls"""
    # dim = [width, length, height]
    dim = np.array([0.2, 8, 0.5])

    """"Coordinates for the walls"""
    # walls = [[x_position, y_position, orientation], ...]
    walls = [
    [0, 12, 0], [4, 12, 0],
    [0, 4, 0], [4, 4, 0]]
    env.add_shapes(
            shape_type="GEOM_BOX", dim=dim, mass=0, poses_2d=walls
        )
    env.add_shapes(
        shape_type="GEOM_BOX", dim=np.array([0.2, 4, 0.5]), mass=0, poses_2d=[[2, 16, 0.5 * np.pi]]
    )

    # env.add_walls(dim = dim, poses_2d = walls)
    print("Starting episode")
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_env(render=True)