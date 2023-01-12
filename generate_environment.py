import os
import gym
from urdfenvs.robots.prius import Prius
import numpy as np
from RRT import pathComputation

def run_env(obstacles_coordinates, obstacles_dimensions,
            n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        Prius(mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.array([0.8, 0.2]) #Velocity and rate of change of steering angle
    pos0 = np.array([1.0, 2.0, -1.0])
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")

    """"Dimensions for the walls"""
    # dim = [width, length, height]
    dim = np.array([0.2, 8, 0.5])

    """"Coordinates for the walls"""
    for i in range(len(obstacles_coordinates)):
            env.add_shapes(
                shape_type="GEOM_BOX", dim=obstacles_dimensions[i], mass=0, poses_2d=[obstacles_coordinates[i]]
            )

    # env.add_walls(dim = dim, poses_2d = walls)
    print("Starting episode")
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history

environments = {0: {"obstacle_coordinates": [[-5, -5, 0], [5, 5, 0]],
                    "obstacle_dimensions": [[20, 1, 1], [20, 1, 1]],
                    "startpos": (-13, -13),
                    "endpos": (13, 13),
                    "boundary_coordinates": [[0, -15, 0], [-15, 0, 0],
                                            [15, 0, 0], [0, 15, 0]],
                    "boundary_dimensions": [[30, 0.5, 1], [0.5, 30, 1], 
                                             [0.5, 30, 1], [30, 0.5, 1] ]},

                1: {"obstacle_coordinates": [[-7.5, -11, 0], [7.5, 0, 0], [6 , 10, 0], [0, -2., 0], [-7.5, -2.5, 0]],
                    "obstacle_dimensions": [[15, 0.5, 1], [15, 0.5, 1], [18, 0.5, 1], [0.5, 4, 1], [5, 2.5, 1]],
                    "startpos": (-13, -13),
                    "endpos": (13, 13), 
                    "boundary_coordinates": [[0, -15, 0], [-15, 0, 0],
                                            [15, 0, 0], [0, 15, 0]],
                    "boundary_dimensions": [[30, 0.5, 1], [0.5, 30, 1], 
                                             [0.5, 30, 1], [30, 0.5, 1] ]},

                2: {"obstacle_coordinates":[[-11.25, 11.25, 0], [-3.75, 11.25, 0], [3.75, 11.25, 0], [11.25, 11.25, 0],
                                            [-11.25, 3.75, 0], [-3.75, 3.75, 0], [3.75, 3.75, 0], [11.25, 3.75, 0],
                                            [-11.25, -3.75, 0], [-3.75, -3.75, 0], [3.75, -3.75, 0], [11.25, -3.75, 0],
                                            [-11.25, -11.25, 0], [-3.75, -11.25, 0], [3.75, -11.25, 0], [11.25, -11.25, 0]],
                    "obstacle_dimensions": [[1.5, 1.5, 1], [1.5, 1.5, 1], [1.5, 1.5, 1], [1.5, 1.5, 1],
                                            [1.5, 1.5, 1], [1.5, 1.5, 1], [1.5, 1.5, 1], [1.5, 1.5, 1],
                                            [1.5, 1.5, 1], [1.5, 1.5, 1], [1.5, 1.5, 1], [1.5, 1.5, 1],
                                            [1.5, 1.5, 1], [1.5, 1.5, 1], [1.5, 1.5, 1], [1.5, 1.5, 1]],
                    "startpos": (-11.25, -13.5),
                    "endpos": (11.25, 13.5), 
                    "boundary_coordinates": [[0, -15, 0], [-15, 0, 0],
                                            [15, 0, 0], [0, 15, 0]],
                    "boundary_dimensions": [[30, 0.5, 1], [0.5, 30, 1], 
                                             [0.5, 30, 1], [30, 0.5, 1] ]},

                3:  {"obstacle_coordinates":[[-12, -11, 0], [-7, -7, 0], [3, -11, 0], [12, -13, 0], [-11, -3, 0], [10, -5, 0],
                                            [1, -1, 0], [-7, 3, 0], [6, 7, 0], [-6, 9, 0],
                                            [-5, -11, 0], [1, -13, 0], [9, -11, 0], [-11, 0, 0], [-1, -3, 0], [3, 3, 0], [-7, 12, 0],
                                            [9, 1, 0], [6, 14, 0]],
                    "obstacle_dimensions": [[6, .5, 1], [4, .5, 1], [4, .5, 1], [6, .5, 1], [8, .5, 1], [10, .5, 1],
                                            [4, .5, 1], [8, .5, 1], [6, .5, 1], [10, .5, 1],
                                            [.5, 8, 1], [.5, 4, 1], [.5, 4, 1], [.5, 6, 1], [.5, 4, 1], [.5, 8, 1], [.5, 6, 1],
                                            [4, 4, 1], [6, 2, 1]],
                    "startpos": (-13, -13),
                    "endpos": (13, 13),
                    "boundary_coordinates": [[0, -15, 0], [-15, 0, 0],
                                            [15, 0, 0], [0, 15, 0]],
                    "boundary_dimensions": [[30, 0.5, 1], [0.5, 30, 1], 
                                             [0.5, 30, 1], [30, 0.5, 1] ]}
        }

# Environment consists of obstacles coordinates and dimensions

if __name__ == "__main__":
    MAKE_ANIMATION = True
    environment_id = 3
    boundary_coordinates = [[0, -15, 0], [-15, 0, 0],
                    [15, 0, 0], [0, 15, 0]]    
    boundary_dimensions = [[30, 0.5, 1], [0.5, 30, 1], 
                            [0.5, 30, 1], [30, 0.5, 1] ]


    obstacle_coordinates = environments[environment_id]["obstacle_coordinates"] + environments[environment_id]["boundary_coordinates"]
    obstacle_dimensions = environments[environment_id]["obstacle_dimensions"] + environments[environment_id]["boundary_dimensions"]

    startpos = environments[environment_id]["startpos"]
    endpos = environments[environment_id]["endpos"]
    n_iter = 20000
    

    path = pathComputation(obstacle_coordinates, obstacle_dimensions, environment_id, 
                            startpos, endpos, n_iter, make_animation= MAKE_ANIMATION)

    # save the plot
    if not os.path.exists("./paths"):
        os.makedirs("./paths")
    
    np.save("./paths/path{}.npy".format(environment_id), np.array(path))
    print(path)

    run_env(obstacle_coordinates, obstacle_dimensions, render=True)