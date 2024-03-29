import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from RRT import Obstacle

def plot(path, environments, environment_id, obstacles_coordinates, obstacles_dimensions, x_sim, u_sim):
    # plot trajectory
    grid = plt.GridSpec(4, 5)

    plt.figure(figsize=(15, 10))

    plt.subplot(grid[0:4, 0:4])
    ax = plt.gca()
    circle = patches.Circle(environments[environment_id]['startpos'], radius=0.25, color='black')
    ax.add_artist(circle)
    circle = patches.Circle(environments[environment_id]['endpos'], radius=0.25, color='black')
    ax.add_artist(circle)
    for i in range(len(obstacles_coordinates)):
        obstacle = Obstacle(obstacles_coordinates[i][0], obstacles_coordinates[i][1],
                            obstacles_dimensions[i][0], obstacles_dimensions[i][1])
        rect = patches.Rectangle((obstacle.x-obstacle.width/2, obstacle.y - obstacle.height/2), obstacle.width, obstacle.height, color='black')
        ax.add_artist(rect)
    plt.plot(path[0, :], path[1, :], "b+")
    plt.plot(x_sim[0, :], x_sim[1, :], color = "red")
    plt.axis("equal")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.axis([-15, 15, -15, 15])

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

    file_folder = os.path.dirname(os.path.realpath(__file__))
    path_directory = f"{file_folder}/performance/performance{environment_id}.png"

    # save the plot
    if not os.path.exists(f"{file_folder}/performance"):
        os.makedirs(f"{file_folder}/performance")

    plt.savefig(path_directory)
    return

def plot_distance(n_robots, dist_obstacles):
    plt.figure(figsize=(10, 10))
    y_zero = np.zeros((dist_obstacles.shape[0], 1))
    plt.plot(y_zero, label = "y = 0")
    for i in range(1, n_robots):
        plt.plot(dist_obstacles[:, i - 1], label = f"MO #{i}")
    plt.legend(loc='best')
    plt.show()

def plot_deviation(deviation_history):
    plt.figure(figsize=(10, 10))
    y_zero = np.zeros((deviation_history.shape[0], 1))
    plt.plot(y_zero, label = "y = 0")
    plt.plot(deviation_history, label = f"Deviation from the RRT path")
    plt.legend(loc='best')
    plt.show()