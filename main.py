import os
import gym
from urdfenvs.robots.prius import Prius
import numpy as np
import car_model as cm
import mpc
from scipy.integrate import odeint
import pybullet as p
from matplotlib import pyplot as plt
from generate_path import generatePath
from moving_obstacle import MovingObstacle
import matplotlib.patches as patches
from RRT import Obstacle
from urdfenvs.sensors.obstacle_sensor import ObstacleSensor
from obstacle_avoidance import obstacle_avoid

DELTA_TIME = 0.1
MAX_SPEED = 1.5  # m/s
MAX_ACC = 1.0  # m/ss
MAX_D_ACC = 1.0  # m/sss
MAX_STEER = np.radians(30)  # rad
MAX_D_STEER = np.radians(30)  # rad/s

def run_env(obstacles_coordinates, obstacles_dimensions, environment_id, moving_obstacles, n_steps = 500, render=False, goal=True, obstacles=True):
    robots = [
        Prius(mode="vel"),
    ]

    for moving_obs in moving_obstacles:
        robots.append(MovingObstacle("vel", moving_obs[0], moving_obs[1], moving_obs[2], moving_obs[3], moving_obs[4]))

    L = robots[0]._wheel_distance

    env = gym.make(
        "urdf-env-v0",
        dt = DELTA_TIME, robots=robots, render=render
    )

    N = 4
    M = 2

    speed = 0
    ang_vel = 0

    x_sim = np.zeros((N, n_steps))
    u_sim = np.zeros((M, n_steps - 1))

    current_folder = "assignment_pdm" if os.getcwd()[:-13] != "gym_envs_urdf" else "."
    path_points = np.load(f"{current_folder}/paths/path{environment_id}.npy")

    x_points = []
    y_points = []

    for points in path_points:
        x_points.append(points[0])
        y_points.append(points[1])

    path = mpc.compute_path_from_wp(
    x_points, y_points, 0.1
    )

    for x_, y_ in zip(path[0, :], path[1, :]):
        p.addUserDebugLine([x_, y_, 0], [x_, y_, 0.33], [0, 0, 1])

    x_sim = np.zeros((N, n_steps + 1))
    u_sim = np.zeros((M, n_steps))

    # Starting Condition
    x0 = np.zeros(N)
    x0[0] = x_points[0]  # x
    x0[1] = y_points[0]  # y
    x0[2] = 0.0  # v
    x0[3] = np.radians(-0)  # yaw
    x_sim[:, 0] = x0  # simulation_starting conditions

    n_per_robot = env.n_per_robot()
    ns_per_robot = env.ns_per_robot()
    
    initial_positions = np.array([np.zeros(n) for n in ns_per_robot])
    for i in range(len(initial_positions)):
        if ns_per_robot[i] != n_per_robot[i]:
            initial_positions[i][0:2] = np.array([0.0, i])
        
    initial_positions[0] = np.array(x0[:3])

    for i in range(1, len(robots)):
        initial_positions[i] = np.array([robots[i].x, robots[i].y, robots[i].angle])

    ob = env.reset(pos = initial_positions)  

    print(f"Initial observation : {ob}")

    print("Starting episode")
    history = []

    # starting guess
    action = np.zeros(M)
    action[0] = MAX_ACC / 2  # a
    action[1] = 0.0  # delta
    u_sim[:, 0] = action

    # Cost Matrices
    Q = np.diag([20, 50, 10, 20])  # state error cost
    Qf = np.diag([30, 30, 30, 30])  # state final error cost
    R = np.diag([10, 10])  # input cost
    R_ = np.diag([10, 10])  # input rate of change cost

    controller = mpc.MPC(N, M, Q, R, horizon = 10, dt = DELTA_TIME)
    REF_VEL = 1.0

    prev_steer = 0.0

    """"Dimensions for the walls"""
    # dim = [width, length, height]
    dim = np.array([0.2, 8, 0.5])

    """"Coordinates for the walls"""
    for i in range(len(obstacles_coordinates)):
            env.add_shapes(
                shape_type="GEOM_BOX", dim=obstacles_dimensions[i], mass=0, poses_2d=[obstacles_coordinates[i]]
            )

    # Add sensor
    sensor = ObstacleSensor()
    env.add_sensor(sensor, [0])
    
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
            A, B, C, start_state, target, moving_obstacles = robots[1:], time_horizon=10, verbose=False
        )

        # retrieved optimized U and assign to u_bar to linearize in next step
        u_bar = np.vstack(
            (np.array(u_mpc.value[0, :]).flatten(), (np.array(u_mpc.value[1, :]).flatten()))
        )

        u_sim[:, sim_step] = u_bar[:, 0]

        speed += u_bar[:, 0][0] * DELTA_TIME
        steering_angle = u_bar[:, 0][1]

        steering_angle_delta = (steering_angle - prev_steer)/DELTA_TIME
        prev_steer = steering_angle

        # print([speed, ang_vel])

        mu = 0.7 # Typical friction coefficient for a patterned tire in dry conditions
        g = 9.81 # Standard gravity

        # Breaking distance
        d_crit = 1.5 * speed**2 / (2*mu*g)
        d_safe = 2 * speed**2 / (2*mu*g)
        pos = robots[0].state["joint_state"]["position"]

        print(d_crit)

        x_size = abs(1.7526/2 * np.sin(pos[2])) + abs(2.15/2 * np.cos(pos[2]))
        y_size = abs(1.7526/2 * np.cos(pos[2])) + abs(2.15/2 * np.sin(pos[2]))

        for robot in robots[1:]:

            robot_x = robot.state["joint_state"]["position"][0]
            robot_y = robot.state["joint_state"]["position"][1]

            dist1 = np.sqrt((pos[0] + x_size - (robot_x))**2 + (pos[1] - y_size - robot_y)**2) - robot.width
            dist3 = np.sqrt((pos[0] + x_size - (robot_x))**2 + (pos[1] + y_size - robot_y)**2) - robot.width
            dist2 = np.sqrt((pos[0] - x_size - (robot_x))**2 + (pos[1] + y_size - robot_y)**2) - robot.width
            dist4 = np.sqrt((pos[0] - x_size - (robot_x))**2 + (pos[1] - y_size - robot_y)**2) - robot.width

            # print(dist1, dist2, dist3, dist4)

            if dist1 < d_crit or dist2 < d_crit or dist3 < d_crit or dist4 < d_crit:
                print('Warning, impending collision')
                speed = 0
                steering_angle_delta = 0

        second_speed = 1.1 if int(sim_step/25) % 2 else -1.1

        ob, _, _, _ = env.step([speed, steering_angle_delta, second_speed, 0])
        pos = robots[0].state["joint_state"]["position"]

        x_sim[0, sim_step + 1] = pos[0]
        x_sim[1, sim_step + 1] = pos[1]
        x_sim[2, sim_step + 1] = speed
        x_sim[3, sim_step + 1] = pos[2]
        history.append(ob)
    env.close()

    history2 = []
    for _ in range(n_steps):
        ob, reward, done, info = env.step(action)
        # In observations, information about obstacles is stored in ob['obstacleSensor']
        history2.append(ob)
    env.close()

    # Obstacle avoidance
    obstacles = False
    obstacle_avoid(n_steps, history2, obstacles)

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
    
    # save the plot
    if not os.path.exists("./performance"):
        os.makedirs("./performance")

    plt.savefig('./performance/performance{}.png'.format(environment_id))

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
                                             [0.5, 30, 1], [30, 0.5, 1] ]},

}

if __name__ == "__main__":
    MAKE_ANIMATION = False
    environment_id = 3
    
    environment_dict = environments[environment_id]

    current_folder = "assignment_pdm" if os.getcwd()[:-13] != "gym_envs_urdf" else "."

    if not os.path.exists(f"{current_folder}/paths/path{environment_id}.npy") or MAKE_ANIMATION:
        generatePath(environment_dict, environment_id, n_iter = 20000, make_animation = MAKE_ANIMATION)

    obstacle_coordinates = environments[environment_id]["obstacle_coordinates"] + environments[environment_id]["boundary_coordinates"]
    obstacle_dimensions = environments[environment_id]["obstacle_dimensions"] + environments[environment_id]["boundary_dimensions"]
    
    run_env(obstacle_coordinates, obstacle_dimensions, environment_id, [[-13, -8, np.pi, 0.5, 0.5]], render=True)

    #TODO: Add obstacles to performance plot