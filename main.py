import os
import gym
from urdfenvs.robots.prius import Prius
import numpy as np
import car_model as cm
import mpc
from scipy.integrate import odeint
import pybullet as p
from generate_path import generatePath
from moving_obstacle import MovingObstacle
from urdfenvs.sensors.obstacle_sensor import ObstacleSensor
from obstacle_avoidance import obstacle_avoid
import plot_trajectory
from utils import environments, get_dist_point_rect

DELTA_TIME = 0.1
MAX_SPEED = 1.5  # m/s
MAX_ACC = 1.0  # m/ss
MAX_D_ACC = 1.0  # m/sss
MAX_STEER = np.radians(30)  # rad
MAX_D_STEER = np.radians(30)  # rad/s

def run_env(obstacles_coordinates, obstacles_dimensions, environment_id, moving_obstacles, n_steps = 500, render=False, goal=True, obstacles=True):

    # Generate each robot
    robots = [Prius(mode="vel")]
    for moving_obs in moving_obstacles:
        robots.append(MovingObstacle("vel", moving_obs[0], moving_obs[1], moving_obs[2], moving_obs[3], moving_obs[4]))

    env = gym.make("urdf-env-v0", dt = DELTA_TIME, robots = robots, render = render)

    """----- PATH PLANNING -----"""
    # Search from the current file's path and look for a path file from there
    file_folder = os.path.dirname(os.path.realpath(__file__))
    path_directory = f"{file_folder}/paths/path{environment_id}.npy"

    # If no path is found, search for it
    if not os.path.exists(path_directory) or MAKE_ANIMATION:
        path_points = generatePath(environment_dict, environment_id, n_iter = 20000, make_animation = MAKE_ANIMATION, directory = path_directory)
    else:
        path_points = np.load(path_directory)

    # Add points to lists in order to compute a path
    x_points = [points[0] for points in path_points]
    y_points = [points[1] for points in path_points]

    # Create intermediate points
    path = mpc.compute_path_from_wp(x_points, y_points, step = 0.1)

    # Show the path inside of the simulation
    for x_, y_ in zip(path[0, :], path[1, :]):
        p.addUserDebugLine([x_, y_, 0], [x_, y_, 0.33], [0, 0, 1])


    """"----- INITIALIZE VARIABLES FOR THE SIMULATION ------"""
    N = 4   # Number of variables for each configuration of the Prius
    M = 2   # Number of variables for each action of the Prius
    REF_VEL = 1.0   # Reference velocity for the trajectory

    mu = 0.7 # Typical friction coefficient for a patterned tire in dry conditions
    g = 9.81 # Standard gravity
    
    # Distance between the wheels of the Prius, useful for later calculations
    L = robots[0]._wheel_distance
    vehicle_width = 1.7526
    vehicle_length = 2.15

    # Useful variables for the actuation of the vehicle
    speed = 0
    steering_angle_delta = 0.0
    steering_angle = 0.0
    prev_steer = 0.0

    # Starting Condition
    x_sim = np.zeros((N, n_steps + 1))
    u_sim = np.zeros((M, n_steps))

    x0 = np.zeros(N)
    x0[0] = x_points[0]     # x
    x0[1] = y_points[0]     # y
    x0[2] = 0.0             # velocity
    x0[3] = np.radians(-0)  # yaw

    x_sim[:, 0] = x0        # Simulation starting conditions

    # Starting guess for input
    action = np.zeros(M)
    action[0] = MAX_ACC / 2  # a
    action[1] = 0.0  # delta
    u_sim[:, 0] = action

    # Cost Matrices
    Q = np.diag([50, 50, 10, 20])   # state error cost
    R = np.diag([10, 5])           # input cost

    controller = mpc.MPC(N, M, Q, R, horizon = 10, dt = DELTA_TIME)


    """----- INITIALIZE ROBOTS -----"""
    n_per_robot = env.n_per_robot()     # Dimensions for workspace
    ns_per_robot = env.ns_per_robot()   # Dimensions for configurations
    
    initial_positions = np.array([np.zeros(n) for n in ns_per_robot])
    for i in range(len(initial_positions)):
        if ns_per_robot[i] != n_per_robot[i]:
            initial_positions[i][0:2] = np.array([0.0, i])
        
    initial_positions[0] = np.array(x0[:3])

    for i in range(1, len(robots)):
        initial_positions[i] = np.array([robots[i].x, robots[i].y, robots[i].angle])

    ob = env.reset(pos = initial_positions)  


    """----- ADD WALLS TO THE SIMULATION ------"""
    # Dimensions for the walls (dim = [width, length, height])
    dim = np.array([0.2, 8, 0.5])

    # Coordinates for the walls
    for i in range(len(obstacles_coordinates)):
        env.add_shapes(shape_type="GEOM_BOX", dim = obstacles_dimensions[i], mass=0, poses_2d=[obstacles_coordinates[i]])

    # Add sensor
    sensor = ObstacleSensor()
    env.add_sensor(sensor, [0])


    """"----- SIMULATE THE ENVIRONMENT -----"""
    print("Starting the simulation")
    for sim_step in range(n_steps):

        # Dynamics state from the robot's frame, all zero except for velocity
        start_state = np.array([0, 0, x_sim[2, sim_step], 0])

        # OPTIONAL: Add time delay to starting State- y
        current_action = np.array([u_sim[0, sim_step], u_sim[1, sim_step]])

        # State Matrices
        A, B, C = mpc.get_linear_model_matrices(start_state, current_action, controller, L)

        # Get Reference_traj -> inputs are in worldframe
        target, _ = mpc.get_ref_trajectory(x_sim[:, sim_step], path, REF_VEL, controller)

        x_mpc, u_mpc = controller.optimize_linearized_model(A, B, C, start_state, target, moving_obstacles = robots[1:], time_horizon=10, verbose=False)

        # Retrieve optimized U and assign to u_bar to linearize in next step
        u_bar = np.vstack((np.array(u_mpc.value[0, :]).flatten(), (np.array(u_mpc.value[1, :]).flatten())))
        u_sim[:, sim_step] = u_bar[:, 0]

        # Update the action
        speed += u_bar[:, 0][0] * DELTA_TIME
        steering_angle = u_bar[:, 0][1]

        # Update the steering angle
        steering_angle_delta = (steering_angle - prev_steer) / DELTA_TIME
        prev_steer = steering_angle


        """"----- OBSTACLE AVOIDANCE -----"""
        # Breaking distance
        d_crit = 1.5 * speed**2 / (2*mu*g) + 0.5
        d_safe = 2 * speed**2 / (2*mu*g)
        pos = robots[0].state["joint_state"]["position"]

        # Vehicle dimensions for collision on each axis
        x_size = abs(vehicle_width / 2 * np.sin(pos[2])) + abs(vehicle_length / 2 * np.cos(pos[2]))
        y_size = abs(vehicle_width / 2 * np.cos(pos[2])) + abs(vehicle_length / 2 * np.sin(pos[2]))

        # Check distances with every robot
        for robot in robots[1:]:
            robot_x = robot.state["joint_state"]["position"][0]
            robot_y = robot.state["joint_state"]["position"][1]

            delta_x = robot_x - pos[0]
            delta_y = robot_y - pos[1]

            x_min = -vehicle_width/2
            y_min = -vehicle_length/2

            x_max = vehicle_width/2
            y_max = vehicle_length/2

            # Get closest distance from point to rectangle
            d = get_dist_point_rect(delta_x, delta_y, x_min, y_min, x_max, y_max)
            d -= robot.width

            if d < d_crit:
                print('Warning, impending collision')
                speed = 0
                steering_angle_delta = 0

        """"CHANGE ME PLS"""
        # PLEASE CHANGE, TEMP VARIABLE FOR THE MOVING OBSTACLE
        second_speed = 1.1 if int(sim_step/25) % 2 else -1.1

        # Execute the optimized action on the agent
        ob, _, _, _ = env.step([speed, steering_angle_delta, second_speed, 0])

        # Update the robot's configuration on the file
        pos = robots[0].state["joint_state"]["position"]

        x_sim[0, sim_step + 1] = pos[0]
        x_sim[1, sim_step + 1] = pos[1]
        x_sim[2, sim_step + 1] = speed
        x_sim[3, sim_step + 1] = pos[2]

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

    plot_trajectory.plot(path, environments, environment_id, obstacle_coordinates, obstacles_dimensions, x_sim, u_sim)

if __name__ == "__main__":
    MAKE_ANIMATION = False
    environment_id = 3
    
    environment_dict = environments[environment_id]

    obstacle_coordinates = environments[environment_id]["obstacle_coordinates"] + environments[environment_id]["boundary_coordinates"]
    obstacle_dimensions = environments[environment_id]["obstacle_dimensions"] + environments[environment_id]["boundary_dimensions"]

    run_env(obstacle_coordinates, obstacle_dimensions, environment_id, [[-13, -8, np.pi, 0.5, 0.5]], render=True)

    #TODO: Add obstacles to performance plot