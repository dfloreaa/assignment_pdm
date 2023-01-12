import numpy as np
import math
import vectors

def obstacle_avoid(n_steps, history, obstacles):
    if obstacles == True:
        prius_pos = []
        prius_vel = []
        obstacle_0_pos = []
        obstacle_0_vel = []
        obstacle_1_pos = []
        obstacle_1_vel = []
        obstacle_2_pos = []
        obstacle_2_vel = []
        
        dist_0 = np.zeros(n_steps)
        dist_1 = np.zeros(n_steps)
        dist_2 = np.zeros(n_steps)
        
        prius_speed = np.zeros(n_steps)
        obstacle_0_speed = np.zeros(n_steps)
        obstacle_1_speed = np.zeros(n_steps)
        obstacle_2_speed = np.zeros(n_steps)
        
        d_safe = np.zeros(n_steps)
        d_crit = np.zeros(n_steps)
        
        vel_angle_0 = np.zeros(n_steps)
        vel_angle_1 = np.zeros(n_steps)
        vel_angle_2 = np.zeros(n_steps)
        
        mu = 0.7 # Typical friction coefficient for a patterned tire in dry conditions
        g = 9.81 # Standard gravity
    
        for i in range(n_steps):
            # Store required information for obstacle avoidance
            prius_pos.append(history[i]['robot_0']['joint_state']['position'])
            prius_vel.append(history[i]['robot_0']['joint_state']['velocity'])
            obstacle_0_pos.append(history[i]['robot_0']['obstacle_0']['pose']['position'])
            obstacle_0_vel.append(history[i]['robot_0']['obstacle_0']['twist']['linear'])
            obstacle_1_pos.append(history[i]['robot_0']['obstacle_1']['pose']['position'])
            obstacle_1_vel.append(history[i]['robot_0']['obstacle_1']['twist']['linear'])
            obstacle_2_pos.append(history[i]['robot_0']['obstacle_2']['pose']['position'])
            obstacle_2_vel.append(history[i]['robot_0']['obstacle_2']['twist']['linear'])
        
            # Compute distance between vehicle and moving obstacles
            dist_0[i] = np.sqrt((prius_pos[i][0] - obstacle_0_pos[i][0])**2 + (prius_pos[i][1] - obstacle_0_pos[i][1])**2)
            dist_1[i] = np.sqrt((prius_pos[i][0] - obstacle_1_pos[i][0])**2 + (prius_pos[i][1] - obstacle_1_pos[i][1])**2)
            dist_2[i] = np.sqrt((prius_pos[i][0] - obstacle_2_pos[i][0])**2 + (prius_pos[i][1] - obstacle_2_pos[i][1])**2)
        
            # Speed of Prius and moving obstacles
            prius_speed[i] = np.sqrt(prius_vel[i][0]**2 + prius_vel[i][1]**2)
            obstacle_0_speed[i] = np.sqrt(obstacle_0_vel[i][0]**2 + obstacle_0_vel[i][0]**2)
            obstacle_1_speed[i] = np.sqrt(obstacle_1_vel[i][0]**2 + obstacle_1_vel[i][0]**2)
            obstacle_2_speed[i] = np.sqrt(obstacle_2_vel[i][0]**2 + obstacle_2_vel[i][0]**2)
        
            # Breaking distance
            d_crit[i] = 1.1 * prius_speed[i]**2 / (2*mu*g)
            d_safe[i] = 2 * prius_speed[i]**2 / (2*mu*g)

            # Velocities in ground plane
            prius_vel2 = np.array([prius_vel[i][0], prius_vel[i][1]])
            obstacle_0_vel2 = np.array([obstacle_0_vel[i][0], obstacle_0_vel[i][1]])
            obstacle_1_vel2 = np.array([obstacle_1_vel[i][0], obstacle_1_vel[i][1]])
            obstacle_2_vel2 = np.array([obstacle_2_vel[i][0], obstacle_2_vel[i][1]])

            # Angles between velocities
            if prius_speed[i] == 0 or obstacle_0_speed[i] == 0:
                vel_angle_0[i] = 0
            else:
                vel_angle_0[i] = vectors.angle_between(prius_vel2, obstacle_0_vel2)
        
            if prius_speed[i] == 0 or obstacle_1_speed[i] == 0:
                vel_angle_1[i] = 0
            else:
                vel_angle_1[i] = vectors.angle_between(prius_vel2, obstacle_1_vel2)
        
            if prius_speed[i] == 0 or obstacle_2_speed[i] == 0:
                vel_angle_2[i] = 0
            else:
                vel_angle_2[i] = vectors.angle_between(prius_vel2, obstacle_2_vel2)

            # Collision avoidance
            if dist_0[i] < d_crit[i] or dist_1[i] < d_crit[i] or dist_2[i] < d_crit[i]:
                print('Warning, impending collision', i)
            if dist_0[i] < d_safe[i] or dist_1[i] < d_safe[i] or dist_2[i] < d_safe[i] and 0.25*math.pi <= vel_angle_0[i] <= 1.75*math.pi or 0.25*math.pi <= vel_angle_1[i] <= 1.75*math.pi or 0.25*math.pi <= vel_angle_2[i] <= 1.75*math.pi:
                print('Warning, possibly impending collision', i)
            else:
                print(i)