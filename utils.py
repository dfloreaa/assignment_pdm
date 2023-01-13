import numpy as np

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

def get_dist_point_rect(delta_x, delta_y, x_min, y_min, x_max, y_max):
    """"Compute the distance between a point and the closest point inside the perimeter of the rectangle"""
    d = 0.0

    if (delta_x < x_min):
        if (delta_y <  y_min):
            d = np.hypot(x_min-delta_x, y_min-delta_y)
        elif (delta_y <= y_max):
            d = x_min - delta_x
        else:
            d = np.hypot(x_min-delta_x, y_max-delta_y)

    elif (delta_x <= x_max):
        if (delta_y <  y_min):
            d = y_min - delta_y
        elif (delta_y <= y_max):
            d = 0
        else:
            d = delta_y - y_max

    else:
        if (delta_y <  y_min):
            d = np.hypot(x_max-delta_x, y_min-delta_y)
        elif (delta_y <= y_max):
            d = delta_x - x_max
        else:
            d = np.hypot(x_max-delta_x, y_max-delta_y)
    
    return d