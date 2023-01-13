import os
import numpy as np
from RRT import pathComputation


# Environment consists of obstacles coordinates and dimensions
def generatePath(environment_dict, environment_id, n_iter = 20000, make_animation = False, directory = None):
    print("Generating path for environment: ", environment_id)

    obstacle_coordinates = environment_dict["obstacle_coordinates"] + environment_dict["boundary_coordinates"]
    obstacle_dimensions = environment_dict["obstacle_dimensions"] + environment_dict["boundary_dimensions"]

    startpos = environment_dict["startpos"]
    endpos = environment_dict["endpos"]
    
    
    path = pathComputation(obstacle_coordinates, obstacle_dimensions, environment_id, 
                            startpos, endpos, n_iter, make_animation= make_animation)

    if path is None:
        print("No path found for the current map")
        path = []

    if directory is None:
        # save the path
        if not os.path.exists("./paths"):
            os.makedirs("./paths")
        
        np.save("./paths/path{}.npy".format(environment_id), np.array(path))
        print(path)

    else:
        np.save(directory, np.array(path))

    return path