# Obstacle avoidance for a Prius in a gym environment

## Description
In this package, a planning algorithm is implemented for a simulated prius in a gym environment. This prius is an asset of the generic urdf robot packages provided on github.

## Pre-requisites
- Python>3.6, <3.10. *Note that only python versions 3.8 and 3.9 have been tested*
- pip3 
- Installation of the `gym\_envs\_urdf` from [the github repository](https://github.com/maxspahn/gym_envs_urdf)

In addition the following packages need to be installed:
- cvxpy
- scipy
- numpy
- matplotlib

## Installation and usage
Clone the repo. Next run the main.py file. Note that in the main file, it is possible to select an environment by changing the `environment_id`.
On first run of every environment, the algorithm will first create a path for that specific environment; this will take some time. Every following run of that environment will use the same path again. If you want to create a new path, change the global variable `MAKE_ANIMATION` to true

## Output
After running the main file, 3 important files will be created. Firstly, an animation will be created, showing the exploration of the workspace using RRT. Next, an image will be created, showing the shortest path that will be used for the MPC algorithm. Lastly, the performance can be evaluated in the performance plot, showing the actual path of prius alongside the optimal path, and other important metrics.
These three files can be found in the following directories, respectively:
- `animation/animation#`
- `graph/graph#`
- `performance/performance#`

with # being the environment ID.

## Environments
The following environments have been defined for the Prius. These can be selected in the main file by chanign `enviornment_id` to one of the following values `{0, 1, 2, 3}`. 
<table>
 <tr>
  <td> Environment 0 </td>
  <td> Environment 1 </td>
 </tr>
 <tr>
  <td> <img src = Environment0.png width="250" height="250"/> </td>
  <td> <img src = Environment1.png width="250" height="250"/> </td>  

</table>

<table>
 <tr>
  <td> Environment 2 </td>
  <td> Environment 3 </td>
 </tr>
 <tr>
  <td> <img src = Environment2.png width="250" height="250"/> </td>
  <td> <img src=Environment3.png width="250" height="250"/> </td>
 </tr>
</table>
