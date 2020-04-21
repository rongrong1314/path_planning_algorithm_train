import numpy as np
import matplotlib.pyplot as plt
from grid import create_grid
from skimage.morphology import medial_axis
from skimage.util import invert
from planning import a_star
plt.rcParams['figure.figsize'] = 12,12

filename = 'colliders.csv'
data = np.loadtxt(filename,delimiter=',',dtype='Float64',skiprows=2)
print(data)
start_ne = (25,100)
goal_ne = (650,500)
drone_altitude = 5
safety_distance = 2
grid = create_grid(data,drone_altitude,safety_distance)
#medial axis algorithm
skeleton = medial_axis(invert(grid))

#Search neighborhood
def find_start_goal(skel,start,goal):
    #Index non-0 elements,exchange the order
    skel_cells = np.transpose(skel.nonzero())
    #compute 2-norm by row,The position of the minimum
    start_min_dist = np.linalg.norm(np.array(start)-np.array(skel_cells),axis=1).argmin()
    near_start = skel_cells[start_min_dist]
    goal_min_dist = np.linalg.norm(np.array(goal)-np.array(skel_cells),axis=1).argmin()
    near_goal = skel_cells[goal_min_dist]
    return near_start,near_goal

skel_start, skel_goal = find_start_goal(skeleton, start_ne, goal_ne)
print(start_ne,goal_ne)
print(skel_start,skel_goal)

def heuristic_func(position, goal_position):
    return np.sqrt((position[0] - goal_position[0])**2 + (position[1] - goal_position[1])**2)

#A* on the skeleton
path,cost = a_star(invert(skeleton).astype(np.int),heuristic_func,tuple(skel_start),tuple(skel_goal))
print("Path length = {0}, path cost = {1}".format(len(path),cost))

#compare to regular A* on the gird
path2, cost2 = a_star(grid, heuristic_func, start_ne, goal_ne)
print("Path length = {0}, path cost = {1}".format(len(path2), cost2))

plt.imshow(grid,origin='lower')
plt.imshow(skeleton,cmap='Greys',origin='lower',alpha=0.7)
plt.plot(start_ne[1],start_ne[0],'rx')
plt.plot(goal_ne[1],goal_ne[0],'rx')

pp = np.array(path)
plt.plot(pp[:, 1], pp[:, 0], 'g')
pp2 = np.array(path2)
plt.plot(pp2[:, 1], pp2[:, 0], 'r')

plt.xlabel('EAST')
plt.ylabel('NORTH')
plt.show()
