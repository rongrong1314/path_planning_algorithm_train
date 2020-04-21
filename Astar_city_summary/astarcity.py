#you will find a path use A* in city
import numpy as np
import matplotlib.pyplot as plt
from grid import create_grid
from planning import a_star

#extend C-space and show grid
plt.rcParams['figure.figsize'] = 12,12
filename = 'colliders.csv'
data = np.loadtxt(filename,delimiter=',',dtype='Float64',skiprows = 2)
print(data)
drone_altitude = 5
safe_distance = 3
grid = create_grid(data,drone_altitude,safe_distance)
plt.imshow(grid,origin='lower')
plt.xlabel('EAST')
plt.ylabel('NORTH')
plt.show()

#compute the path use A*
start_ne = (25,100)
goal_ne = (750.,370.)

#write a heuristic function use Manhattan distance
def heuristic_func(position,goal_position):
    return np.abs(position[0] - goal_position[0]) + np.abs(position[1]-goal_position[1])
#compute the lower cost
path,cost = a_star(grid,heuristic_func,start_ne,goal_ne)

#plot the path
print(len(path),cost)
plt.imshow(grid, cmap='Greys', origin='lower')
#for the purposes of the visual the east coordinate lay along
#the x-axis
plt.plot(start_ne[1],start_ne[0],'x')
plt.plot(goal_ne[1], goal_ne[0], 'x')

pp = np.array(path)
plt.plot(pp[:,1],pp[:,0],'g')

plt.xlabel('EAST')
plt.ylabel('NORTH')
plt.show()

#unfortunately this path is impractical
#we can consider a new waypoint when the drone's direction change

#path pruning
def point(p):
    return np.array([p[0],p[1],1.]).reshape(1,-1)
def collinearity_check(p1,p2,p3,epsilon = 1e-6):
    #stack the array by rows
    m=np.concatenate((p1,p2,p3),0)
    #compute the value of the determinant
    det = np.linalg.det(m)
    return abs(det)<epsilon
#we use collinearity here,but you could use bresenham as well
def prune_path(path):
    pruned_path = [p for p in path]
    i = 0
    while i< len(pruned_path)-2:
        p1 = point(pruned_path[i])
        p2 = point(pruned_path[i+1])
        p3 = point(pruned_path[i+2])

        if collinearity_check(p1,p2,p3):
            pruned_path.remove(pruned_path[i+1])
        else:
            i+=1
    return pruned_path
pruned_path = prune_path(path)
print(len(pruned_path))

plt.imshow(grid, cmap='Greys', origin='lower')

plt.plot(start_ne[1], start_ne[0], 'x')
plt.plot(goal_ne[1], goal_ne[0], 'x')

pp = np.array(pruned_path)
plt.plot(pp[:, 1], pp[:, 0], 'g')
plt.scatter(pp[:, 1], pp[:, 0])

plt.xlabel('EAST')
plt.ylabel('NORTH')

plt.show()


