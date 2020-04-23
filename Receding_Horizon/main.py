import numpy as np
import matplotlib.pyplot as plt
#grid creation routine
from grid import create_grid
#voxel map creation routine
from voxmap import create_voxmap
from planning import a_star
from PRM import create_prm
from mpl_toolkits.mplot3d import Axes3D

start_ne = (25,100)
goal_ne = (750.,370.)

plt.rcParams['figure.figsize'] = 14,14
filename = 'colliders.csv'
data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
print(data)
flight_altitude = 20
safety_distance = 1
grid = create_grid(data, flight_altitude, safety_distance)
#generate PRM points
nodes,g = create_prm(data)
voxmap = create_voxmap(data,10)
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.voxels(voxmap,edgecolors='k')
nmin = np.min(data[:, 0])
emin = np.min(data[:, 1])
#draw edges
for (n1,n2) in g.edges:
    ax.plot([(n1[1]-emin)/10,(n2[1]-emin)/10],[(n1[0]-nmin)/10,(n2[0]/10-nmin)/10],[n1[2],n2[2]],'red',alpha=1)


plt.show()
