import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = 16,16

filename = 'colliders.csv'
data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
print(data)

def create_voxmap(data,voxel_size = 5):
    # minimum and maximum north coordinates
    north_min = np.floor(np.amin(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.amax(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.amin(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.amax(data[:, 1] + data[:, 4]))

    #maximum altitude
    alt_max = np.ceil(np.amax(data[:,2]+data[:,5]))
    #given the minimu and maximum coordinates
    #we can caculate the size of the grid
    #//round down
    north_size = int(np.ceil(north_max-north_min))//voxel_size
    east_size = int(np.ceil(east_max-east_min))//voxel_size
    alt_size = int(alt_max)//voxel_size
    #create an empty grid
    voxmap = np.zeros((north_size,east_size,alt_size),dtype=np.bool)
    #fill in the voxels obstacle with true
    for i in range(data.shape[0]):
        north,east,alt,d_north,d_east,d_alt = data[i,:]
        obstacle = [
            int(north - d_north-north_min) // voxel_size,
            int(north + d_north-north_min) // voxel_size,
            int(east - d_east - east_min)  //voxel_size,
            int(east + d_east - east_min) //voxel_size,
        ]

        height = int(alt+d_alt) // voxel_size
        voxmap[obstacle[0]:obstacle[1],obstacle[2]:obstacle[3],0:height] = True
    return voxmap

voxmap = create_voxmap(data,10)
print(voxmap.shape)
#plot the 3D grid
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.voxels(voxmap,edgecolors='k')
ax.set_xlim(voxmap.shape[0],0)
ax.set_ylim(0,voxmap.shape[1])
ax.set_zlim(0,voxmap.shape[2]+20)
plt.xlabel('North')
plt.ylabel('East')
plt.show()