#the obstacle data in .csv which consists of
# six columns x,y,z and 𝛿𝑥, 𝛿𝑦,  𝛿𝑧
#.csv first line gives the map center coordinates
#each (x,y,z) coordinate is the center of the obstacle
#𝛿𝑥, 𝛿𝑦,  𝛿𝑧 are the half widths of obstacle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12,12]
#read the cvs file which contains the coordinates of the obstacle
filename = 'colliders.csv'
#note:skip first two lines
data = np.loadtxt(filename,delimiter=',',dtype = 'Float64',skiprows=2)
print(data)

#static drone altitude
drone_altitude = 10
#minimum safe distance,think of this as padding around the obstacle
safe_distance = 3

def create_grid(data,dron_altitude,safety_distance):

    ##-- generate the minimum grid according to the obstacles
    #minimum and maximum north coordinates
    #posX-halfSizeX ,round-off number toward down
    north_min = np.floor(np.min(data[:,0]-data[:,3]))
    #posX-halfSizeX ,round-off number toward up
    north_max = np.ceil(np.max(data[:,0]+data[:,3]))
    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    #initialize an empty grid
    grid = np.zeros((north_size,east_size))

    #populate the grid with obstacles
    for i in range(data.shape[0]):
        north,east,alt,d_north,d_east,d_alt = data[i,:]
        #set the obstacle higher than drone altitude
        if alt +d_alt + safety_distance > drone_altitude:
            obstacle = [
                #data,min,max
                #local = global ± width ± safe_dis ± boundary
                int(np.clip(north-d_north-safety_distance-north_min,0,north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size - 1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size - 1))
            ]
            grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = 1
    return grid

grid = create_grid(data,drone_altitude,safe_distance)
plt.imshow(grid,origin='lower')
plt.xlabel('EAST')
plt.ylabel('NORTH')
plt.show()