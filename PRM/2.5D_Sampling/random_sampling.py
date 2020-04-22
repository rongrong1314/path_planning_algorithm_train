import time
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon,Point
plt.rcParams['figure.figsize'] = 12,12

filename = 'colliders.csv'
data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
print(data)
print(data.shape)
print(np.max(data[:,4]))

def extract_polygons(data):
    polygons =[]
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        #Extract the 4 corners of each obstacle
        obstacle = [north - d_north,north+d_north,east-d_north,east+d_north]
        corners = [(obstacle[0], obstacle[2]),(obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])]
        height = alt +d_alt
        p = Polygon(corners)
        polygons.append((p,height))
    return polygons

polygons = extract_polygons(data)
print(len(polygons))
xmin = np.min(data[:,0] - data[:,3])
xmax = np.max(data[:,0] + data[:,3])
ymin = np.min(data[:, 1] - data[:, 4])
ymax = np.max(data[:, 1] + data[:, 4])
zmin = 0
zmax = 10
print("X")
print("min = {0}, max = {1}\n".format(xmin, xmax))

print("Y")
print("min = {0}, max = {1}\n".format(ymin, ymax))

print("Z")
print("min = {0}, max = {1}".format(zmin, zmax))

#sampling
num_samples = 100
xvals = np.random.uniform(xmin, xmax, num_samples)
yvals = np.random.uniform(ymin, ymax, num_samples)
zvals = np.random.uniform(zmin, zmax, num_samples)
samples = np.array(list(zip(xvals, yvals, zvals)))

def collides(polygons,point):
    for (p,height) in polygons:
        #check collide
        if p.contains(Point(point)) and height >= point[2]:
            return True
    return False

t0 = time.time()
to_keep = []
for point in samples:
    if not collides(polygons,point):
        to_keep.append(point)
time_taken = time.time() - t0
print("time taken {0} seconds ...".format(time_taken))
print(len(to_keep))

from grid import create_grid
grid = create_grid(data,zmax,1)
fig = plt.figure()
plt.imshow(grid, cmap='Greys', origin='lower')
nmin = np.min(data[:,0])
emin = np.min(data[:,1])
#draw points
all_pts = np.array(to_keep)
north_vals =all_pts[:,0]
east_vals = all_pts[:,1]
plt.scatter(east_vals -emin,north_vals-nmin,c='red')
plt.ylabel('NORTH')
plt.xlabel('EAST')
plt.show

#Note:use a k-d tree can save the time
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree