import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sampling import Sampler
from shapely.geometry import  Polygon,Point,LineString
from queue import PriorityQueue
plt.rcParams['figure.figsize'] = 12,12

filename = 'colliders.csv'
data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
print(data)
sampler = Sampler(data)
polygons = sampler._polygons
nodes = sampler.sample(300)
print(len(nodes))

import numpy.linalg as LA
from sklearn.neighbors import KDTree

def can_connect(n1,n2):
    l = LineString([n1,n2])
    for p in polygons:
        #check collide
        if p.crosses(l) and p.height >= min(n1[2],n2[2]):
            return False
    return True

def create_graph(nodes,k):
    g = nx.Graph()
    tree = KDTree(np.array(nodes))
    for n1 in nodes:
        #for each node connect try to connect to k nearest nodes
        idxs = tree.query([n1],k,return_distance=False)[0]
        for idx in idxs:
            n2 = nodes[idx]
            if n2 == n1:
                continue

            if can_connect(n1,n2):
                g.add_edge(n1,n2,weight=1)
    return g
import time
t0 = time.time()
g = create_graph(nodes,10)
print('graph took {0} seconds to build'.format(time.time()-t0))
print("Number of edges",len(g.edges))

from grid import create_grid
grid = create_grid(data,sampler._zmax,1)

fig = plt.figure()

plt.imshow(grid, cmap='Greys', origin='lower')

nmin = np.min(data[:, 0])
emin = np.min(data[:, 1])

#draw edges
for (n1,n2) in g.edges:
    plt.plot([n1[1]-emin,n2[1]-emin],[n1[0]-nmin,n2[0]-nmin],'black',alpha=0.5)

# draw all nodes
for n1 in nodes:
    plt.scatter(n1[1] - emin, n1[0] - nmin, c='blue')

# draw connected nodes
for n1 in g.nodes:
    plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')

plt.xlabel('NORTH')
plt.ylabel('EAST')

plt.show()