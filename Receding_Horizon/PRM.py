import numpy as np
from sampling import Sampler
from shapely.geometry import LineString
import networkx as nx
from sklearn.neighbors import KDTree

def create_prm(data):
    sampler = Sampler(data)
    polygons = sampler._polygons
    nodes = sampler.sample(300)
    g = create_graph(nodes, 10,polygons)
    return nodes,g

def can_connect(n1,n2,poly):
    l = LineString([n1,n2])
    for p in poly:
        #check collide
        if p.crosses(l) and p.height >= min(n1[2],n2[2]):
            return False
    return True

def create_graph(nodes,k,poly):
    g = nx.Graph()
    tree = KDTree(np.array(nodes))
    for n1 in nodes:
        #for each node connect try to connect to k nearest nodes
        idxs = tree.query([n1],k,return_distance=False)[0]
        for idx in idxs:
            n2 = nodes[idx]
            if n2 == n1:
                continue

            if can_connect(n1,n2,poly):
                g.add_edge(n1,n2,weight=1)
    return g