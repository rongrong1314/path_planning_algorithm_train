import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 12,12
#the function should return the list of gird cells required to draw the line
def bres(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    cells = []

    dx,dy = x2-x1,y2-y1
    d = 0
    i = x1
    j = y1
    while i<x2 and j<y2:
        cells.append([i,j])
        if d<dx-dy:
            d+=dy
            i+=1
        elif d == dx-dy:
            d += dy
            i += 1
            d -= dx
            j += 1
        else:
            d -= dx
            j += 1
    return np.array(cells)

#start and end points
p1 = (0,0)
p2 = (5,5)
cells = bres(p1,p2)
plt.figure(1)
plt.plot([p1[0],p2[0]],[p1[1],p2[1]])

for q in cells:
    plt.plot([q[0],q[0]+1],[q[1],q[1]],'k')
    plt.plot([q[0], q[0] + 1], [q[1] + 1, q[1] + 1], 'k')
    plt.plot([q[0], q[0]], [q[1], q[1] + 1], 'k')
    plt.plot([q[0] + 1, q[0] + 1], [q[1], q[1] + 1], 'k')
plt.grid()
plt.axis('equal')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Integer based Bresenham algorithm")
plt.show()

from bresenham import bresenham
line = (0, 0, 7, 5)

cells = list(bresenham(line[0], line[1], line[2], line[3]))
print(cells)


plt.figure(2)
plt.plot([line[0], line[2]], [line[1], line[3]])


for q in cells:
    plt.plot([q[0], q[0]+1], [q[1], q[1]], 'k')
    plt.plot([q[0], q[0]+1], [q[1]+1, q[1]+1], 'k')
    plt.plot([q[0], q[0]], [q[1],q[1]+1], 'k')
    plt.plot([q[0]+1, q[0]+1], [q[1], q[1]+1], 'k')

plt.grid()
plt.axis('equal')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Python package Bresenham algorithm")
plt.show()